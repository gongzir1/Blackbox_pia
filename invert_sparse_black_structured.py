"""Adaptive block-coordinate, query-only sparse text inversion.

Reuses the original attack's model, dictionary, oracle, decoding and metrics.
Groups are one token row and an interval of dictionary atoms. Atom index order
is only a reproducible initial partition, not a claim of semantic similarity.
No target tokens, private weights or target gradients enter the optimizer.
"""

from dataclasses import dataclass
import math

import torch

import invert_sparse_black as black


@dataclass
class Group:
    row: int
    start: int
    stop: int
    score: float = 0.0
    visits: int = 0
    last_visit: int = -1


def initial_groups(rows, width, size):
    return [Group(row, start, min(start + size, width))
            for row in range(rows) for start in range(0, width, size)]


def select_groups(groups, count, exploration_fraction, selection):
    """Coverage first; then exploit EMA scores and revisit oldest groups."""
    count = min(count, len(groups))
    if selection == "uniform":
        return [groups[i] for i in torch.randperm(len(groups))[:count].tolist()]
    oldest = sorted(groups, key=lambda group: (group.visits > 0, group.last_visit))
    explore_count = min(count, max(1, math.ceil(count * exploration_fraction)))
    selected = oldest[:explore_count]
    selected_ids = {id(group) for group in selected}
    others = sorted((group for group in groups if id(group) not in selected_ids),
                    key=lambda group: (group.visits == 0, group.score), reverse=True)
    return selected + others[:count - explore_count]


def split_groups(groups, selected, minimum_size, minimum_visits, max_splits):
    eligible = [group for group in selected
                if group.visits >= minimum_visits
                and group.stop - group.start >= 2 * minimum_size]
    chosen = {id(group) for group in sorted(
        eligible, key=lambda group: group.score, reverse=True)[:max_splits]}
    result = []
    for group in groups:
        if id(group) not in chosen:
            result.append(group)
            continue
        mid = (group.start + group.stop) // 2
        # Children must obtain their own measurements before further splitting.
        result.extend([Group(group.row, group.start, mid, group.score),
                       Group(group.row, mid, group.stop, group.score)])
    return result


def block_gradient(directions, responses, probes_per_group):
    """Sum disjoint block estimates, averaging only repeated probes per block."""
    return (responses[:, None, None] * directions).sum(dim=0) / probes_per_group


def optimize_coefficients(args, oracle, dictionary, target_hidden, attention_mask):
    rows, width = attention_mask.shape[1], dictionary.shape[1]
    alpha = torch.randn(rows, width, device=target_hidden.device) * args.alpha_init_std
    groups = initial_groups(rows, width, args.group_size)
    first = torch.zeros_like(alpha)
    second = torch.zeros_like(alpha)
    steps = torch.zeros_like(alpha)
    history = []
    best_alpha, best_loss = alpha.clone(), float("inf")
    exhausted = False
    initial_budget = oracle.remaining_queries

    for iteration in range(args.iterations):
        # Query-based schedules allow different numbers of blocks per iteration.
        progress = min(1.0, oracle.queries / max(initial_budget - 2, 1))
        lr = args.alpha_lr * ((args.alpha_lr_final or args.alpha_lr) / args.alpha_lr) ** progress
        mu = args.perturbation_radius * (
            (args.perturbation_radius_final or args.perturbation_radius)
            / args.perturbation_radius) ** progress
        center_cost = int(args.return_best_alpha)
        available = (oracle.remaining_queries - center_cost) // 2
        count = min(args.groups_per_step, available // args.probes_per_group, len(groups))
        if count < 1:
            exhausted = True
            break
        if args.return_best_alpha:
            loss = black.state_distances(oracle.query(
                black.candidate_embeddings(alpha, dictionary)[None].to(torch.float16),
                attention_mask), target_hidden, args.hidden_mse_weight).item()
            if loss < best_loss:
                best_loss, best_alpha = loss, alpha.clone()
        selected = select_groups(groups, count, args.exploration_fraction, args.group_selection)
        probe_count = count * args.probes_per_group
        directions = torch.zeros(probe_count, rows, width, device=alpha.device)
        active = torch.zeros_like(alpha, dtype=torch.bool)
        for index, group in enumerate(selected):
            sl = slice(index * args.probes_per_group, (index + 1) * args.probes_per_group)
            directions[sl, group.row, group.start:group.stop] = black.rademacher_directions(
                args.probes_per_group, (group.stop - group.start,), device=alpha.device)
            active[group.row, group.start:group.stop] = True
        mask = attention_mask.expand(probe_count, -1)
        plus = black.state_distances(oracle.query(black.candidate_embeddings(
            alpha[None] + mu * directions, dictionary).to(torch.float16), mask),
            target_hidden, args.hidden_mse_weight)
        minus = black.state_distances(oracle.query(black.candidate_embeddings(
            alpha[None] - mu * directions, dictionary).to(torch.float16), mask),
            target_hidden, args.hidden_mse_weight)
        responses = (plus - minus) / (2 * mu)
        gradient = block_gradient(directions, responses, args.probes_per_group)
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("Non-finite structured gradient")
        for index, group in enumerate(selected):
            local = responses[index * args.probes_per_group:(index + 1) * args.probes_per_group]
            # RMS / sqrt(size) estimates per-coordinate gradient strength.
            score = (local.square().mean() / (group.stop - group.start)).sqrt().item()
            group.score = (args.score_decay * group.score + (1 - args.score_decay) * score
                           if group.visits else score)
            group.visits += 1
            group.last_visit = iteration
        update = gradient[active]
        if args.alpha_optimizer == "adam":
            first[active] = args.adam_beta1 * first[active] + (1 - args.adam_beta1) * update
            second[active] = args.adam_beta2 * second[active] + (1 - args.adam_beta2) * update.square()
            steps[active] += 1
            update = (first[active] / (1 - args.adam_beta1 ** steps[active])) / (
                (second[active] / (1 - args.adam_beta2 ** steps[active])).sqrt() + args.adam_eps)
        # Unqueried coordinates are neither moved by momentum nor shrunk.
        alpha[active] = black.soft_threshold(alpha[active] - lr * update, lr * args.alpha_l1)
        if args.alpha_clip > 0:
            alpha[active] = alpha[active].clamp(-args.alpha_clip, args.alpha_clip)
        if (iteration + 1) % args.split_interval == 0 and args.group_selection == "adaptive":
            groups = split_groups(groups, selected, args.min_group_size,
                                  args.split_min_visits, args.splits_per_interval)
        history.append({"iteration": iteration + 1, "queries": oracle.queries,
                        "estimated_hidden_loss": ((plus + minus) / 2).mean().item(),
                        "groups": len(groups), "selected_groups": count,
                        "directions": probe_count, "alpha_lr": lr,
                        "perturbation_radius": mu,
                        "active_fraction": active.float().mean().item(),
                        "nonzero_fraction": (alpha != 0).float().mean().item()})
        if (iteration + 1) % args.log_interval == 0:
            print(f"step={iteration + 1} queries={oracle.queries} groups={len(groups)} "
                  f"probe_loss={history[-1]['estimated_hidden_loss']:.6f}", flush=True)
    if args.return_best_alpha:
        alpha = best_alpha
    return alpha, history, exhausted


def build_parser():
    parser = black.build_parser()
    parser.description = __doc__
    parser.set_defaults(output_dir="results_black_structured", iterations=100000)
    parser.add_argument("--group-size", type=int, default=512)
    parser.add_argument("--min-group-size", type=int, default=64)
    parser.add_argument("--groups-per-step", type=int, default=8)
    parser.add_argument("--probes-per-group", type=int, default=4)
    parser.add_argument("--group-selection", choices=("adaptive", "uniform"), default="adaptive")
    parser.add_argument("--exploration-fraction", type=float, default=0.25)
    parser.add_argument("--score-decay", type=float, default=0.8)
    parser.add_argument("--split-interval", type=int, default=20)
    parser.add_argument("--split-min-visits", type=int, default=4)
    parser.add_argument("--splits-per-interval", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=50)
    return parser


def validate_args(args):
    if args.representation != "sparse" or args.alpha_top_k != 0:
        raise ValueError("Structured attack requires sparse representation and alpha-top-k=0")
    for name in ("group_size", "min_group_size", "groups_per_step", "probes_per_group",
                 "split_interval", "split_min_visits", "splits_per_interval", "log_interval"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if not 0 < args.exploration_fraction <= 1 or not 0 <= args.score_decay < 1:
        raise ValueError("Require exploration-fraction in (0,1] and score-decay in [0,1)")
    if args.alpha_l1 < 0:
        raise ValueError("alpha-l1 must be non-negative")


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    validate_args(arguments)
    black.run(arguments, optimizer_fn=optimize_coefficients,
              method_name="black_box_sparse_structured_recovery")
