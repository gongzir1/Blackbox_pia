"""Query-only sparse inversion using per-position hidden-state losses.

One oracle response already contains every position's state. Each row uses its
own position's loss response instead of the sequence-averaged response. This
estimates diagonal blocks of the loss Jacobian, not the full gradient of the
original sequence loss. In a causal model, upstream perturbations remain noise;
there is no assumption that token positions are independent.
"""
import torch
import torch.nn.functional as F

import invert_sparse_black as black


def position_distances(candidate, target, mse_weight):
    candidate, target = candidate.float(), target.float()
    cosine = 1 - (F.normalize(candidate, dim=-1)
                  * F.normalize(target, dim=-1)).sum(dim=-1)
    if mse_weight:
        cosine = cosine + mse_weight * (candidate - target).square().mean(dim=-1) / (
            target.square().mean(dim=-1).clamp_min(1e-12))
    return cosine


def tokenwise_gradient(directions, responses):
    return (responses[..., None] * directions).mean(dim=0)


def optimize_coefficients(args, oracle, dictionary, target_hidden, attention_mask):
    alpha = torch.randn(attention_mask.shape[1], dictionary.shape[1],
                        device=target_hidden.device) * args.alpha_init_std
    history = []
    exhausted = False
    for iteration in range(args.iterations):
        count = min(args.directions, oracle.remaining_queries // 2)
        if not count:
            exhausted = True
            break
        lr = black.geometric_schedule(args.alpha_lr, args.alpha_lr_final,
                                      iteration, args.iterations)
        mu = black.geometric_schedule(args.perturbation_radius,
                                      args.perturbation_radius_final,
                                      iteration, args.iterations)
        directions = black.rademacher_directions(count, alpha.shape, device=alpha.device)
        mask = attention_mask.expand(count, -1)
        plus = position_distances(oracle.query(black.candidate_embeddings(
            alpha[None] + mu * directions, dictionary).to(torch.float16), mask),
            target_hidden, args.hidden_mse_weight)
        minus = position_distances(oracle.query(black.candidate_embeddings(
            alpha[None] - mu * directions, dictionary).to(torch.float16), mask),
            target_hidden, args.hidden_mse_weight)
        gradient = tokenwise_gradient(directions, (plus - minus) / (2 * mu))
        if not torch.isfinite(gradient).all():
            raise FloatingPointError('Non-finite tokenwise gradient')
        alpha = black.soft_threshold(alpha - lr * gradient, lr * args.alpha_l1)
        if args.alpha_clip > 0:
            alpha = alpha.clamp(-args.alpha_clip, args.alpha_clip)
        alpha = black.keep_topk_coefficients(alpha, args.alpha_top_k)
        history.append({'iteration': iteration + 1, 'queries': oracle.queries,
                        'estimated_hidden_loss': ((plus + minus) / 2).mean().item(),
                        'directions': count, 'alpha_lr': lr, 'perturbation_radius': mu,
                        'nonzero_fraction': (alpha != 0).float().mean().item()})
        if (iteration + 1) % 50 == 0:
            print(f'step={iteration + 1} queries={oracle.queries} '
                  f'position_loss={history[-1]["estimated_hidden_loss"]:.6f}', flush=True)
    return alpha, history, exhausted


def validate_args(args):
    if args.representation != 'sparse' or args.alpha_optimizer != 'sgd':
        raise ValueError('Tokenwise variant currently requires sparse representation and SGD')
    if args.return_best_alpha:
        raise ValueError('Use --no-return-best-alpha for the tokenwise variant')
    if args.alpha_l1 < 0:
        raise ValueError('alpha-l1 must be non-negative')


if __name__ == '__main__':
    parser = black.build_parser()
    parser.description = __doc__
    parser.set_defaults(output_dir='results_black_tokenwise')
    args = parser.parse_args()
    validate_args(args)
    black.run(args, optimizer_fn=optimize_coefficients,
              method_name='black_box_sparse_tokenwise_recovery')
