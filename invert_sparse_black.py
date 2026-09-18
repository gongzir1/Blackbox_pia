"""Query-only black-box sparse prompt inversion for Llama-family models.

The private LoRA target is used only through ``BlackBoxOracle.query``: the
recovery loop never reads its parameters or requests gradients.  It implements
the two-sided finite-difference estimator and proximal soft-thresholding from
the paper's black-box sparse recovery section.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # Keep CPU-only mathematical helper tests dependency-free.
    PeftModel = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

class QueryBudgetExceeded(RuntimeError):
    """Raised when an oracle call would exceed the configured budget."""


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_base_model(model_name):
    if AutoModelForCausalLM is None:
        raise RuntimeError("Install requirements.txt to load a Hugging Face model.")
    if not torch.cuda.is_available():
        raise RuntimeError("This implementation requires a CUDA GPU.")
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        use_cache=False,
        trust_remote_code=True,
    )


def transformer_layers(model):
    base = model.get_base_model() if PeftModel is not None and isinstance(model, PeftModel) else model
    return base.model.layers


def capture_hidden(model, layer_index, *, input_ids=None, inputs_embeds=None,
                   attention_mask=None):
    layers = transformer_layers(model)
    if not 0 <= layer_index < len(layers):
        raise ValueError(
            f"split layer {layer_index} is outside [0, {len(layers) - 1}]"
        )
    captured = []

    def hook(_module, _inputs, output):
        captured.append(output[0] if isinstance(output, tuple) else output)

    handle = layers[layer_index].register_forward_hook(hook)
    kwargs = {"attention_mask": attention_mask, "use_cache": False}
    if input_ids is not None:
        kwargs["input_ids"] = input_ids
    else:
        kwargs["inputs_embeds"] = inputs_embeds
    try:
        with torch.no_grad():
            model(**kwargs)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected one split-layer capture, got {len(captured)}")
    return captured[0]


def hidden_distances(candidate, target):
    """Return cosine distances per candidate in a batch."""
    candidate = F.normalize(candidate.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    return 1.0 - (candidate * target).sum(dim=-1).mean(dim=-1)


def hidden_distance(candidate, target):
    return hidden_distances(candidate, target).mean()


def state_distances(candidate, target, mse_weight=0.0):
    """Combined cosine and scale-normalized MSE distances per candidate."""
    cosine = hidden_distances(candidate, target)
    if mse_weight == 0:
        return cosine
    squared_error = (candidate.float() - target.float()).pow(2)
    normalized_mse = squared_error.mean(dim=(-1, -2)) / target.float().pow(2).mean()
    return cosine + mse_weight * normalized_mse


def soft_threshold(values, threshold):
    return values.sign() * torch.relu(values.abs() - threshold)


def keep_topk_coefficients(values, top_k):
    """Keep only the largest-magnitude coefficients in each token row."""
    if top_k <= 0 or top_k >= values.shape[-1]:
        return values
    indices = values.abs().topk(top_k, dim=-1).indices
    mask = torch.zeros_like(values, dtype=torch.bool)
    mask.scatter_(-1, indices, True)
    return values * mask


def geometric_schedule(initial, final, iteration, total_iterations):
    """Geometrically interpolate from initial to final, including endpoints."""
    if final is None or total_iterations <= 1:
        return initial
    progress = iteration / (total_iterations - 1)
    return initial * (final / initial) ** progress


def rademacher_directions(count, shape, *, device, generator=None):
    """Sample ±1 directions, for which E[U U^T] is the identity."""
    signs = torch.randint(
        0, 2, (count, *shape), device=device, generator=generator
    )
    return signs.to(torch.float32).mul_(2).sub_(1)


class BlackBoxOracle:
    """A counted, no-gradient interface to the private upstream model."""

    def __init__(self, model, split_layer, max_queries, batch_size=None):
        self._model = model
        self._split_layer = split_layer
        self.max_queries = max_queries
        self.queries = 0
        self.batch_size = batch_size

    @property
    def remaining_queries(self):
        return self.max_queries - self.queries

    def query(self, inputs_embeds, attention_mask):
        count = inputs_embeds.shape[0]
        if count > self.remaining_queries:
            raise QueryBudgetExceeded(
                f"query needs {count} evaluations with only "
                f"{self.remaining_queries} remaining"
            )
        outputs = []
        batch_size = self.batch_size or count
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            outputs.append(capture_hidden(
                self._model, self._split_layer,
                inputs_embeds=inputs_embeds[start:stop],
                attention_mask=attention_mask[start:stop],
            ))
            self.queries += stop - start
        return torch.cat(outputs, dim=0)


def candidate_embeddings(alpha, dictionary, representation="sparse"):
    if representation == "dense":
        return alpha
    return alpha @ dictionary.T


def optimize_coefficients(args, oracle, dictionary, target_hidden, attention_mask):
    sequence_length = attention_mask.shape[1]
    width = (target_hidden.shape[-1] if args.representation == "dense"
             else dictionary.shape[1])
    alpha = torch.randn(
        sequence_length, width, device=target_hidden.device,
        dtype=torch.float32,
    ) * args.alpha_init_std
    history = []
    budget_exhausted = False
    first_moment = torch.zeros_like(alpha)
    second_moment = torch.zeros_like(alpha)
    best_alpha = alpha.clone()
    best_loss = float("inf")

    for iteration in range(args.iterations):
        perturbation_radius = geometric_schedule(
            args.perturbation_radius, args.perturbation_radius_final,
            iteration, args.iterations,
        )
        alpha_lr = geometric_schedule(
            args.alpha_lr, args.alpha_lr_final, iteration, args.iterations
        )
        direction_count = min(args.directions, oracle.remaining_queries // 2)
        if direction_count == 0:
            budget_exhausted = True
            break
        directions = rademacher_directions(
            direction_count, alpha.shape, device=alpha.device
        )
        plus = candidate_embeddings(
            alpha.unsqueeze(0) + perturbation_radius * directions, dictionary,
            args.representation,
        ).to(torch.float16)
        minus = candidate_embeddings(
            alpha.unsqueeze(0) - perturbation_radius * directions, dictionary,
            args.representation,
        ).to(torch.float16)
        probe_mask = attention_mask.expand(direction_count, -1)
        plus_loss = state_distances(
            oracle.query(plus, probe_mask), target_hidden, args.hidden_mse_weight
        )
        minus_loss = state_distances(
            oracle.query(minus, probe_mask), target_hidden, args.hidden_mse_weight
        )
        gradient = (
            ((plus_loss - minus_loss) / (2 * perturbation_radius))
            .view(direction_count, 1, 1) * directions
        ).mean(dim=0)
        if args.alpha_optimizer == "adam":
            first_moment.mul_(args.adam_beta1).add_(
                gradient, alpha=1 - args.adam_beta1
            )
            second_moment.mul_(args.adam_beta2).addcmul_(
                gradient, gradient, value=1 - args.adam_beta2
            )
            step = iteration + 1
            corrected_first = first_moment / (1 - args.adam_beta1 ** step)
            corrected_second = second_moment / (1 - args.adam_beta2 ** step)
            update = corrected_first / (corrected_second.sqrt() + args.adam_eps)
        else:
            update = gradient
        alpha = alpha - alpha_lr * update
        if args.representation == "sparse":
            alpha = soft_threshold(alpha, alpha_lr * args.alpha_l1)
        if args.alpha_clip > 0:
            alpha = alpha.clamp(-args.alpha_clip, args.alpha_clip)
        if args.representation == "sparse":
            alpha = keep_topk_coefficients(alpha, args.alpha_top_k)
        estimated_loss = float(((plus_loss + minus_loss) / 2).mean().item())
        if estimated_loss < best_loss:
            best_loss = estimated_loss
            best_alpha = alpha.clone()
        history.append({
            "iteration": iteration + 1,
            "estimated_hidden_loss": estimated_loss,
            "mean_abs_alpha": float(alpha.abs().mean().item()),
            "nonzero_fraction": float((alpha != 0).float().mean().item()),
            "directions": direction_count,
            "alpha_lr": alpha_lr,
            "perturbation_radius": perturbation_radius,
            "queries": oracle.queries,
        })

    if args.return_best_alpha:
        alpha = best_alpha
    return alpha, history, budget_exhausted


def topk_tokens(embeddings, embedding_weight, top_k):
    normalized_weight = F.normalize(embedding_weight.float(), dim=-1)
    return [
        torch.topk(
            normalized_weight @ F.normalize(embedding.float(), dim=-1), top_k
        ).indices.tolist()
        for embedding in embeddings
    ]


def token_accuracy(predicted, target):
    return sum(a == b for a, b in zip(predicted, target)) / len(target)


def refine_tokens(oracle, candidates, embedding_weight, target_hidden,
                  attention_mask, passes=1, mse_weight=0.0):
    """Greedily select token candidates using only split-state queries."""
    predicted = [items[0] for items in candidates]
    last_position = -1
    complete = True
    for _pass in range(passes):
        for position, items in enumerate(candidates):
            candidate_count = min(len(items), oracle.remaining_queries)
            if candidate_count == 0:
                complete = False
                break
            trial_ids = []
            for token_id in items[:candidate_count]:
                trial = predicted.copy()
                trial[position] = token_id
                trial_ids.append(trial)
            trial_ids = torch.tensor(trial_ids, device=embedding_weight.device)
            trial_embeds = embedding_weight[trial_ids].to(torch.float16)
            trial_mask = attention_mask.expand(candidate_count, -1)
            scores = state_distances(
                oracle.query(trial_embeds, trial_mask), target_hidden,
                mse_weight,
            )
            predicted[position] = trial_ids[
                int(scores.argmin().item()), position
            ].item()
            last_position = position
            if candidate_count < len(items):
                complete = False
                break
        if not complete:
            break
    return predicted, last_position, complete


def load_prompts(args):
    from dataset import Dataset

    prompts = list(Dataset(args.dataset_path, args.dataset_type).get_data())
    return prompts[args.prompt_offset:args.prompt_offset + args.dataset_len]


def build_target_model(args):
    if PeftModel is None:
        raise RuntimeError("Install requirements.txt to load a private LoRA adapter.")
    base = load_base_model(args.base_model_name)
    model = PeftModel.from_pretrained(base, args.target_adapter, torch_dtype=torch.float16)
    model.eval()
    return model


def intercepted_records(args, tokenizer, target_model, prompts):
    records = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
        input_ids = encoded["input_ids"].to("cuda:0")
        attention_mask = encoded["attention_mask"].to("cuda:0")
        hidden = capture_hidden(
            target_model, args.split_layer, input_ids=input_ids,
            attention_mask=attention_mask,
        )
        records.append({
            "prompt": prompt,
            "input_ids": input_ids.cpu(),
            "attention_mask": attention_mask.cpu(),
            "target_hidden": hidden.cpu(),
        })
    return records


def run(args, optimizer_fn=None, method_name=None):
    if args.dataset_len < 1 or args.prompt_offset < 0:
        raise ValueError("--dataset-len must be positive and --prompt-offset non-negative")
    if args.iterations < 1 or args.directions < 1 or args.max_queries < 1:
        raise ValueError("--iterations, --directions, and --max-queries must be positive")
    if args.perturbation_radius <= 0 or args.top_k < 1:
        raise ValueError("--perturbation-radius and --top-k must be positive")
    if (args.perturbation_radius_final is not None
            and args.perturbation_radius_final <= 0):
        raise ValueError("--perturbation-radius-final must be positive")
    if args.alpha_lr <= 0 or (args.alpha_lr_final is not None
                              and args.alpha_lr_final <= 0):
        raise ValueError("--alpha-lr and --alpha-lr-final must be positive")
    if args.alpha_top_k < 0 or args.refinement_passes < 1:
        raise ValueError("--alpha-top-k must be non-negative and --refinement-passes positive")
    if args.hidden_mse_weight < 0:
        raise ValueError("--hidden-mse-weight must be non-negative")
    if not 0 <= args.adam_beta1 < 1 or not 0 <= args.adam_beta2 < 1:
        raise ValueError("Adam beta values must be in [0, 1)")
    if args.adam_eps <= 0:
        raise ValueError("--adam-eps must be positive")
    if getattr(args, "query_batch_size", 1) < 1:
        raise ValueError("--query-batch-size must be positive")
    if getattr(args, "optimization_queries", None) is not None:
        if not 2 <= args.optimization_queries <= args.max_queries:
            raise ValueError("--optimization-queries must be in [2, max-queries]")
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if AutoTokenizer is None:
        raise RuntimeError("Install requirements.txt to run the black-box attack.")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name, trust_remote_code=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = load_prompts(args)
    if not prompts:
        raise ValueError("No prompts were selected.")
    target_model = build_target_model(args)
    records = intercepted_records(args, tokenizer, target_model, prompts)
    embedding_weight = target_model.get_input_embeddings().weight.detach().float()
    dictionary = None
    if args.representation == "sparse":
        dictionary = torch.load(args.dictionary_path, map_location="cpu")
        if not isinstance(dictionary, torch.Tensor) or dictionary.ndim != 2:
            raise ValueError("dictionary must be a rank-2 tensor")
        if dictionary.shape[0] != embedding_weight.shape[1]:
            raise ValueError("dictionary and model embedding dimensions do not match")
        dictionary = dictionary.float().to("cuda:0")

    results = []
    for prompt_index, record in enumerate(records):
        seed_everything(args.seed + prompt_index)
        oracle = BlackBoxOracle(
            target_model, args.split_layer, args.max_queries,
            getattr(args, "query_batch_size", None),
        )
        attention_mask = record["attention_mask"].to("cuda:0")
        target_hidden = record["target_hidden"].to("cuda:0")
        oracle.max_queries = getattr(args, "optimization_queries", None) or args.max_queries
        try:
            alpha, history, exhausted = (optimizer_fn or optimize_coefficients)(
                args, oracle, dictionary, target_hidden, attention_mask
            )
        finally:
            oracle.max_queries = args.max_queries
        optimization_queries = oracle.queries
        embeddings = candidate_embeddings(alpha, dictionary, args.representation)
        candidates = topk_tokens(embeddings, embedding_weight, args.top_k)
        direct_ids = [items[0] for items in candidates]
        refined_ids, last_position, refinement_complete = direct_ids, -1, False
        if args.refine:
            refined_ids, last_position, refinement_complete = refine_tokens(
                oracle, candidates, embedding_weight, target_hidden,
                attention_mask, args.refinement_passes,
                args.hidden_mse_weight,
            )
        target_ids = record["input_ids"][0].tolist()
        results.append({
            "prompt_index": args.prompt_offset + prompt_index,
            "prompt": record["prompt"],
            "sequence_length": len(target_ids),
            "direct_accuracy": token_accuracy(direct_ids, target_ids),
            "refined_accuracy": token_accuracy(refined_ids, target_ids),
            "direct_text": tokenizer.decode(direct_ids[1:]),
            "refined_text": tokenizer.decode(refined_ids[1:]),
            "queries": oracle.queries,
            "optimization_queries": optimization_queries,
            "refinement_queries": oracle.queries - optimization_queries,
            "max_queries": oracle.max_queries,
            "optimization_budget_exhausted": exhausted,
            "last_refined_position": last_position,
            "refinement_complete": refinement_complete,
            "history": history,
        })
        print(json.dumps({"prompt_index": args.prompt_offset + prompt_index,
                          "direct_accuracy": results[-1]["direct_accuracy"],
                          "refined_accuracy": results[-1]["refined_accuracy"],
                          "queries": oracle.queries,
                          "optimization_queries": optimization_queries,
                          "refinement_queries": oracle.queries - optimization_queries,
                          "refinement_complete": refinement_complete}), flush=True)

    summary = {
        "method": method_name or f"black_box_{args.representation}_recovery",
        "settings": vars(args),
        "mean_direct_accuracy": float(np.mean([item["direct_accuracy"] for item in results])),
        "mean_refined_accuracy": float(np.mean([item["refined_accuracy"] for item in results])),
        "results": results,
    }
    stamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = output_dir / f"black_layer{args.split_layer}_seed{args.seed}_{stamp}.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "mean_direct_accuracy": summary["mean_direct_accuracy"],
        "mean_refined_accuracy": summary["mean_refined_accuracy"],
    }, indent=2))
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-name", required=True)
    parser.add_argument("--target-adapter", required=True)
    parser.add_argument("--dictionary-path", default="sparse_dict/auto_encoder.pt")
    parser.add_argument("--representation", choices=("dense", "sparse"), default="sparse")
    parser.add_argument("--dataset-path", default="glue/sst2")
    parser.add_argument("--dataset-type", choices=("local", "datasets", "github"), default="datasets")
    parser.add_argument("--dataset-len", type=int, default=1)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--split-layer", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--directions", type=int, default=10)
    parser.add_argument("--perturbation-radius", type=float, default=1e-3)
    parser.add_argument("--perturbation-radius-final", type=float)
    parser.add_argument("--alpha-lr", type=float, default=0.1)
    parser.add_argument("--alpha-lr-final", type=float)
    parser.add_argument("--alpha-l1", type=float, default=1e-3)
    parser.add_argument("--alpha-init-std", type=float, default=0.1)
    parser.add_argument("--alpha-clip", type=float, default=0.2)
    parser.add_argument(
        "--alpha-top-k", type=int, default=0,
        help="keep this many largest coefficients per token after each update; 0 disables",
    )
    parser.add_argument("--alpha-optimizer", choices=("sgd", "adam"), default="sgd")
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--hidden-mse-weight", type=float, default=0.0)
    parser.add_argument(
        "--return-best-alpha", action=argparse.BooleanOptionalAction, default=False,
        help="decode coefficients from the iteration with the lowest probe loss",
    )
    parser.add_argument("--max-queries", type=int, default=2000)
    parser.add_argument("--optimization-queries", type=int,
                        help="optional optimization cap; remaining total budget is for refinement")
    parser.add_argument("--query-batch-size", type=int, default=8,
                        help="maximum candidates per GPU forward; each candidate still counts")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--refine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refinement-passes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results_black")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
