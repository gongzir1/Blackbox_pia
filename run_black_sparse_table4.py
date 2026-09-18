"""Frozen tokenwise black-box evaluation, live W&B metrics and three-seed tables.

Defaults reproduce both selected pilot budgets on the existing Table 4 heldout
manifest. Completed trials are reused; interrupted trials restart from prompt 0.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

from grey_tracking import Tracker, save_json

ROOT = Path(__file__).resolve().parent
SOURCES = ('invert_sparse_black.py', 'invert_sparse_black_tokenwise.py',
           'dataset.py', 'grey_tracking.py', 'run_black_sparse_table4.py')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def command_for(args, root, layer, budget, seed):
    directory = root / f'layer_{layer}_q{budget}_seed_{seed}'
    return [sys.executable, '-u', str(root / 'code/invert_sparse_black_tokenwise.py'),
            '--base-model-name', args.base_model_name, '--target-adapter', args.target_adapter,
            '--dictionary-path', args.dictionary_path, '--dataset-path', str(root / 'heldout_prompts.json'),
            '--dataset-type', 'local', '--dataset-len', str(args.prompts),
            '--split-layer', str(layer), '--seed', str(seed),
            '--iterations', str(budget // 64), '--directions', '32',
            '--optimization-queries', str(budget), '--max-queries', str(budget + 4096),
            '--query-batch-size', str(args.query_batch_size),
            '--perturbation-radius', '0.01', '--perturbation-radius-final', '0.00025',
            '--alpha-lr', '0.001', '--alpha-lr-final', '0.0001',
            '--alpha-init-std', '0.001', '--alpha-clip', '0.2', '--alpha-l1', '0.001',
            '--alpha-optimizer', 'sgd', '--hidden-mse-weight', '0.1',
            '--no-return-best-alpha', '--top-k', '10', '--refine', '--refinement-passes', '1',
            '--output-dir', str(directory)]


def validate_result(payload, count, budget):
    rows = payload['results']
    if len(rows) != count or [r['prompt_index'] for r in rows] != list(range(count)):
        raise ValueError('Incomplete or duplicated prompt results')
    for row in rows:
        if not all(math.isfinite(row[k]) and 0 <= row[k] <= 1
                   for k in ('direct_accuracy', 'refined_accuracy')):
            raise ValueError('Invalid accuracy')
        if (row['optimization_queries'] != budget or row['queries'] > budget + 4096
                or row['queries'] != row['optimization_queries'] + row['refinement_queries']
                or not row['refinement_complete']):
            raise ValueError('Query accounting or refinement incomplete')
    return rows


def aggregate(root, args):
    rows = [json.loads(p.read_text()) for p in root.glob('layer_*/completed.json')]
    table = []
    for layer in args.layers:
        for budget in args.budgets:
            selected = [r for r in rows if r['layer'] == layer and r['budget'] == budget]
            if {r['seed'] for r in selected} != set(args.seeds):
                continue
            item = dict(layer=layer, optimization_budget=budget, seeds=len(selected), prompts=args.prompts)
            for key in ('top1_accuracy', 'refined_accuracy', 'optimization_queries',
                        'refinement_queries', 'total_queries'):
                values = [r[key] for r in selected]
                item[key + '_mean'] = statistics.mean(values)
                item[key + '_std'] = statistics.stdev(values) if len(values) > 1 else 0
            table.append(item)
    save_json(root / 'table4_summary.json', table)
    if table:
        with (root / 'table4_summary.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    lines = ['# Black-box sparse recovery', '',
             'Accuracy is mean per-prompt token accuracy; ± is sample SD across seeds.',
             'Only settings with every requested seed completed appear below.', '',
             '| Layer | Optimization budget | Top-1 (%) | Refined (%) | Total queries/prompt |',
             '|---|---:|---:|---:|---:|']
    for row in table:
        lines.append(f"| {row['layer']} | {row['optimization_budget']} | "
                     f"{100*row['top1_accuracy_mean']:.2f} ± {100*row['top1_accuracy_std']:.2f} | "
                     f"{100*row['refined_accuracy_mean']:.2f} ± {100*row['refined_accuracy_std']:.2f} | "
                     f"{row['total_queries_mean']:.1f} |")
    (root / 'TABLE4.md').write_text('\n'.join(lines) + '\n')
    return table


def execute(args, root, layer, budget, seed):
    directory = root / f'layer_{layer}_q{budget}_seed_{seed}'
    directory.mkdir(exist_ok=True)
    command = command_for(args, root, layer, budget, seed)
    if (directory / 'completed.json').exists():
        return
    save_json(directory / 'command.json', command)
    config = dict(layer=layer, split_layer=layer, budget=budget, seed=seed,
                  prompts=args.prompts, method='tokenwise_sparse', command=command,
                  protocol_sha256=digest(root / 'protocol.json'))
    tracker = Tracker(directory, args, config)
    if args.wandb_mode == 'online' and tracker.run is None:
        raise RuntimeError('W&B connection failed; see trial wandb_status.json')
    save_json(root / 'status.json', dict(status='running', **config, started_at=time.time()))
    samples = []
    try:
        native = list(directory.glob('black_layer*.json'))
        if not native:
            with (directory / 'console.log').open('a') as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, bufsize=1)
                try:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(event, dict) or 'prompt_index' not in event:
                            continue
                        samples.append(event)
                        metrics = {'progress/completed_prompts': len(samples),
                                   'accuracy/top1': statistics.mean(r['direct_accuracy'] for r in samples),
                                   'accuracy/refined': statistics.mean(r['refined_accuracy'] for r in samples),
                                   'prompt/top1_accuracy': event['direct_accuracy'],
                                   'prompt/refined_accuracy': event['refined_accuracy']}
                        for key in ('optimization_queries', 'refinement_queries', 'queries'):
                            name = 'total_queries' if key == 'queries' else key
                            metrics['queries/' + name + '_mean'] = statistics.mean(r[key] for r in samples)
                            metrics['prompt/' + name] = event[key]
                        tracker.log(metrics)
                        save_json(directory / 'progress.json', dict(completed_prompts=len(samples), **event))
                        print(f'{directory.name}: {len(samples)}/{args.prompts} prompts', flush=True)
                    if process.wait():
                        raise RuntimeError(f'Attack failed: {directory / "console.log"}')
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait()
            native = list(directory.glob('black_layer*.json'))
        if len(native) != 1:
            raise ValueError('Expected exactly one native result file')
        payload = json.loads(native[0].read_text())
        rows = validate_result(payload, args.prompts, budget)
        summary = dict(status='completed', layer=layer, budget=budget, seed=seed,
                       prompts=args.prompts, result_file=str(native[0]),
                       top1_accuracy=statistics.mean(r['direct_accuracy'] for r in rows),
                       refined_accuracy=statistics.mean(r['refined_accuracy'] for r in rows))
        for key in ('optimization_queries', 'refinement_queries', 'queries'):
            summary['total_queries' if key == 'queries' else key] = statistics.mean(r[key] for r in rows)
        tracker.finish(summary)
        save_json(directory / 'completed.json', summary)
    except BaseException:
        tracker.finish({'status': 'failed', 'completed_prompts': len(samples)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    protocol_path = ROOT / 'protocol.json'
    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
    for key in ('base_model_name', 'target_adapter', 'dictionary_path'):
        parser.add_argument('--' + key.replace('_', '-'), default=protocol.get(key))
    parser.add_argument('--manifest', default=str(ROOT / 'data/heldout_prompts.json'))
    parser.add_argument('--output-dir', default=str(ROOT / 'results_black_sparse_table4'))
    parser.add_argument('--layers', nargs='+', type=int, default=[5, 10, 15, 20])
    parser.add_argument('--budgets', nargs='+', type=int, default=[8192, 16384])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--prompts', type=int, default=200)
    parser.add_argument('--query-batch-size', type=int, default=32)
    parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default='online')
    parser.add_argument('--wandb-entity', default='griffith')
    parser.add_argument('--wandb-project', default='black-sparse-table4')
    parser.add_argument('--wandb-group', default='tokenwise-frozen-heldout')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.prompts < 1 or any(b < 64 or b % 64 for b in args.budgets):
        parser.error('Positive prompt count and budgets divisible by 64 required')
    for values in (args.layers, args.budgets, args.seeds):
        if len(values) != len(set(values)):
            parser.error('Duplicate grid entries')
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prompts = json.loads(Path(args.manifest).read_text())[:args.prompts]
        if len(prompts) != args.prompts:
            raise ValueError('Not enough heldout prompts')
        frozen = {k: v for k, v in vars(args).items() if k != 'prepare_only'}
        frozen['manifest_sha256'] = digest(args.manifest)
        frozen['dictionary_sha256'] = digest(args.dictionary_path)
        frozen['source_sha256'] = {name: digest(ROOT / name) for name in SOURCES}
        frozen['selection_source'] = 'results_black_structured/tuning_20260908_152136/VALIDATION_SUMMARY.md'
        if (root / 'protocol.json').exists():
            if json.loads((root / 'protocol.json').read_text()) != frozen:
                raise ValueError('Protocol changed: use a new output directory')
        else:
            save_json(root / 'protocol.json', frozen)
            save_json(root / 'heldout_prompts.json', prompts)
            (root / 'code').mkdir(exist_ok=True)
            for name in SOURCES:
                shutil.copy2(ROOT / name, root / 'code' / name)
        aggregate(root, args)
        if args.prepare_only:
            print(f'Prepared {len(args.layers)*len(args.budgets)*len(args.seeds)} runs in {root}')
            return
        os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', PYTHONUNBUFFERED='1')
        try:
            for layer in args.layers:
                for budget in args.budgets:
                    for seed in args.seeds:
                        execute(args, root, layer, budget, seed)
                        aggregate(root, args)
            save_json(root / 'status.json', {'status': 'completed', 'finished_at': time.time()})
        except BaseException as exc:
            save_json(root / 'status.json', {'status': 'failed', 'error': str(exc), 'time': time.time()})
            raise


if __name__ == '__main__':
    main()
