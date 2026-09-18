"""Bounded local development search, followed by separate-prompt validation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
MODEL = '/home/test/hf_cache/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6'
ADAPTER = '/home/test/DEML/sst2_lora_adapter'
TRIALS = [
    {'name': 'original_lr001', 'kind': 'original', 'lr': 0.01},
    {'name': 'original_lr005', 'kind': 'original', 'lr': 0.05},
    {'name': 'original_lr01', 'kind': 'original', 'lr': 0.1},
    {'name': 'adaptive_lr005', 'kind': 'adaptive', 'lr': 0.05},
    {'name': 'adaptive_lr01', 'kind': 'adaptive', 'lr': 0.1},
    {'name': 'token_groups_lr005', 'kind': 'token_groups', 'lr': 0.05},
    {'name': 'tokenwise_lr0001', 'kind': 'tokenwise', 'lr': 0.001},
    {'name': 'tokenwise_lr0003', 'kind': 'tokenwise', 'lr': 0.003},
    {'name': 'tokenwise_lr001', 'kind': 'tokenwise', 'lr': 0.01},
]


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--wait-for-comparison', type=Path)
    args = parser.parse_args()
    output = (args.output_dir or ROOT / 'results_black_structured' /
              time.strftime('tuning_%Y%m%d_%H%M%S')).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'runner.pid').write_text(str(os.getpid()) + '\n')
    code = output / 'code'
    code.mkdir()
    hashes = {}
    for name in ['invert_sparse_black.py', 'invert_sparse_black_structured.py',
                 'invert_sparse_black_tokenwise.py', 'dataset.py', 'run_black_tuning.py']:
        shutil.copy2(ROOT / name, code / name)
        hashes[name] = hashlib.sha256((code / name).read_bytes()).hexdigest()
    atomic_json(output / 'source_sha256.json', hashes)
    state = {'status': 'starting', 'pid': os.getpid(), 'trials': [],
             'development': {'prompt_offset': 0, 'prompts': 2, 'seeds': [0], 'budget': 8192},
             'validation': {'prompt_offset': 2, 'prompts': 2, 'seeds': [0, 1],
                            'budgets': [8192, 16384]},
             'selection_rule': 'highest development refined accuracy, then direct accuracy; '
                               'select original and new method independently; freeze before validation'}
    state_path = output / 'status.json'
    atomic_json(state_path, state)
    print(f'Output: {output}', flush=True)
    if args.wait_for_comparison:
        state['status'] = 'waiting_for_previous_comparison'
        atomic_json(state_path, state)
        deadline = time.monotonic() + 1800
        while not list((args.wait_for_comparison / 'adaptive_layer5_q16384_seed1').glob('black_layer*.json')):
            if time.monotonic() > deadline:
                raise TimeoutError('Previous comparison did not finish within 30 minutes')
            time.sleep(10)
    env = dict(os.environ, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               PYTHONUNBUFFERED='1', PYTHONPATH=str(code))

    def run_trial(config, phase, budget, seed, offset):
        name = f'{phase}_{config["name"]}_q{budget}_seed{seed}'
        directory = output / name
        directory.mkdir()
        script = {'original': 'invert_sparse_black.py',
                  'adaptive': 'invert_sparse_black_structured.py',
                  'token_groups': 'invert_sparse_black_structured.py',
                  'tokenwise': 'invert_sparse_black_tokenwise.py'}[config['kind']]
        command = [sys.executable, str(code / script), '--base-model-name', MODEL,
                   '--target-adapter', ADAPTER, '--dictionary-path', str(ROOT / 'sparse_dict/auto_encoder.pt'),
                   '--representation', 'sparse', '--dataset-path', 'glue/sst2', '--dataset-type', 'datasets',
                   '--dataset-len', '2', '--prompt-offset', str(offset), '--seed', str(seed),
                   '--split-layer', '5', '--iterations', str(budget // 64), '--directions', '32',
                   '--optimization-queries', str(budget), '--max-queries', str(budget + 4096),
                   '--query-batch-size', '32', '--perturbation-radius', '0.01',
                   '--perturbation-radius-final', '0.00025', '--alpha-lr', str(config['lr']),
                   '--alpha-lr-final', str(config['lr'] / 10), '--alpha-init-std', '0.001',
                   '--alpha-clip', '0.2', '--alpha-l1', '0.001', '--alpha-optimizer', 'sgd',
                   '--hidden-mse-weight', '0.1', '--no-return-best-alpha', '--top-k', '10',
                   '--refine', '--refinement-passes', '1', '--output-dir', str(directory)]
        if config['kind'] == 'adaptive':
            command += ['--groups-per-step', '8', '--probes-per-group', '4']
        elif config['kind'] == 'token_groups':
            command += ['--group-size', '2048', '--groups-per-step', '4', '--probes-per-group', '8',
                        '--group-selection', 'uniform']
        atomic_json(directory / 'command.json', command)
        entry = dict(config, phase=phase, budget=budget, seed=seed,
                     directory=str(directory), status='running', started=time.time())
        state['trials'].append(entry)
        state['status'] = 'running'
        atomic_json(state_path, state)
        print(f'Running {name}', flush=True)
        with (directory / 'run.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        entry['elapsed_seconds'] = time.time() - entry['started']
        files = list(directory.glob('black_layer*.json'))
        if result.returncode or len(files) != 1:
            entry['status'] = state['status'] = 'failed'
            atomic_json(state_path, state)
            raise RuntimeError(f'Trial failed; inspect {directory / "run.log"}')
        data = json.loads(files[0].read_text())
        assert len(data['results']) == 2
        for sample in data['results']:
            assert sample['optimization_queries'] == budget
            assert sample['queries'] <= sample['max_queries']
            assert sample['refinement_complete']
            assert 0 <= sample['direct_accuracy'] <= 1 and 0 <= sample['refined_accuracy'] <= 1
        entry.update(status='completed', direct=data['mean_direct_accuracy'],
                     refined=data['mean_refined_accuracy'],
                     mean_queries=sum(r['queries'] for r in data['results']) / 2,
                     result_file=str(files[0]))
        atomic_json(state_path, state)
        print(f'Completed {name}: direct={entry["direct"]:.4f} refined={entry["refined"]:.4f}', flush=True)
        report()
        return entry

    def report():
        lines = ['# Local tuning results', '',
                 'Layer index 5; token accuracy averaged over two prompts per trial. '
                 'Development and validation use different prompts. These are pilot results.', '',
                 '| Phase | Configuration | Optimization budget | Seed | Direct | Refined | Actual queries (mean) |',
                 '|---|---|---:|---:|---:|---:|---:|']
        for entry in state['trials']:
            if entry['status'] == 'completed':
                lines.append(f'| {entry["phase"]} | {entry["name"]} | {entry["budget"]} | '
                             f'{entry["seed"]} | {100*entry["direct"]:.2f}% | '
                             f'{100*entry["refined"]:.2f}% | {entry["mean_queries"]:.0f} |')
        (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')

    development = [run_trial(config, 'development', 8192, 0, 0) for config in TRIALS]
    key = lambda item: (item['refined'], item['direct'])
    baseline = max((entry for entry in development if entry['kind'] == 'original'), key=key)
    candidate = max((entry for entry in development if entry['kind'] != 'original'), key=key)
    selected = [next(config for config in TRIALS if config['name'] == entry['name'])
                for entry in [baseline, candidate]]
    atomic_json(output / 'frozen_selection.json', selected)
    for budget in [8192, 16384]:
        for seed in [0, 1]:
            for config in selected:
                run_trial(config, 'validation', budget, seed, 2)
    state['status'] = 'completed'
    atomic_json(state_path, state)
    report()
    print(f'Finished. Report: {output / "REPORT.md"}', flush=True)


if __name__ == '__main__':
    main()
