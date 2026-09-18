"""Summarize only completed comparison JSONs; never infer success from logs."""
import argparse
import csv
import json
from pathlib import Path


def summarize(root):
    rows = []
    for path in sorted(root.glob('*/black_layer*.json')):
        data = json.loads(path.read_text())
        settings = data['settings']
        for result in data['results']:
            if result['queries'] > result['max_queries']:
                raise ValueError(f'Query budget exceeded: {path}')
            rows.append({
                'method': settings.get('group_selection', 'original'),
                'layer': settings['split_layer'], 'seed': settings['seed'],
                'optimization_budget': settings['optimization_queries'],
                'prompt_index': result['prompt_index'],
                'direct_accuracy': result['direct_accuracy'],
                'refined_accuracy': result['refined_accuracy'],
                'queries': result['queries'],
                'optimization_queries': result['optimization_queries'],
                'refinement_queries': result['refinement_queries'],
                'refinement_complete': result['refinement_complete'],
                'result_file': str(path),
            })
    if not rows:
        print('No completed results yet.')
        return
    with (root / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped = {}
    for row in rows:
        key = (row['layer'], row['optimization_budget'], row['method'])
        grouped.setdefault(key, []).append(row)
    lines = ['layer budget method observations direct_acc refined_acc mean_queries']
    for (layer, budget, method), values in sorted(grouped.items()):
        mean = lambda field: sum(value[field] for value in values) / len(values)
        lines.append(f'{layer} {budget} {method} {len(values)} '
                     f'{mean("direct_accuracy"):.4f} {mean("refined_accuracy"):.4f} '
                     f'{mean("queries"):.0f}')
    report = '\n'.join(lines) + '\n'
    (root / 'summary.txt').write_text(report)
    print(report, end='')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    summarize(parser.parse_args().directory)
