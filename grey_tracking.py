"""Persistent metric journal with optional resumable W&B mirroring."""
import hashlib
import json
from pathlib import Path


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class Tracker:
    def __init__(self, directory, args, config):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.journal = self.directory / 'metrics.jsonl'
        self.status_path = self.directory / 'wandb_status.json'
        self.run = None
        self.mode = args.wandb_mode
        self.status = (json.loads(self.status_path.read_text())
                       if self.status_path.exists() else {})
        run_id = self.status.get('run_id') or hashlib.sha256(
            str(self.directory.resolve()).encode()).hexdigest()[:20]
        self.status.update(run_id=run_id, mode=self.mode)
        self.sequence = len(self.journal.read_text().splitlines()) if self.journal.exists() else 0
        if self.mode != 'disabled':
            try:
                import wandb
                self.run = wandb.init(
                    entity=args.wandb_entity, project=args.wandb_project,
                    group=args.wandb_group, id=run_id, resume='allow',
                    name=self.directory.name, job_type=getattr(args, 'stage', 'overview'),
                    tags=[str(config.get('split_layer', 'all')),
                          str(config.get('seed', 'all')),
                          str(config.get('optimization_mode', 'overview')),
                          getattr(args, 'stage', 'overview')],
                    dir=str(self.directory), mode=self.mode, config=config,
                    save_code=False,
                    settings=wandb.Settings(console='off', disable_git=True,
                                            init_timeout=15, x_disable_stats=True))
                self.run.define_metric('optimization/update')
                self.run.define_metric('optimization/*', step_metric='optimization/update')
                self.run.define_metric('progress/completed_prompts')
                self.run.define_metric('accuracy/*', step_metric='progress/completed_prompts')
                self.status['url'] = self.run.url if self.mode == 'online' else None
                self.status['state'] = self.mode
                if self.mode == 'online' and self.journal.exists():
                    uploaded = self.status.get('uploaded_sequence', -1)
                    for line in self.journal.read_text().splitlines():
                        event = json.loads(line)
                        if event['sequence'] > uploaded:
                            self.run.log(event['metrics'])
                            self.status['uploaded_sequence'] = event['sequence']
            except Exception as exc:
                self.status.update(state='pending_sync', error=type(exc).__name__)
                self.run = None
        save_json(self.status_path, self.status)

    def log(self, metrics):
        event = {'sequence': self.sequence, 'metrics': metrics}
        with self.journal.open('a') as stream:
            stream.write(json.dumps(event, allow_nan=False) + '\n')
            stream.flush()
        self.sequence += 1
        if self.run is not None:
            try:
                self.run.log(metrics)
                if self.mode == 'online':
                    self.status['uploaded_sequence'] = event['sequence']
            except Exception as exc:
                self.status.update(state='pending_sync', error=type(exc).__name__)
                self.run = None
        save_json(self.status_path, self.status)

    def finish(self, summary):
        save_json(self.directory / 'tracking_summary.json', summary)
        if self.run is not None:
            try:
                self.run.summary.update(summary)
                self.run.finish(exit_code=0 if summary.get('status') == 'completed' else 1)
            except Exception as exc:
                self.status.update(state='pending_sync', error=type(exc).__name__)
        self.status['finished'] = True
        save_json(self.status_path, self.status)
