"""Copy MLflow runs from several per-run SQLite stores into ONE combined store, so they can be
compared side-by-side in a single MLflow UI. Copies params, tags, and full metric history
(preserving step + timestamp). Snapshot only — re-run to refresh a still-training run.

Usage: uv run python tools/mlflow_consolidate.py
Then:  uv run mlflow ui --backend-store-uri sqlite:///$PWD/runs/mlflow_combined/mlflow.db --port 5000
"""
import os
import mlflow
from mlflow.tracking import MlflowClient

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = {                                   # display label -> run dir under runs/
    'decoder-closeup (old data)':      'coarse-swin-l-decoder-closeup',
    'decoder-closeup (new data, bs8)': 'coarse-swin-l-decoder-closeup-newdata',
}
DEST_DIR = os.path.join(BASE, 'runs', 'mlflow_combined')
DEST_URI = 'sqlite:///' + os.path.join(DEST_DIR, 'mlflow.db')
EXP_NAME = 'decoder-closeup-comparison'

os.makedirs(DEST_DIR, exist_ok=True)
mlflow.set_tracking_uri(DEST_URI)
dst = MlflowClient(tracking_uri=DEST_URI)
exp_id = (dst.get_experiment_by_name(EXP_NAME) or type('x', (), {'experiment_id': None})).experiment_id
if exp_id is None:
    exp_id = dst.create_experiment(EXP_NAME)

for label, rundir in SOURCES.items():
    src_uri = 'sqlite:///' + os.path.join(BASE, 'runs', rundir, 'tb', 'mlflow.db')
    if not os.path.isfile(src_uri.replace('sqlite:///', '')):
        print('skip {} (no db)'.format(label)); continue
    src = MlflowClient(tracking_uri=src_uri)
    src_runs = []
    for e in src.search_experiments():
        src_runs += src.search_runs([e.experiment_id], max_results=1000)
    if not src_runs:
        print('skip {} (no runs)'.format(label)); continue
    r = src_runs[0]                            # one run per store in this project
    new = dst.create_run(exp_id, run_name=label)
    nid = new.info.run_id
    # params
    for k, v in r.data.params.items():
        dst.log_param(nid, k, v)
    # tags (skip mlflow-internal except a couple useful ones)
    for k, v in r.data.tags.items():
        if not k.startswith('mlflow.') or k in ('mlflow.note.content',):
            dst.log_param(nid, 'tag.' + k, v) if False else dst.set_tag(nid, k, v)
    dst.set_tag(nid, 'source_run', rundir)
    # full metric history (preserve step + timestamp)
    n_pts = 0
    for key in r.data.metrics:
        for m in src.get_metric_history(r.info.run_id, key):
            dst.log_metric(nid, m.key, m.value, timestamp=m.timestamp, step=m.step)
            n_pts += 1
    dst.set_terminated(nid, status='FINISHED')
    print('copied {:35s} -> run {}  ({} metrics, {} points)'.format(label, nid[:8], len(r.data.metrics), n_pts))

print('\ncombined store: {}'.format(DEST_URI))
print('serve: uv run mlflow ui --backend-store-uri {} --port 5000'.format(DEST_URI))
