#!/usr/bin/env python
"""Task-balanced normalisation statistics of the state and action (the RoboCasa365 recipe).

Every task contributes with the same weight to the mean and the standard deviation, whatever the number of frames
it has.  The per-episode statistics stored in ``meta/episodes`` are first aggregated inside each task (frame
weighted), then averaged across tasks with equal weights.  The result is written next to the canonical
``meta/stats.json`` and passed to training with ``--dataset.stats_override_path`` (scripts/train_robocasa365.sh does
this automatically when the file exists).

    python scripts/compute_task_balanced_stats.py --dataset-root /path/to/robocasa365_human300
    python scripts/compute_task_balanced_stats.py --dataset-root /path/to/robocasa365_kitchen18 \\
        --episodes-file configs/kitchen18_30demos.json --out .../meta/stats_task_balanced_30demos.json

The task blocks come from the manifest written by scripts/make_task_manifest.py (default
``meta/task_manifest.parquet``).  With ``--episodes-file`` only the listed episodes are used (30-demo protocol).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as pads
import pyarrow.parquet as pq

FEATURES = ("observation.state", "action")


def aggregate(rows: dict[str, list], indices: np.ndarray, feature: str) -> tuple[np.ndarray, ...]:
    means = np.asarray([rows[f"stats/{feature}/mean"][i] for i in indices], dtype=np.float64)
    stds = np.asarray([rows[f"stats/{feature}/std"][i] for i in indices], dtype=np.float64)
    minimums = np.asarray([rows[f"stats/{feature}/min"][i] for i in indices], dtype=np.float64)
    maximums = np.asarray([rows[f"stats/{feature}/max"][i] for i in indices], dtype=np.float64)
    raw_counts = [rows[f"stats/{feature}/count"][i] for i in indices]
    counts = np.asarray([v[0] if isinstance(v, (list, np.ndarray)) else v for v in raw_counts], dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        raise ValueError(f"feature {feature} has a non-positive sample count")
    mean = np.sum(means * counts[:, None], axis=0) / total
    second = np.sum((np.square(stds) + np.square(means)) * counts[:, None], axis=0) / total
    return mean, np.sqrt(np.maximum(second - np.square(mean), 0.0)), minimums.min(axis=0), maximums.max(axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None, help="default: <dataset-root>/meta/task_manifest.parquet")
    ap.add_argument("--episodes-file", type=Path, default=None, help='json with {"episodes": [...]} restricting the episodes')
    ap.add_argument("--out", type=Path, default=None, help="default: <dataset-root>/meta/stats_task_balanced.json")
    args = ap.parse_args()
    root = args.dataset_root.expanduser().resolve()
    manifest = args.manifest or root / "meta" / "task_manifest.parquet"
    out = args.out or root / "meta" / "stats_task_balanced.json"
    for p in (manifest, root / "meta" / "stats.json", root / "meta" / "episodes"):
        if not p.exists():
            raise FileNotFoundError(p)

    groups = pq.read_table(manifest, columns=["source_task_index", "output_episode_from", "output_episode_to"]).to_pylist()
    keep = None
    if args.episodes_file:
        keep = set(json.loads(args.episodes_file.read_text())["episodes"])

    columns = ["episode_index"]
    for f in FEATURES:
        columns += [f"stats/{f}/{s}" for s in ("mean", "std", "min", "max", "count")]
    rows = pads.dataset(root / "meta" / "episodes", format="parquet").to_table(columns=columns).to_pydict()
    episode_index = np.asarray(rows["episode_index"], dtype=np.int64)
    position = {int(e): i for i, e in enumerate(episode_index)}

    stats = json.loads((root / "meta" / "stats.json").read_text())
    n_tasks_used = 0
    for f in FEATURES:
        task_means, task_stds, task_mins, task_maxs = [], [], [], []
        for g in groups:
            eps = [e for e in range(int(g["output_episode_from"]), int(g["output_episode_to"])) if keep is None or e in keep]
            if not eps:
                continue
            idx = np.asarray([position[e] for e in eps])
            m, s, lo, hi = aggregate(rows, idx, f)
            task_means.append(m), task_stds.append(s), task_mins.append(lo), task_maxs.append(hi)
        means, stds = np.asarray(task_means), np.asarray(task_stds)
        mean = means.mean(axis=0)
        var = np.maximum((np.square(stds) + np.square(means)).mean(axis=0) - mean**2, 0.0)
        stats[f]["mean"] = mean.tolist()
        stats[f]["std"] = np.sqrt(var).tolist()
        stats[f]["min"] = np.asarray(task_mins).min(axis=0).tolist()
        stats[f]["max"] = np.asarray(task_maxs).max(axis=0).tolist()
        n_tasks_used = len(task_means)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"wrote task-balanced statistics over {n_tasks_used} tasks: {out}")


if __name__ == "__main__":
    main()
