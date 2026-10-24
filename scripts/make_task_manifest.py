#!/usr/bin/env python
"""Write the task manifest used by the RoboCasa365 task-balanced sampler and by the task-balanced statistics.

A RoboCasa365 LeRobot dataset stores the demonstrations of each task as a contiguous block of episodes (Human300:
task ``i`` owns episodes ``100 i .. 100 i + 99``).  The manifest lists these blocks; ``--dataset.task_sampling_manifest``
then draws a task with probability proportional to ``frames ** 0.4``, a demonstration uniformly inside the task and a
frame uniformly inside the demonstration (the official Human300 loader).

    python scripts/make_task_manifest.py --dataset-root /path/to/robocasa365_human300 --episodes-per-task 100
    python scripts/make_task_manifest.py --dataset-root /path/to/robocasa365_kitchen18 --episodes-per-task 100

Writes ``<dataset-root>/meta/task_manifest.parquet`` (columns: source_task_index, output_episode_from,
output_episode_to) unless ``--out`` is given.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--episodes-per-task", type=int, default=100)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    info = json.loads((args.dataset_root / "meta" / "info.json").read_text())
    n = int(info["total_episodes"])
    k = args.episodes_per_task
    if n % k:
        raise SystemExit(f"{n} episodes is not a multiple of {k} episodes per task")
    rows = [(i, i * k, (i + 1) * k) for i in range(n // k)]
    table = pa.table(
        {
            "source_task_index": pa.array([r[0] for r in rows], pa.int64()),
            "output_episode_from": pa.array([r[1] for r in rows], pa.int64()),
            "output_episode_to": pa.array([r[2] for r in rows], pa.int64()),
        }
    )
    out = args.out or args.dataset_root / "meta" / "task_manifest.parquet"
    pq.write_table(table, out)
    print(f"wrote {out}: {len(rows)} tasks x {k} episodes")


if __name__ == "__main__":
    main()
