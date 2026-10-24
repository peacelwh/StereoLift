#!/usr/bin/env python
"""Episode lists of the Kitchen-18 protocol (configs/kitchen18_{100,30}demos.json) from a Human300 dataset.

Human300 stores the 300 tasks in alphabetical order inside each family (65 atomic tasks first, then 235 composite
tasks) and task ``i`` owns episodes ``100 i .. 100 i + 99``.  Kitchen-18 is the set of the 18 atomic tasks of the
Target50 benchmark; the 100-demonstration protocol takes all episodes of each task, the 30-demonstration protocol
the first 30.  The shipped json files were produced by this script; re-run it only to rebuild them or to apply the
protocol to another ordering (``--task-index`` maps task name -> Human300 task index).

    python scripts/make_kitchen18_subsets.py --out-dir configs
    python scripts/make_kitchen18_subsets.py --release-dataset   # episode lists for the Kitchen-18 release dataset

With ``--release-dataset`` the lists refer to the 1800-episode Kitchen-18 dataset on the Hub (tasks in the same
order, 100 episodes each) instead of the full Human300.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", type=Path, default=ROOT / "configs" / "kitchen18_tasks.json")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "configs")
    ap.add_argument("--episodes-per-task", type=int, default=100)
    ap.add_argument("--release-dataset", action="store_true")
    args = ap.parse_args()
    spec = json.loads(args.tasks.read_text())
    tasks = spec["tasks"]
    index = spec["task_index"]
    for demos in (100, 30):
        episodes = []
        for k, t in enumerate(tasks):
            base = (k if args.release_dataset else index[t]) * args.episodes_per_task
            episodes += list(range(base, base + demos))
        name = f"kitchen18_{demos}demos{'_release' if args.release_dataset else ''}.json"
        out = args.out_dir / name
        out.write_text(json.dumps({"tasks": tasks, "demos_per_task": demos, "episodes": episodes}) + "\n")
        print(f"wrote {out}: {len(episodes)} episodes")


if __name__ == "__main__":
    main()
