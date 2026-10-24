#!/usr/bin/env python
"""Sharded multi-task RoboCasa365 evaluation of a StereoLift checkpoint (used by scripts/eval_robocasa365.sh).

Examples
--------
Target50 (official protocol: 50 rollouts, seed 7, re-planning every 25 steps), 8 shards on 4 GPUs (2 per GPU):

    for s in 0 1 2 3 4 5 6 7; do
      python scripts/eval_robocasa_multitask.py --checkpoint CKPT --output-root OUT --groups atomic_seen composite_seen \
          composite_unseen --n-action-steps 25 --num-shards 8 --shard-index $s --gpu $((s % 4)) &
    done; wait
    python scripts/eval_robocasa_multitask.py --checkpoint CKPT --output-root OUT --groups ... --summarize-only

Kitchen-18 subset with a rig perturbation and an eval-time policy override:

    python scripts/eval_robocasa_multitask.py --checkpoint CKPT --output-root OUT --task-file configs/kitchen18_tasks.json \
        --episodes-per-task 25 --cam-perturb '{"yaw_deg": 10}' --policy-args "--policy.ffs_iters=0" ...

Results: <output-root>/<group>/<idx>_<task>/eval_info.json (lerobot-eval), summary.json, per_task.csv.
Completed tasks are skipped on re-runs (resumable).  Use the launcher scripts/eval_robocasa365.sh.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

GROUP_ORDER = ("atomic_seen", "composite_seen", "composite_unseen")
ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--groups", nargs="*", choices=GROUP_ORDER, default=[])
    p.add_argument("--tasks", nargs="*", default=[], help="explicit task names (group 'custom' unless in Target50)")
    p.add_argument("--task-file", type=Path, help='json with {"tasks": [...]} (e.g. configs/kitchen18_tasks.json)')
    p.add_argument("--split", default="pretrain")
    p.add_argument("--episodes-per-task", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-action-steps", type=int, default=5)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--gpu", default="0")
    p.add_argument("--object-registries", default='["objaverse","lightwheel"]')
    p.add_argument("--eval-script", type=Path, default=ROOT / "scripts/eval_robocasa_task.sh")
    p.add_argument("--policy-args", default="", help="extra lerobot-eval policy overrides, e.g. '--policy.ffs_iters=0'")
    p.add_argument("--cam-perturb", default="", help="JSON rig perturbation (STEREOLIFT_CAM_PERTURB)")
    p.add_argument("--recording", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    return p.parse_args()


def load_tasks(args) -> list[tuple[str, int, str]]:
    from robocasa.utils.dataset_registry import TARGET_TASKS

    tasks: list[tuple[str, int, str]] = []
    for group in GROUP_ORDER:
        if group in args.groups:
            tasks.extend((group, i, t) for i, t in enumerate(TARGET_TASKS[group]))
    explicit = list(args.tasks)
    if args.task_file:
        explicit += json.loads(args.task_file.read_text())["tasks"]
    lookup = {t: (g, i) for g in GROUP_ORDER for i, t in enumerate(TARGET_TASKS[g])}
    for k, t in enumerate(explicit):
        g, i = lookup.get(t, ("custom", k))
        if (g, i, t) not in tasks:
            tasks.append((g, i, t))
    if not tasks:
        raise SystemExit("no tasks selected: pass --groups, --tasks or --task-file")
    return tasks


def task_dir(root: Path, group: str, idx: int, task: str) -> Path:
    return root / group / f"{idx:02d}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', task)}"


def successes_of(info_path: Path, expected: int) -> list[bool]:
    info = json.loads(info_path.read_text())
    per_task = info.get("per_task", [])
    if len(per_task) != 1:
        raise ValueError(f"expected one per_task entry in {info_path}")
    s = per_task[0].get("metrics", {}).get("successes", [])
    if len(s) != expected:
        raise ValueError(f"expected {expected} episodes in {info_path}, got {len(s)}")
    return [bool(v) for v in s]


def summarize(args, tasks) -> int:
    records, missing, invalid = [], [], []
    for group, idx, task in tasks:
        info = task_dir(args.output_root, group, idx, task) / "eval_info.json"
        if not info.is_file():
            missing.append(f"{group}/{task}")
            continue
        try:
            s = successes_of(info, args.episodes_per_task)
        except Exception as e:  # noqa: BLE001
            invalid.append(f"{group}/{task}: {e}")
            continue
        records.append(
            {"group": group, "task_index": idx, "task": task, "episodes": len(s), "successes": sum(s),
             "pc_success": 100.0 * sum(s) / len(s)}
        )
    groups = {}
    for group in dict.fromkeys(g for g, _, _ in tasks):
        rs = [r for r in records if r["group"] == group]
        groups[group] = {
            "expected_tasks": sum(g == group for g, _, _ in tasks),
            "completed_tasks": len(rs),
            "task_mean_pc_success": (sum(r["pc_success"] for r in rs) / len(rs)) if rs else None,
        }
    overall = (sum(r["pc_success"] for r in records) / len(records)) if records else None
    summary = {
        "checkpoint": str(args.checkpoint),
        "protocol": {"split": args.split, "episodes_per_task": args.episodes_per_task, "seed": args.seed,
                     "n_action_steps": args.n_action_steps, "object_registries": json.loads(args.object_registries),
                     "policy_args": args.policy_args, "cam_perturb": args.cam_perturb},
        "expected_tasks": len(tasks), "completed_tasks": len(records),
        "complete": len(records) == len(tasks) and not invalid,
        "per_group": groups, "overall_task_mean_pc_success": overall,
        "per_task": records, "missing": missing, "invalid": invalid,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output_root / "per_task.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "task_index", "task", "episodes", "successes", "pc_success"])
        w.writeheader()
        w.writerows(records)
    print("\n========== RoboCasa evaluation ==========")
    for g, v in groups.items():
        m = v["task_mean_pc_success"]
        print(f"{g:18s} {'n/a' if m is None else f'{m:6.2f}%'}  ({v['completed_tasks']}/{v['expected_tasks']} tasks)")
    print(f"overall            {'n/a' if overall is None else f'{overall:6.2f}%'}   complete={summary['complete']}")
    if missing:
        print(f"missing: {missing}")
    if invalid:
        print(f"invalid: {invalid}")
    return 0 if summary["complete"] else 3


def main() -> None:
    args = parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    tasks = load_tasks(args)
    if args.summarize_only:
        raise SystemExit(summarize(args, tasks))
    if not (args.checkpoint / "config.json").is_file():
        raise SystemExit(f"missing checkpoint config: {args.checkpoint}")
    if not args.eval_script.is_file():
        raise SystemExit(f"missing eval script: {args.eval_script}")
    mine = tasks[args.shard_index :: args.num_shards]
    print(f"shard {args.shard_index + 1}/{args.num_shards}: {len(mine)} of {len(tasks)} tasks on GPU {args.gpu}", flush=True)
    for k, (group, idx, task) in enumerate(mine, 1):
        out = task_dir(args.output_root, group, idx, task)
        info = out / "eval_info.json"
        if info.is_file() and not args.force:
            try:
                successes_of(info, args.episodes_per_task)
                print(f"[{k}/{len(mine)}] skip complete {group}/{task}", flush=True)
                continue
            except Exception as e:  # noqa: BLE001
                print(f"[{k}/{len(mine)}] redo {group}/{task}: {e}", flush=True)
        out.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        # robosuite requires MUJOCO_EGL_DEVICE_ID to be listed in CUDA_VISIBLE_DEVICES.  Some containers enumerate
        # a single EGL device: set STEREOLIFT_EGL_SINGLE_DEVICE=1 to pass "<gpu>,0" so torch's cuda:0 is the
        # requested GPU while device 0 stays visible for that check.
        single = os.environ.get("STEREOLIFT_EGL_SINGLE_DEVICE", "0") == "1"
        visible = str(args.gpu) if (str(args.gpu) == "0" or not single) else f"{args.gpu},0"
        env.update(
            {
                "CHECKPOINT": str(args.checkpoint), "TASK_NAME": task, "SPLIT": args.split,
                "OUTPUT_DIR": str(out), "RUN_NAME": f"robocasa_{group}_{task}",
                "CUDA_VISIBLE_DEVICES": visible,
                "MUJOCO_EGL_DEVICE_ID": str(args.gpu),
                "N_EPISODES": str(args.episodes_per_task), "EVAL_BATCH_SIZE": str(args.eval_batch_size),
                "N_ACTION_STEPS": str(args.n_action_steps), "SEED": str(args.seed),
                "OBJ_REGISTRIES": args.object_registries, "RECORDING": "true" if args.recording else "false",
                "EXTRA_POLICY_ARGS": args.policy_args,
            }
        )
        if args.cam_perturb:
            env["STEREOLIFT_CAM_PERTURB"] = args.cam_perturb
        else:
            env.pop("STEREOLIFT_CAM_PERTURB", None)
        print(f"[{k}/{len(mine)}] evaluate {group}/{task}: {args.episodes_per_task} rollouts, seed {args.seed}", flush=True)
        if args.dry_run:
            continue
        with (out / "eval.log").open("a") as log:
            r = subprocess.run(["bash", str(args.eval_script)], cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, text=True, check=False)
        if r.returncode != 0:
            print(f"[error] {group}/{task} failed (exit {r.returncode}); see {out / 'eval.log'}", flush=True)
            continue
        try:
            s = successes_of(info, args.episodes_per_task)
            print(f"  done {group}/{task}: {sum(s)}/{len(s)}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[error] {group}/{task}: {e}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
