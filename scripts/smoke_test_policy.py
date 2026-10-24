#!/usr/bin/env python
"""Smoke test: build the StereoLift policy from a pi0.5 checkpoint, run a few training steps and one inference on a
small batch of a LeRobot dataset.  Checks the installation end to end (lerobot, FFS weights, rectification).

RoboCasa365 (256 x 256, homography rectification):
  CUDA_VISIBLE_DEVICES=0 python scripts/smoke_test_policy.py --root /path/to/robocasa365_kitchen18 --repo-id local/kitchen18 \\
      --left-key observation.images.robot0_agentview_left --right-key observation.images.robot0_agentview_right \\
      --rectify-json configs/robocasa_agentview_rectification.json --scale 1.0 --lift-fov-deg 60 --lift-scale 0.03
Real robot (640 x 480, vertical shift):
  CUDA_VISIBLE_DEVICES=0 python scripts/smoke_test_policy.py --root /path/to/InsertRing --repo-id local/InsertRing \\
      --right-dy -4.15 --scale 0.5 --lift-fov-deg 90 --lift-scale 0.1
Add --freeze-backbone to train the interface only (fits in about 15 GB).
"""

from __future__ import annotations

import argparse
import time

import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--pretrained", default="lerobot/pi05_base")
    p.add_argument("--left-key", default="observation.images.head_left")
    p.add_argument("--right-key", default="observation.images.head_right")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--iters", type=int, default=4)
    p.add_argument("--right-dy", type=float, default=0.0)
    p.add_argument("--rectify-json", default=None, help="homography rectification json (RoboCasa toe-in pair)")
    p.add_argument("--lift-fov-deg", type=float, default=90.0)
    p.add_argument("--lift-scale", type=float, default=0.1)
    p.add_argument("--num-relation-tokens", type=int, default=16)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    import stereolift  # noqa: F401
    from lerobot.configs.types import FeatureType, NormalizationMode
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.feature_utils import dataset_to_policy_features

    from stereolift.configuration_pi05_stereolift import PI05StereoLiftConfig
    from stereolift.modeling_pi05_stereolift import PI05StereoLiftPolicy

    image_keys = [args.left_key, args.right_key]
    ds = LeRobotDataset(args.repo_id, root=args.root, video_backend="pyav", selected_image_features=image_keys)
    features = dataset_to_policy_features(ds.meta.features)
    input_features = {k: v for k, v in features.items() if (v.type is not FeatureType.VISUAL or k in image_keys) and v.type is not FeatureType.ACTION}
    output_features = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
    cfg = PI05StereoLiftConfig(
        input_features=input_features,
        output_features=output_features,
        normalization_mapping={"ACTION": NormalizationMode.MEAN_STD, "STATE": NormalizationMode.MEAN_STD, "VISUAL": NormalizationMode.IDENTITY},
        stereo_left_key=args.left_key,
        stereo_right_key=args.right_key,
        ffs_input_scale=args.scale,
        ffs_iters=args.iters,
        rectify_mode="homography" if args.rectify_json else ("vertical_shift" if args.right_dy else "none"),
        rectify_right_dy=args.right_dy,
        rectify_homography_path=args.rectify_json,
        lift_fov_deg=args.lift_fov_deg,
        lift_scale=args.lift_scale,
        num_relation_tokens=args.num_relation_tokens,
        freeze_policy_backbone=args.freeze_backbone,
        gradient_checkpointing=True,
        dtype=args.dtype,
        device=args.device,
    )
    cfg.pretrained_path = args.pretrained
    delta_timestamps = resolve_delta_timestamps(cfg, ds.meta)
    ds = LeRobotDataset(args.repo_id, root=args.root, video_backend="pyav", delta_timestamps=delta_timestamps, selected_image_features=image_keys)

    t0 = time.time()
    policy = PI05StereoLiftPolicy.from_pretrained(args.pretrained, config=cfg)
    policy.to(args.device)
    print(f"policy built in {time.time() - t0:.1f}s")
    n_total = sum(p.numel() for p in policy.parameters())
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_adapter = sum(p.numel() for p in policy.model.adapter.parameters())
    n_ffs = sum(p.numel() for p in policy.model.ffs.parameters())
    print(f"params: total {n_total / 1e6:.1f}M  trainable {n_train / 1e6:.1f}M  interface {n_adapter / 1e6:.2f}M  ffs (frozen) {n_ffs / 1e6:.2f}M")

    pre, _post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=None, dataset_stats=ds.meta.stats)
    step_ = max(len(ds) // (args.batch + 1), 1)
    items = [ds[i] for i in range(0, step_ * args.batch, step_)][: args.batch]
    batch = {}
    for k in items[0]:
        vals = [it[k] for it in items]
        batch[k] = torch.stack(vals) if isinstance(vals[0], torch.Tensor) else vals
    batch = pre(batch)
    for k in sorted(batch):
        if isinstance(batch[k], torch.Tensor):
            print(f"  {k}: {tuple(batch[k].shape)} {batch[k].dtype}")

    policy.train()
    opt = torch.optim.AdamW(policy.get_optim_params(), lr=1e-5)
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, info = policy.forward(batch)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        print(f"step {step}: loss {loss.item():.4f}  {time.time() - t0:.2f}s  { {k: round(v, 4) for k, v in info.items() if isinstance(v, float)} }")
    print(f"peak train memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")

    policy.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(2):
            actions = policy.predict_action_chunk(batch)
        torch.cuda.synchronize()
        t0 = time.time()
        actions = policy.predict_action_chunk(batch)
        torch.cuda.synchronize()
        dt = time.time() - t0
    conf = policy.last_stereo_confidence
    print(f"inference: actions {tuple(actions.shape)} finite={bool(torch.isfinite(actions).all())} {dt * 1000:.0f} ms for batch {args.batch}; "
          f"mean stereo confidence {conf.mean().item():.3f}")
    print("OK")


if __name__ == "__main__":
    main()
