#!/usr/bin/env python
"""Qualitative figure of a trained StereoLift checkpoint: rectified pair, FFS disparity, confidence and the attention
of the relation tokens over the native belief cells.

  CUDA_VISIBLE_DEVICES=0 python scripts/visualize_beliefs.py \
      --checkpoint outputs/kitchen18_100_stereolift/checkpoints/005000/pretrained_model \
      --root /path/to/robocasa365_kitchen18 --repo-id local/kitchen18 \
      --episodes 150 250 450 1350 --out outputs/qual_attn.png

Each row: rectified left | rectified right | FFS disparity | confidence c_j | mean relation attention over the cells |
the three queries with the sharpest attention.  Attention maps are at the quarter-resolution FFS grid and are drawn
over the left image.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--frames", type=int, nargs="*", default=[])
    p.add_argument("--episodes", type=int, nargs="*", default=[])
    p.add_argument("--frac", type=float, default=0.5, help="position inside the episode (0 = first frame, 1 = last)")
    p.add_argument("--out", default="outputs/qual_attn.png")
    p.add_argument("--npz", default=None)
    return p.parse_args()


def to_img(t):
    return t.permute(1, 2, 0).clamp(0, 1).float().cpu().numpy()


def main():
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import stereolift  # noqa: F401
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from stereolift.modeling_pi05_stereolift import PI05StereoLiftPolicy

    dev = torch.device("cuda")
    policy = PI05StereoLiftPolicy.from_pretrained(args.checkpoint).to(dev).eval()
    cfg = policy.config
    keys = [cfg.stereo_left_key, cfg.stereo_right_key]
    ds0 = LeRobotDataset(args.repo_id, root=args.root, video_backend="pyav", selected_image_features=keys)
    dts = resolve_delta_timestamps(cfg, ds0.meta)
    ds = LeRobotDataset(args.repo_id, root=args.root, video_backend="pyav", delta_timestamps=dts, selected_image_features=keys)
    pre, _ = make_pre_post_processors(policy_cfg=cfg, pretrained_path=args.checkpoint, dataset_stats=ds.meta.stats)

    frames = list(args.frames)
    for e in args.episodes:
        a = int(ds.meta.episodes["dataset_from_index"][e])
        b = int(ds.meta.episodes["dataset_to_index"][e])
        frames.append(a + int((b - a - 1) * args.frac))
    rows = []
    for fi in frames:
        item = ds[fi]
        batch = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v]) for k, v in item.items()}
        batch = pre(batch)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            policy._prepare_stereo(batch)
            st = policy.model._stereo
            beliefs = st["beliefs"]
            left = policy._to_bchw(batch[cfg.stereo_left_key]).to(dev).float()
            right = policy._to_bchw(batch[cfg.stereo_right_key]).to(dev).float()
            if left.max() > 1.5:
                left, right = left / 255.0, right / 255.0
            left_r, right_r = policy.rectifier(left, right)
            policy.predict_action_chunk(batch)
            aux = dict(policy.model.stereo_aux)
        disp = beliefs["disp"][0, 0].float().cpu().numpy()
        conf = beliefs["conf"][0, 0].float().cpu().numpy()
        h, w = disp.shape
        row = {"task": str(batch.get("task", [""])[0]), "left": to_img(left_r[0]), "right": to_img(right_r[0]), "disp": disp, "conf": conf}
        if "relation_attn" in aux:
            a = aux["relation_attn"][0].float().cpu()  # [M, K]
            K = a.shape[1]
            if K == h * w:
                maps = a.view(-1, h, w)
            else:  # strided key subsampling: scatter back
                stride = math.ceil(h * w / K)
                full = torch.zeros(a.shape[0], h * w)
                full[:, ::stride][:, : a.shape[1]] = a
                maps = full.view(-1, h, w)
            row["attn_mean"] = maps.mean(0).numpy()
            peak = maps.flatten(1).max(-1).values
            top = torch.argsort(peak, descending=True)[:3].tolist()
            row["attn_top"] = [maps[i].numpy() for i in top]
            row["attn_entropy"] = float(aux.get("relation_attn_entropy", torch.tensor(float("nan"))))
        rows.append(row)
        print(f"frame {fi}: task={row['task'][:60]!r} conf mean {conf.mean():.3f} low-conf {(conf < 0.05).mean() * 100:.1f}% "
              f"disp {disp.min():.1f}-{disp.max():.1f} attn_entropy {row.get('attn_entropy', float('nan')):.2f}", flush=True)

    ncol = 7
    fig, axes = plt.subplots(len(rows), ncol, figsize=(1.6 * ncol, 1.7 * len(rows)))
    axes = np.atleast_2d(axes)
    titles = ["left (rectified)", "right (rectified)", "FFS disparity", "confidence $c_j$", "relation attention (mean)", "query 1", "query 2"]
    for r, row in enumerate(rows):
        ims = [row["left"], row["right"], row["disp"], row["conf"], row.get("attn_mean")] + row.get("attn_top", [None] * 3)[:2]
        H, W = row["left"].shape[:2]
        for c, im in enumerate(ims):
            ax = axes[r, c]
            ax.axis("off")
            if r == 0:
                ax.set_title(titles[c], fontsize=7, pad=10)
            if im is None:
                continue
            if c in (0, 1):
                ax.imshow(im)
            elif c == 2:
                ax.imshow(im, cmap="turbo")
            elif c == 3:
                ax.imshow(im, cmap="viridis", vmin=0, vmax=1)
            else:
                ax.imshow(row["left"])
                up = np.kron(im, np.ones((H // im.shape[0], W // im.shape[1])))
                ax.imshow(up, cmap="hot", alpha=0.6, vmin=0, vmax=max(float(up.max()), 1e-6))
        axes[r, 0].annotate(row["task"].split(", State")[0].replace("Task: ", "")[:70], (0.0, 1.03), xycoords="axes fraction",
                            fontsize=6, ha="left", va="bottom", annotation_clip=False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01, wspace=0.04, hspace=0.25)
    fig.savefig(args.out, dpi=180)
    print("wrote", args.out)
    if args.npz:
        np.savez_compressed(args.npz, **{f"{k}_{r}": v for r, row in enumerate(rows) for k, v in row.items() if isinstance(v, np.ndarray)},
                            **{f"attn_top{j}_{r}": v for r, row in enumerate(rows) for j, v in enumerate(row.get("attn_top", []))},
                            tasks=np.array([row["task"] for row in rows]))


if __name__ == "__main__":
    main()
