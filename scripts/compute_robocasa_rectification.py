#!/usr/bin/env python
"""Compute fixed rectifying homographies for RoboCasa's toe-in ``agentview_left/right`` pair.

Run inside the RoboCasa environment (needs MuJoCo/EGL).  The shipped configs/robocasa_agentview_rectification.json
was produced by exactly this command:

  MUJOCO_GL=egl python scripts/compute_robocasa_rectification.py \
      --task CloseFridge --width 256 --height 256 --out configs/robocasa_agentview_rectification.json

The output json holds ``H_left``, ``H_right`` (output <- input pixel homographies),
the rectified projection matrices, the baseline and the expected disparity range for
the given depth range, and is consumed by ``rectify_mode=homography``.
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="CloseFridge")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--left", default="robot0_agentview_left")
    p.add_argument("--right", default="robot0_agentview_right")
    p.add_argument("--out", default="configs/robocasa_agentview_rectification.json")
    p.add_argument("--save-example", default=None, help="png path to save an example rectified pair")
    args = p.parse_args()

    import robosuite.utils.camera_utils as CU
    from robocasa.utils.env_utils import create_env

    from stereolift.rectification import stereo_rectify_homographies

    env = create_env(env_name=args.task, render_onscreen=False, camera_names=[args.left, args.right],
                     camera_widths=args.width, camera_heights=args.height)
    obs = env.reset()
    sim = env.sim
    K1 = CU.get_camera_intrinsic_matrix(sim, args.left, args.height, args.width)
    K2 = CU.get_camera_intrinsic_matrix(sim, args.right, args.height, args.width)
    E1 = CU.get_camera_extrinsic_matrix(sim, args.left)  # camera -> world (OpenGL convention)
    E2 = CU.get_camera_extrinsic_matrix(sim, args.right)
    # robosuite/MuJoCo cameras look down -z; convert to OpenCV (z forward, y down)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    E1cv, E2cv = E1 @ flip, E2 @ flip
    W2C1, W2C2 = np.linalg.inv(E1cv), np.linalg.inv(E2cv)
    R1, t1 = W2C1[:3, :3], W2C1[:3, 3]
    R2, t2 = W2C2[:3, :3], W2C2[:3, 3]
    H_l, H_r, P1, P2, baseline = stereo_rectify_homographies(K1, R1, t1, K2, R2, t2, args.width, args.height)
    f_new = float(P1[0, 0])
    report = {
        "task": args.task,
        "size": [args.width, args.height],
        "K_left": K1.tolist(),
        "K_right": K2.tolist(),
        "H_left": H_l.tolist(),
        "H_right": H_r.tolist(),
        "P1": P1.tolist(),
        "P2": P2.tolist(),
        "baseline_m": baseline,
        "f_rectified_px": f_new,
        "toe_in_deg": float(np.degrees(np.arccos(np.clip((np.trace(R2 @ R1.T) - 1) / 2, -1, 1)))),
        "disparity_px_at_depth": {str(z): f_new * baseline / z for z in (0.5, 0.8, 1.0, 1.5, 2.0, 3.0)},
        "baseline_over_depth": {str(z): baseline / z for z in (0.5, 0.8, 1.0, 1.5, 2.0, 3.0)},
    }
    print(json.dumps({k: v for k, v in report.items() if k not in ("H_left", "H_right", "P1", "P2", "K_left", "K_right")}, indent=2))
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", args.out)

    if args.save_example:
        import cv2
        import torch

        from stereolift.rectification import warp_homography

        l_img = obs[f"{args.left}_image"][::-1].copy()
        r_img = obs[f"{args.right}_image"][::-1].copy()
        lt = torch.from_numpy(l_img).permute(2, 0, 1)[None].float() / 255
        rt = torch.from_numpy(r_img).permute(2, 0, 1)[None].float() / 255
        lw = warp_homography(lt, torch.tensor(H_l, dtype=torch.float32))
        rw = warp_homography(rt, torch.tensor(H_r, dtype=torch.float32))
        top = np.concatenate([l_img, r_img], 1)
        bot = np.concatenate([(lw[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8), (rw[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)], 1)
        panel = np.concatenate([top, bot], 0)
        for y in range(0, panel.shape[0], 32):
            cv2.line(panel, (0, y), (panel.shape[1] - 1, y), (0, 255, 0), 1)
        cv2.imwrite(args.save_example, panel[..., ::-1])
        print("saved", args.save_example)


if __name__ == "__main__":
    main()
