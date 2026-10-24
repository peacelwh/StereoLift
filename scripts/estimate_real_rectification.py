#!/usr/bin/env python
"""Estimate the residual vertical/horizontal misalignment of the real head stereo pair.

The G1-D head camera delivers a single 1280x480 frame split into left/right
halves.  We match ORB/SIFT features between the halves on a subsample of frames
and report the robust median vertical offset (dy) and disparity statistics.  The
median dy is what ``rectify_mode=vertical_shift`` should compensate.

  python scripts/estimate_real_rectification.py --root /data/InsertRing --repo-id local/InsertRing --samples 40

Pass the reported ``recommended_rectify_right_dy`` as ``RECTIFY_RIGHT_DY`` to scripts/train_real_robot.sh.
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--left-key", default="observation.images.head_left")
    p.add_argument("--right-key", default="observation.images.head_right")
    p.add_argument("--samples", type=int, default=40)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    import cv2
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(args.repo_id, root=args.root, video_backend="pyav")
    idx = np.linspace(0, len(ds) - 1, args.samples).astype(int)
    sift = cv2.SIFT_create(nfeatures=2000)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    dys, dxs = [], []
    for i in idx:
        item = ds[int(i)]
        l = (item[args.left_key].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        r = (item[args.right_key].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        gl, gr = cv2.cvtColor(l, cv2.COLOR_RGB2GRAY), cv2.cvtColor(r, cv2.COLOR_RGB2GRAY)
        kl, dl = sift.detectAndCompute(gl, None)
        kr, dr = sift.detectAndCompute(gr, None)
        if dl is None or dr is None:
            continue
        for m, n in matcher.knnMatch(dl, dr, k=2):
            if m.distance < 0.7 * n.distance:
                pl, pr = kl[m.queryIdx].pt, kr[m.trainIdx].pt
                dx, dy = pl[0] - pr[0], pl[1] - pr[1]
                if -5 <= dx <= 200:  # plausible positive disparity range
                    dxs.append(dx)
                    dys.append(dy)
    dys, dxs = np.array(dys), np.array(dxs)
    good = np.abs(dys - np.median(dys)) < 3 * (np.median(np.abs(dys - np.median(dys))) + 1e-6) + 2
    report = {
        "num_matches": int(len(dys)),
        "dy_median_px": float(np.median(dys)),
        "dy_mad_px": float(np.median(np.abs(dys - np.median(dys)))),
        "dy_p90_abs_px": float(np.percentile(np.abs(dys), 90)),
        "disparity_median_px": float(np.median(dxs[good])),
        "disparity_p10_px": float(np.percentile(dxs[good], 10)),
        "disparity_p90_px": float(np.percentile(dxs[good], 90)),
        # dy = y_left - y_right < 0 means the right-image content sits lower; StereoRectifier's
        # translate_image moves content DOWN for positive dy, so pass +median(dy) (negative here).
        "recommended_rectify_right_dy": float(np.median(dys)),
        "image_size": [int(l.shape[1]), int(l.shape[0])],
    }
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
