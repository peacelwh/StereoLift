"""Test-time stereo-rig perturbations for RoboCasa evaluations.

Activated by the environment variable ``STEREOLIFT_CAM_PERTURB`` (a JSON object).  The perturbation is
applied to the MuJoCo model after every ``_reset_internal`` of the robosuite environment, i.e. before the
first observation of each episode is rendered, so that every frame of the episode is captured by the
perturbed rig.  All quantities are expressed in the frame of the parent body of the cameras
(``mobilebase0_support`` for the PandaOmron agentview pair; z up, x forward, y to the left):

    yaw_deg          rigid rotation of the whole rig about the vertical axis through the rig midpoint
    pitch_deg        rigid rotation about the rig's lateral (baseline) axis through the midpoint
    roll_deg         rigid rotation about the rig's forward axis through the midpoint
    dx, dy, dz       rigid translation of the rig (metres)
    baseline_scale   scale the baseline about the midpoint (1.0 = unchanged); orientations unchanged
    cameras          list of camera names (default: the agentview stereo pair)

Rigid rig motions and baseline scaling leave the *relative* pose of the two cameras unchanged up to the
baseline length, so the fixed rectifying homographies used for training remain valid.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np

log = logging.getLogger(__name__)

ENV_VAR = "STEREOLIFT_CAM_PERTURB"
DEFAULT_CAMERAS = ("robot0_agentview_left", "robot0_agentview_right")
_installed = False


def _quat_from_axis_angle(axis: np.ndarray, deg: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    half = np.deg2rad(deg) / 2.0
    return np.concatenate([[np.cos(half)], np.sin(half) * axis])  # MuJoCo (w, x, y, z)


def _quat_mul(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def parse_spec(text: str | None = None) -> dict | None:
    text = os.environ.get(ENV_VAR, "") if text is None else text
    text = text.strip()
    if not text:
        return None
    spec = json.loads(text)
    if not isinstance(spec, dict):
        raise ValueError(f"{ENV_VAR} must be a JSON object, got {text!r}")
    return spec


def apply_rig_perturbation(model, spec: dict) -> dict:
    """Modify ``model.cam_pos`` / ``model.cam_quat`` in place.  Returns a small report."""
    names = list(spec.get("cameras", DEFAULT_CAMERAS))
    ids = [model.camera_name2id(n) for n in names]
    pos = np.array([np.asarray(model.cam_pos[i], dtype=np.float64) for i in ids])
    quat = np.array([np.asarray(model.cam_quat[i], dtype=np.float64) for i in ids])
    mid = pos.mean(axis=0)

    # baseline scaling about the midpoint (orientations unchanged)
    scale = float(spec.get("baseline_scale", 1.0))
    pos = mid + scale * (pos - mid)

    # rigid rotation of the rig about its midpoint: yaw about body z, pitch about the baseline axis,
    # roll about the forward axis (all in the parent-body frame)
    lateral = pos[0] - pos[-1] if len(pos) > 1 else np.array([0.0, 1.0, 0.0])
    if np.linalg.norm(lateral) < 1e-9:
        lateral = np.array([0.0, 1.0, 0.0])
    lateral = lateral / np.linalg.norm(lateral)
    up = np.array([0.0, 0.0, 1.0])
    forward = np.cross(lateral, up)
    forward = forward / max(np.linalg.norm(forward), 1e-12)
    rot = np.array([1.0, 0.0, 0.0, 0.0])
    for key, axis in (("yaw_deg", up), ("pitch_deg", lateral), ("roll_deg", forward)):
        deg = float(spec.get(key, 0.0))
        if deg:
            rot = _quat_mul(_quat_from_axis_angle(axis, deg), rot)
    R = _quat_to_mat(rot)
    pos = mid + (pos - mid) @ R.T
    quat = np.array([_quat_mul(rot, q) for q in quat])

    # rigid translation
    shift = np.array([float(spec.get("dx", 0.0)), float(spec.get("dy", 0.0)), float(spec.get("dz", 0.0))])
    pos = pos + shift

    for k, i in enumerate(ids):
        model.cam_pos[i] = pos[k]
        model.cam_quat[i] = quat[k] / np.linalg.norm(quat[k])
    return {"cameras": names, "cam_pos": pos.tolist(), "cam_quat": quat.tolist()}


def install(spec: dict | None = None) -> bool:
    """Monkey-patch robosuite so the perturbation is applied after every internal reset."""
    global _installed
    spec = parse_spec() if spec is None else spec
    if spec is None or _installed:
        return _installed
    try:
        from robosuite.environments.base import MujocoEnv
    except Exception as exc:  # pragma: no cover - only relevant in the sim env
        log.warning("robosuite not importable; rig perturbation not installed: %s", exc)
        return False

    original = MujocoEnv._reset_internal
    state = {"logged": False}

    def _reset_internal(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        try:
            report = apply_rig_perturbation(self.sim.model, spec)
            if not state["logged"]:
                log.info("StereoLift rig perturbation %s -> %s", json.dumps(spec), json.dumps(report))
                print(f"[stereolift] rig perturbation {json.dumps(spec)} -> {json.dumps(report)}", flush=True)
                state["logged"] = True
        except Exception as exc:
            log.error("rig perturbation failed: %s", exc)
            raise
        return out

    MujocoEnv._reset_internal = _reset_internal
    _installed = True
    return True
