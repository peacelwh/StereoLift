"""Unit tests of the StereoLift interface that need neither lerobot nor the FFS weights.

    python -m pytest tests -q
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from stereolift.ffs_belief import belief_feature_tensor, collapse_beliefs_to_point
from stereolift.lifting import (
    AffineLifter,
    FourierFeatures,
    cell_token_index,
    pool_to_token_grid,
    quarter_grid_native_coords,
    resize_with_pad_geometry,
    warp_grid_to_raw,
)
from stereolift.modules import GeometricTokenEmbedding, RelationResampler, StereoLiftAdapter
from stereolift.rectification import translate_image, warp_homography


def _pinhole_project(X, f, cx, cy, B):
    u = f * X[..., 0] / X[..., 2] + cx
    v = f * X[..., 1] / X[..., 2] + cy
    d = f * B / X[..., 2]
    return u, v, d


def test_affine_lifting_is_linear_in_true_coordinates():
    """Proposition 1: lifted coordinates equal A X for a fixed invertible A, for any assumed intrinsics."""
    torch.manual_seed(0)
    f, cx, cy, B, W, H = 320.0, 330.0, 245.0, 0.06, 640.0, 480.0
    X = torch.rand(200, 3) * torch.tensor([0.6, 0.4, 0.8]) + torch.tensor([-0.3, -0.2, 0.4])
    u, v, d = _pinhole_project(X, f, cx, cy, B)
    lifter = AffineLifter(fov_deg=70.0, scale=0.37)
    Xp = lifter(u, v, d, W, H)
    A, *_ = torch.linalg.lstsq(X, Xp)
    assert torch.allclose(X @ A, Xp, atol=1e-4, rtol=1e-4)
    assert abs(torch.linalg.det(A)) > 1e-6
    # a translation of the scene is a translation in lifted space (X'_i - X'_j = A (X_i - X_j))
    t = torch.tensor([0.05, -0.03, 0.1])
    u2, v2, d2 = _pinhole_project(X + t, f, cx, cy, B)
    shift = lifter(u2, v2, d2, W, H) - Xp
    assert torch.allclose(shift, shift.mean(0, keepdim=True).expand_as(shift), atol=1e-4)


def test_lifting_has_no_trainable_parameters():
    lifter = AffineLifter(60.0, 0.03)
    fourier = FourierFeatures(8, max_log2_freq=4.0)
    assert sum(p.numel() for p in lifter.parameters()) == 0
    assert sum(p.numel() for p in fourier.parameters()) == 0
    assert fourier(torch.zeros(2, 5, 3)).shape == (2, 5, 48)


def test_resize_with_pad_geometry_matches_pi05():
    rh, rw, pt, pl = resize_with_pad_geometry(480, 640)
    assert (rh, rw) == (168, 224) and pt == 28 and pl == 0
    assert resize_with_pad_geometry(256, 256) == (224, 224, 0, 0)


def test_pool_to_token_grid_confidence_weighting():
    feat = torch.zeros(1, 1, 60, 80)
    feat[:, :, :, :40] = 1.0  # left half = 1
    conf = torch.ones(1, 1, 60, 80)
    pooled, pc = pool_to_token_grid(feat, conf, (480, 640))
    assert pooled.shape == (1, 1, 16, 16) and pc.shape == (1, 1, 16, 16)
    assert pc[0, 0, 0, 0] == 0  # padded rows (480x640 -> 168x224 + 28 px of padding) carry no confidence
    assert torch.allclose(pooled[0, 0, 8, :6], torch.ones(6)) and torch.allclose(pooled[0, 0, 8, 10:], torch.zeros(6))
    # zero-confidence cells do not contribute
    conf[:, :, :, :40] = 0.0
    pooled, _ = pool_to_token_grid(feat, conf, (480, 640))
    assert torch.allclose(pooled[0, 0, 8], torch.zeros(16))


def test_quarter_grid_coords_scale():
    u, v = quarter_grid_native_coords(2, 3, 0.5, "cpu")
    assert torch.allclose(u[0], torch.tensor([4.0, 12.0, 20.0])) and torch.allclose(v[:, 0], torch.tensor([4.0, 12.0]))


def test_translate_and_homography_consistency():
    img = torch.rand(1, 3, 32, 48)
    H = torch.tensor([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]])
    a = translate_image(img, 3.0, -2.0)
    b = warp_homography(img, H)
    assert torch.allclose(a[:, :, 2:30, 3:45], b[:, :, 2:30, 3:45], atol=1e-5)


def test_cell_token_index_matches_token_grid():
    h, w, scale, native = 60, 80, 0.5, (480, 640)
    idx = cell_token_index(h, w, scale, native)
    assert idx.shape == (h * w,) and idx.min() >= 0 and idx.max() < 256
    # top-left cell lands in the first non-padded token row (rows 0-1 are padding for 480x640 inputs)
    assert idx[0] == 2 * 16
    assert idx[-1] == 13 * 16 + 15


def test_warp_grid_to_raw_identity_and_translation():
    maps = torch.rand(2, 4, 30, 40)
    assert torch.allclose(warp_grid_to_raw(maps, torch.eye(3), 1.0), maps, atol=1e-5)
    H = torch.tensor([[1.0, 0.0, 8.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # stored -> rectified: +8 native px = +2 cells
    out = warp_grid_to_raw(maps, H, 1.0)
    assert torch.allclose(out[..., :-2], maps[..., 2:], atol=1e-5)
    assert torch.allclose(out[..., -2:], torch.zeros_like(out[..., -2:]))  # outside the rectified image -> 0


def _fake_beliefs(b=2, hid=60, h=16, w=16, bins=48):
    g = torch.Generator().manual_seed(0)
    return {
        "hidden": torch.randn(b, hid, h, w, generator=g),
        "disp": torch.rand(b, 1, h, w, generator=g) * bins * 0.8 + 2,
        "mu": torch.rand(b, 1, h, w, generator=g) * bins,
        "sigma": torch.rand(b, 1, h, w, generator=g) * 3,
        "entropy": torch.rand(b, 1, h, w, generator=g),
        "peak": torch.rand(b, 1, h, w, generator=g),
        "occl": (torch.rand(b, 1, h, w, generator=g) > 0.8).float(),
        "conf": torch.rand(b, 1, h, w, generator=g),
    }


def _cfg(**kw):
    base = dict(
        adapter_hidden_dim=64, adapter_num_heads=4, adapter_dropout=0.0, use_geometric_embedding=True, num_relation_tokens=16,
        relation_semantic_keys=True, relation_use_state=True, relation_max_keys=8192, key_penalty=4.0,
        lift_fov_deg=60.0, lift_scale=0.03, lift_fourier_bands=8, lift_fourier_max_log2=4.0,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_interface_is_identity_and_relation_tokens_are_zero_at_init():
    torch.manual_seed(0)
    cfg = _cfg()
    adapter = StereoLiftAdapter(cfg, token_dim=96, ffs_hidden_dim=60, num_bins=48, state_dim=32)
    left = torch.randn(2, 256, 96)
    lang = torch.randn(2, 7, 96)
    mask = torch.ones(2, 7, dtype=torch.bool)
    state = torch.randn(2, 32)
    fused, extra, aux = adapter(left, lang, mask, _fake_beliefs(), (64, 64), 1.0, state)
    assert torch.allclose(fused, left)  # W_o <- 0: every visual embedding unchanged
    assert extra.shape == (2, 16, 96) and torch.allclose(extra, torch.zeros_like(extra))  # zero-init output projection
    assert aux["token_conf"].shape == (2, 256) and aux["token_pos"].shape == (2, 256, 3)
    assert aux["relation_attn"].shape[1:] == (16, 256) if "relation_attn" in aux else True
    n = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    assert n > 0


def test_ablation_switches():
    cfg_b = _cfg(num_relation_tokens=0)
    adapter = StereoLiftAdapter(cfg_b, 96, 60, 48, 32)
    fused, extra, _ = adapter(torch.randn(1, 256, 96), torch.randn(1, 5, 96), torch.ones(1, 5, dtype=torch.bool), _fake_beliefs(1), (64, 64), 1.0, None)
    assert extra is None and adapter.relation is None
    cfg_c = _cfg(use_geometric_embedding=False)
    adapter = StereoLiftAdapter(cfg_c, 96, 60, 48, 32)
    left = torch.randn(1, 256, 96)
    fused, extra, _ = adapter(left, torch.randn(1, 5, 96), torch.ones(1, 5, dtype=torch.bool), _fake_beliefs(1), (64, 64), 1.0, None)
    assert adapter.gte is None and torch.equal(fused, left) and extra.shape == (1, 16, 96)


def test_point_collapse_keeps_only_the_refined_disparity():
    b = _fake_beliefs(1)
    p = collapse_beliefs_to_point(b)
    assert torch.equal(p["disp"], b["disp"]) and torch.equal(p["mu"], b["disp"])
    assert p["hidden"].abs().sum() == 0 and torch.all(p["conf"] == 1) and torch.all(p["occl"] == 0)
    feats = belief_feature_tensor(p, 48)
    assert feats.shape == (1, 6, 16, 16) and torch.all(feats[:, 4] == 0)  # |d_K - mu| = 0


def test_relation_attention_penalises_low_confidence_cells():
    torch.manual_seed(0)
    fourier = FourierFeatures(4, 2.0)
    rel = RelationResampler(32, 32, 4, 2, 10, fourier, key_penalty=4.0, state_dim=None).eval()
    tokens = torch.randn(1, 4, 32)
    feat = torch.randn(1, 8, 10)
    pos = torch.randn(1, 8, 3)
    conf = torch.ones(1, 8)
    conf[0, :4] = 0.0  # first four cells unreliable
    idx = torch.zeros(8, dtype=torch.long)
    _, aux = rel(tokens, feat, pos, conf, idx, torch.randn(1, 3, 32), torch.ones(1, 3, dtype=torch.bool), None)
    a = aux["attn"][0]  # [M, K]
    assert a[:, :4].sum() < a[:, 4:].sum()


def test_geometric_embedding_output_shape_and_grad():
    gte = GeometricTokenEmbedding(16, 10, 8, FourierFeatures(2, 1.0))
    tokens = torch.randn(2, 5, 16)
    out = gte(tokens, torch.randn(2, 5, 10), torch.randn(2, 5, 3), torch.rand(2, 5))
    assert out.shape == tokens.shape
    out.sum().backward()
    assert gte.out.weight.grad is not None and math.isfinite(float(gte.out.weight.grad.abs().sum()))


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
