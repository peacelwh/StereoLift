"""Fast-FoundationStereo (FFS) belief extractor.

Re-implements the FFS forward pass step by step so that the intermediate quantities that StereoLift reads are
exposed (the released model only returns the final up-sampled disparity):

* ``prob``                initial disparity posterior over ``max_disp / 4`` bins (quarter resolution)
* ``mu / sigma / entropy``  moments of that posterior
* ``disp``                refined disparity ``d_K`` after ``K`` ConvGRU updates (quarter-resolution pixel units)
* ``hidden``              final ConvGRU hidden state ``h_K``
* ``peak``                sharpness ``pi`` of the local correlation profile at the refined disparity
* ``occl``                occlusion / out-of-view indicator ``o``
* ``conf``                combined confidence ``c = (1 - H / log D_b) pi (1 - o)`` in [0, 1]
* ``disp_full``           (optional) convex-upsampled full-resolution disparity, for visualisation

All tensors are float32 at the FFS input resolution (which may be a down-scaled version of the camera resolution).
The stereo network is frozen.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

log = logging.getLogger(__name__)

_IMPORTED_FROM: str | None = None


def _import_ffs(repo_dir: str):
    """Put the FFS repo on ``sys.path`` and import its modules (also registers the pickle aliases)."""
    global _IMPORTED_FROM
    repo_dir = str(Path(repo_dir).expanduser().resolve())
    if not (Path(repo_dir) / "core" / "foundation_stereo.py").is_file():
        raise FileNotFoundError(
            f"Fast-FoundationStereo repo not found at {repo_dir} (run scripts/setup_third_party.sh or set "
            "--policy.ffs_repo_dir)"
        )
    if _IMPORTED_FROM is None:
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        _IMPORTED_FROM = repo_dir
    elif _IMPORTED_FROM != repo_dir:
        log.warning("FFS already imported from %s; ignoring %s", _IMPORTED_FROM, repo_dir)
    import core.foundation_stereo as ffs_module  # noqa: F401
    from core.geometry import Combined_Geo_Encoding_Volume
    from core.submodule import (
        build_concat_volume_optimized_pytorch1,
        build_gwc_volume_optimized_pytorch1,
        context_upsample,
        disparity_regression,
    )

    return {
        "normalize_image": ffs_module.normalize_image,
        "Combined_Geo_Encoding_Volume": Combined_Geo_Encoding_Volume,
        "build_gwc": build_gwc_volume_optimized_pytorch1,
        "build_concat": build_concat_volume_optimized_pytorch1,
        "context_upsample": context_upsample,
        "disparity_regression": disparity_regression,
    }


def pad_to_multiple(x: Tensor, multiple: int = 32) -> tuple[Tensor, tuple[int, int]]:
    """Replicate-pad bottom/right so that H and W are multiples of ``multiple``; returns ``(padded, (pad_h, pad_w))``."""
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x, (pad_h, pad_w)


class FFSBeliefExtractor(nn.Module):
    """Wraps a serialised ``FastFoundationStereo`` model and exposes disparity beliefs."""

    def __init__(self, repo_dir: str, checkpoint: str, iters: int = 4, max_disp: int = 192, amp_dtype=torch.float16):
        super().__init__()
        self.fn = _import_ffs(repo_dir)
        ckpt = Path(checkpoint).expanduser()
        if not ckpt.is_file():
            raise FileNotFoundError(f"FFS checkpoint not found: {ckpt} (run scripts/setup_third_party.sh)")
        model = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        model.args.valid_iters = iters
        model.args.max_disp = max_disp
        self.model = model
        self.iters = iters
        self.max_disp = max_disp
        self.amp_dtype = amp_dtype
        self.corr_levels = int(model.args.corr_levels)
        self.corr_radius = int(model.args.corr_radius)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        # Fast-FoundationStereo prunes the refinement GRU and searches the cost-filtering blocks, so the actual
        # hidden / cost-volume widths differ from ``args``; read them off the modules.
        self.hidden_dim, self.volume_dim = self._probe_dims(model)

    @torch.no_grad()
    def _probe_dims(self, model) -> tuple[int, int]:
        hidden = volume = None
        try:
            hidden = int(model.update_block.disp_head.conv[0].in_channels)
        except Exception:
            pass
        try:
            classifier_first = model.classifier[0]
            conv = getattr(classifier_first, "conv", classifier_first)
            volume = int(conv.in_channels)
        except Exception:
            pass
        if hidden is None or volume is None:
            dev = next(model.parameters()).device
            dummy = torch.rand(1, 3, 64, 96, device=dev)
            out = self._forward(dummy, dummy, return_full_disp=False, _probe=True)
            hidden = int(out["hidden"].shape[1])
            volume = int(out["cost_volume"].shape[1])
        log.info("FFS belief extractor: hidden=%d volume=%d iters=%d max_disp=%d", hidden, volume, self.iters, self.max_disp)
        return hidden, volume

    def train(self, mode: bool = True):  # the stereo network stays frozen (BN/IN statistics included)
        super().train(mode)
        self.model.eval()
        return self

    @property
    def num_bins(self) -> int:
        return self.max_disp // 4

    @torch.no_grad()
    def forward(self, left: Tensor, right: Tensor, return_full_disp: bool = False, iters: int | None = None) -> dict[str, Tensor]:
        """left / right: ``[B,3,H,W]`` float RGB in [0, 1] (rectified, FFS input resolution)."""
        return self._forward(left, right, return_full_disp, iters=iters)

    def _forward(self, left: Tensor, right: Tensor, return_full_disp: bool, _probe: bool = False, iters: int | None = None) -> dict[str, Tensor]:
        m = self.model
        fn = self.fn
        n_iters = self.iters if iters is None else int(iters)
        B, _, H, W = left.shape
        img1, (pad_h, pad_w) = pad_to_multiple(left.float() * 255.0, 32)
        img2, _ = pad_to_multiple(right.float() * 255.0, 32)
        img1 = fn["normalize_image"](img1)
        img2 = fn["normalize_image"](img2)
        max_disp4 = self.max_disp // 4
        use_amp = bool(m.args.mixed_precision) and left.is_cuda

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=self.amp_dtype):
            out = m.feature(torch.cat([img1, img2], dim=0))
            features_left = [o[:B] for o in out]
            features_right = [o[B:] for o in out]
            stem_2x = m.stem_2(img1)

            normalize = bool(m.args.get("normalize", True)) if hasattr(m.args, "get") else bool(getattr(m.args, "normalize", True))
            gwc_volume = fn["build_gwc"](features_left[0], features_right[0], max_disp4, m.cv_group, normalize=normalize)
            left_tmp = m.proj_cmb(features_left[0])
            right_tmp = m.proj_cmb(features_right[0])
            concat_volume = fn["build_concat"](left_tmp, right_tmp, maxdisp=max_disp4)
            comb_volume = torch.cat([gwc_volume, concat_volume], dim=1)
            del gwc_volume, concat_volume, left_tmp, right_tmp
            comb_volume = m.corr_stem(comb_volume)
            comb_volume = m.corr_feature_att(comb_volume, features_left[0])
            comb_volume = m.cost_agg(comb_volume, features_left)  # filtered cost volume

            logits = m.classifier(comb_volume).squeeze(1)  # [B, D4, h, w]
            prob = F.softmax(logits.float(), dim=1)
            init_disp = fn["disparity_regression"](prob, max_disp4)  # [B,1,h,w]

            cnet_list = list(m.cnet(features_left[0], features_left[1], features_left[2]))
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            inp_list = [m.cam(x) * x for x in inp_list]
            att = [m.sam(x) for x in inp_list]

        dtype = m.dtype
        geo_fn = fn["Combined_Geo_Encoding_Volume"](
            features_left[0].to(dtype), features_right[0].to(dtype), comb_volume.to(dtype), num_levels=self.corr_levels
        )
        b, _, h, w = features_left[0].shape
        coords = torch.arange(w, dtype=torch.float, device=init_disp.device).reshape(1, 1, w, 1).repeat(b, h, 1, 1)
        disp = init_disp.to(dtype)
        hidden = net_list[0]
        mask_feat_4 = None
        geo_feat = None
        for _ in range(n_iters):
            disp = disp.detach()
            geo_feat = geo_fn(disp, coords, dx=m.dx, low_memory=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=self.amp_dtype):
                net_list, mask_feat_4, delta_disp = m.update_block(net_list, inp_list, geo_feat.to(dtype), disp, att)
            disp = disp + delta_disp.to(dtype)
            hidden = net_list[0]
        if geo_feat is None:  # K = 0: still evaluate the local profile at the initial disparity
            geo_feat = geo_fn(disp, coords, dx=m.dx, low_memory=True)

        # ---- belief statistics (float32), Eq. 2 -------------------------------------------
        prob = prob.float()
        bins = torch.arange(max_disp4, device=prob.device, dtype=torch.float32).view(1, -1, 1, 1)
        mu = (prob * bins).sum(1, keepdim=True)
        var = (prob * (bins - mu) ** 2).sum(1, keepdim=True)
        sigma = var.clamp_min(1e-6).sqrt()
        entropy = -(prob * prob.clamp_min(1e-8).log()).sum(1, keepdim=True) / math.log(max_disp4)  # H / log D_b
        # local correlation profile at the refined disparity: level-0 init_corr slice of geo_feat
        # (layout per level: [geo_volume taps (C * n_tap), init_corr taps (n_tap)])
        n_tap = 2 * self.corr_radius + 1
        start = n_tap * int(comb_volume.shape[1])
        corr_local = geo_feat[:, start : start + n_tap].float()  # [B, 2r+1, h, w], normalised correlations
        peak_raw = F.softmax(corr_local / 0.1, dim=1).max(dim=1, keepdim=True).values
        peak = ((peak_raw - 1.0 / n_tap) / (1.0 - 1.0 / n_tap)).clamp(0, 1)  # 0 = flat profile, 1 = unique match
        disp = disp.float()
        xr = coords.permute(0, 3, 1, 2).float() - disp  # matched x position in the right view
        occl = (xr < 0).float()  # the match leaves the right image: left-right consistency fails
        conf = ((1.0 - entropy).clamp(0, 1) * peak * (1.0 - occl)).clamp(0, 1)

        result = {
            "hidden": hidden.float(),
            "disp": disp,
            "init_disp": init_disp.float(),
            "mu": mu,
            "sigma": sigma,
            "entropy": entropy,
            "peak": peak,
            "occl": occl,
            "conf": conf,
            "pad": torch.tensor([pad_h, pad_w]),
        }
        h_valid, w_valid = math.ceil(H / 4), math.ceil(W / 4)  # crop the padded region (quarter resolution)
        for k in ("hidden", "disp", "init_disp", "mu", "sigma", "entropy", "peak", "occl", "conf"):
            result[k] = result[k][..., :h_valid, :w_valid].contiguous()
        if _probe:
            result["cost_volume"] = comb_volume.float()[..., :h_valid, :w_valid].contiguous()
        if return_full_disp:
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=self.amp_dtype):
                xspx = m.spx_2_gru(mask_feat_4.to(dtype), stem_2x.to(dtype)) if mask_feat_4 is not None else None
                if xspx is not None:
                    spx_pred = F.softmax(m.spx_gru(xspx), 1)
                    disp_full = fn["context_upsample"](disp * 4.0, spx_pred.float()).unsqueeze(1)
                else:
                    disp_full = F.interpolate(disp * 4.0, scale_factor=4, mode="bilinear", align_corners=False)
            result["disp_full"] = disp_full.float()[..., :H, :W].contiguous()
        return result


def belief_feature_tensor(beliefs: dict[str, Tensor], num_bins: int) -> Tensor:
    """The scalar belief statistics of Eq. 2 as a normalised ``[B, 6, h, w]`` tensor:
    ``[mu, sigma, H, d_K, |d_K - mu|, pi]`` (disparities in units of the bin count)."""
    scale = float(num_bins)
    feats = [
        beliefs["mu"] / scale,
        beliefs["sigma"] / scale,
        beliefs["entropy"],
        beliefs["disp"] / scale,
        (beliefs["disp"] - beliefs["mu"]).abs() / scale,
        beliefs["peak"],
    ]
    return torch.cat(feats, dim=1)


def collapse_beliefs_to_point(beliefs: dict[str, Tensor]) -> dict[str, Tensor]:
    """Ablation D (Table 3): keep the refined disparity ``d_K`` and drop every other belief statistic.

    hidden -> 0, mu -> d_K (so |d_K - mu| = 0), sigma -> 1 bin, entropy -> 0, peak -> 1, occlusion -> 0,
    confidence -> 1.  The interface then sees a point estimate without uncertainty.
    """
    out = dict(beliefs)
    d = beliefs["disp"]
    out["hidden"] = torch.zeros_like(beliefs["hidden"])
    out["mu"] = d.clone()
    out["sigma"] = torch.ones_like(d)
    out["entropy"] = torch.zeros_like(d)
    out["peak"] = torch.ones_like(d)
    out["occl"] = torch.zeros_like(d)
    out["conf"] = torch.ones_like(d)
    return out
