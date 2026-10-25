#!/usr/bin/env python
"""pi0.5 + StereoLift policy (``policy.type=pi05_stereolift``).

Prefix layout (PaliGemma):  [enhanced left SigLIP tokens] [M relation tokens] [auxiliary camera tokens ...] [language]

The stereo pair is consumed twice: the left image by SigLIP (224x224, resize-with-pad, as in pi0.5) and both images
by the frozen Fast-FoundationStereo at native or scaled resolution to obtain the disparity belief.  The backbone,
the action expert, the flow-matching objective and the 50-step action chunks are unchanged.
"""

from __future__ import annotations

import builtins
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch, pad_vector, resize_with_pad_torch
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.utils.import_utils import require_package

from .configuration_pi05_stereolift import PI05StereoLiftConfig
from .ffs_belief import FFSBeliefExtractor, collapse_beliefs_to_point
from .modules import StereoLiftAdapter
from .rectification import StereoRectifier

log = logging.getLogger(__name__)


class PI05StereoLiftPytorch(PI05Pytorch):
    """pi0.5 core with the StereoLift interface between SigLIP and PaliGemma."""

    def __init__(self, config: PI05StereoLiftConfig, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)
        token_dim = self.paligemma_with_expert.paligemma.config.text_config.hidden_size
        self.ffs = FFSBeliefExtractor(
            repo_dir=config.ffs_repo_dir, checkpoint=config.ffs_checkpoint, iters=config.ffs_iters, max_disp=config.ffs_max_disp
        )
        state_dim = config.max_state_dim if config.relation_use_state else None
        self.adapter = StereoLiftAdapter(config, token_dim, self.ffs.hidden_dim, config.ffs_max_disp // 4, state_dim)
        self._stereo: dict[str, Any] | None = None
        self.stereo_aux: dict[str, Tensor] = {}

    def set_stereo_inputs(self, beliefs, native_hw, ffs_scale, state, H_left=None) -> None:
        """Called by the policy wrapper before every forward with the FFS beliefs of the current batch."""
        self._stereo = {"beliefs": beliefs, "native_hw": native_hw, "ffs_scale": ffs_scale, "state": state, "H_left": H_left}

    def embed_prefix(self, images, img_masks, tokens, masks):
        if len(images) < 1 or len(images) != len(img_masks):
            raise ValueError(f"Expected aligned image/mask lists; got {len(images)} images, {len(img_masks)} masks")
        st = self._stereo
        if st is None or st.get("beliefs") is None:
            raise RuntimeError("StereoLift: stereo beliefs were not prepared for this batch (set_stereo_inputs)")

        embs = [self._apply_checkpoint(self.paligemma_with_expert.embed_image, img) for img in images]
        lang = self._apply_checkpoint(self.paligemma_with_expert.embed_language_tokens, tokens)

        li = list(self.config.siglip_image_keys).index(self.config.stereo_left_key)
        fused_left, relation, aux = self.adapter(
            embs[li], lang, masks, st["beliefs"], tuple(st["native_hw"]), float(st["ffs_scale"]), st.get("state"), H_left=st.get("H_left")
        )
        self.stereo_aux = aux

        seq, pads = [], []
        for i, (emb, mask) in enumerate(zip(embs, img_masks, strict=True)):
            if i == li:
                emb = fused_left
            seq.append(emb)
            pads.append(mask[:, None].expand(-1, emb.shape[1]))
            if i == li and relation is not None:  # the relation tokens follow the enhanced left-view tokens
                seq.append(relation.to(emb.dtype))
                pads.append(torch.ones(relation.shape[:2], dtype=torch.bool, device=relation.device))
        seq.append(lang)
        pads.append(masks)
        embeddings = torch.cat(seq, dim=1)
        pad_masks = torch.cat(pads, dim=1)
        att_masks = torch.zeros_like(pad_masks, dtype=torch.bool)  # full attention inside the prefix, as in pi0.5
        return embeddings, pad_masks, att_masks


class PI05StereoLiftPolicy(PI05Policy):
    """pi0.5 policy with the StereoLift stereo interface."""

    config_class = PI05StereoLiftConfig
    name = "pi05_stereolift"

    def __init__(self, config: PI05StereoLiftConfig, **kwargs):
        require_package("transformers", extra="pi")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = PI05StereoLiftPytorch(config, rtc_processor=self.rtc_processor)
        self.rectifier = StereoRectifier(config.rectify_mode, config.rectify_right_dx, config.rectify_right_dy, config.rectify_homography_path)
        if config.freeze_policy_backbone:
            keep = [p for p in self.model.adapter.parameters() if p.requires_grad]
            for p in self.model.parameters():
                p.requires_grad = False
            for p in keep:
                p.requires_grad = True
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(config.device)
        self.reset()

    # ------------------------------------------------------------------ optim
    def get_optim_params(self):
        """One parameter group for the pretrained policy and one (scaled learning rate) for the interface."""
        cfg = self.config
        adapter = [p for p in self.model.adapter.parameters() if p.requires_grad]
        adapter_ids = {id(p) for p in adapter}
        rest = [p for p in self.parameters() if p.requires_grad and id(p) not in adapter_ids]
        if not adapter or cfg.adapter_lr_scale == 1.0:
            return super().get_optim_params()
        groups = [{"params": rest}, {"params": adapter, "lr": cfg.optimizer_lr * cfg.adapter_lr_scale}]
        log.info("StereoLift optimizer groups: policy %d tensors, interface %d tensors at lr x%g", len(rest), len(adapter), cfg.adapter_lr_scale)
        return groups

    # ---------------------------------------------------------------- stereo
    @staticmethod
    def _to_bchw(img: Tensor) -> Tensor:
        if img.ndim != 4:
            raise ValueError(f"Expected a 4D image tensor, got {tuple(img.shape)}")
        if img.shape[1] != 3 and img.shape[-1] == 3:
            img = img.permute(0, 3, 1, 2)
        return img.contiguous()

    def _prepare_stereo(self, batch: dict[str, Tensor]) -> None:
        """Rectify the pair, run the frozen stereo model and hand the beliefs to the core model."""
        cfg = self.config
        for key in (cfg.stereo_left_key, cfg.stereo_right_key):
            if key not in batch:
                raise ValueError(f"pi05_stereolift requires {key!r} in the batch; got {sorted(batch.keys())}")
        device = next(self.parameters()).device
        left = self._to_bchw(batch[cfg.stereo_left_key]).to(device=device, dtype=torch.float32)
        right = self._to_bchw(batch[cfg.stereo_right_key]).to(device=device, dtype=torch.float32)
        if left.max() > 1.5:  # uint8 range
            left, right = left / 255.0, right / 255.0
        left, right = self.rectifier(left, right)
        native_hw = (int(left.shape[-2]), int(left.shape[-1]))

        state = None
        if cfg.relation_use_state and cfg.state_key in batch:
            state = pad_vector(batch[cfg.state_key].to(device=device, dtype=torch.float32), cfg.max_state_dim)

        scale = cfg.ffs_input_scale
        if scale != 1.0:
            hs = max(int(round(native_hw[0] * scale / 8.0)) * 8, 32)
            ws = max(int(round(native_hw[1] * scale / 8.0)) * 8, 32)
            scale = ws / native_hw[1]
            left_s = F.interpolate(left, size=(hs, ws), mode="bilinear", align_corners=False, antialias=True)
            right_s = F.interpolate(right, size=(hs, ws), mode="bilinear", align_corners=False, antialias=True)
        else:
            left_s, right_s = left, right
        beliefs = self.model.ffs(left_s, right_s)
        if cfg.belief_point_only:
            beliefs = collapse_beliefs_to_point(beliefs)

        H_left = None
        if cfg.align_geometry_to_raw and self.rectifier.mode == "homography":
            H_left = self.rectifier.H_left.to(device)
            if self.rectifier.native_size is not None and (native_hw[1], native_hw[0]) != self.rectifier.native_size:
                sx, sy = native_hw[1] / self.rectifier.native_size[0], native_hw[0] / self.rectifier.native_size[1]
                S = torch.diag(torch.tensor([sx, sy, 1.0], device=device))
                H_left = S @ H_left @ torch.diag(torch.tensor([1.0 / sx, 1.0 / sy, 1.0], device=device))
        self.model.set_stereo_inputs(beliefs, native_hw, scale, state, H_left=H_left)

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Resize-with-pad + [-1, 1] normalisation of the SigLIP views only (left view first)."""
        device = next(self.parameters()).device
        images, masks = [], []
        template = None
        for key in self.config.siglip_image_keys:
            if key in batch:
                img = self._to_bchw(batch[key]).to(device=device, dtype=torch.float32)
                if img.max() > 1.5:
                    img = img / 255.0
                img = img.permute(0, 2, 3, 1)
                if img.shape[1:3] != tuple(self.config.image_resolution):
                    img = resize_with_pad_torch(img, *self.config.image_resolution)
                img = (img * 2.0 - 1.0).permute(0, 3, 1, 2)
                images.append(img)
                masks.append(torch.ones(img.shape[0], dtype=torch.bool, device=device))
                template = img
            elif key == self.config.stereo_left_key:
                raise ValueError(f"Stereo view {key!r} missing from batch")
            else:
                if template is None:
                    raise ValueError("Cannot pad a missing camera before any real image was seen")
                images.append(torch.ones_like(template) * -1)
                masks.append(torch.zeros(template.shape[0], dtype=torch.bool, device=device))
        return images, masks

    # --------------------------------------------------------------- loading
    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        strict: bool = False,
        **kwargs: Any,
    ) -> T:
        # A monocular pi0.5 checkpoint has no interface weights: load the shared weights and initialise the rest.
        return super().from_pretrained(pretrained_name_or_path, config=config, strict=strict, **kwargs)

    # ------------------------------------------------------------- inference
    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self._prepare_stereo(batch)
        return super().predict_action_chunk(batch, **kwargs)

    @property
    def last_stereo_confidence(self) -> Tensor | None:
        """Mean stereo confidence c-bar of the last processed batch, the signal used for failure monitoring."""
        return self.model.stereo_aux.get("mean_conf")

    # -------------------------------------------------------------- training
    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        self._prepare_stereo(batch)
        loss, loss_dict = super().forward(batch, reduction=reduction)  # flow-matching loss only
        aux = self.model.stereo_aux
        if "mean_conf" in aux:
            loss_dict["stereo_confidence"] = aux["mean_conf"].float().mean().item()
        if "relation_attn_entropy" in aux:
            loss_dict["relation_attn_entropy"] = float(aux["relation_attn_entropy"])
        return loss, loss_dict
