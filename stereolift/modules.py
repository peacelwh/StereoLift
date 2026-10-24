"""The StereoLift interface: geometric token embedding and geometric relation tokens.

Both modules operate on pi0.5's SigLIP token grid (16x16 = 256 tokens) and on the FFS quarter-resolution belief map.
Their output projections are zero-initialised, so fine-tuning starts from the pretrained monocular model.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .ffs_belief import belief_feature_tensor
from .lifting import (
    AffineLifter,
    FourierFeatures,
    cell_token_index,
    pool_to_token_grid,
    quarter_grid_native_coords,
    warp_grid_to_raw,
)


def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    """x [B,L,D], mask [B,L] -> [B,D]."""
    m = mask.to(x.dtype).unsqueeze(-1)
    return (x * m).sum(1) / m.sum(1).clamp_min(1.0)


class FiLM(nn.Module):
    """Feature-wise linear modulation, zero-initialised (identity at the start of training)."""

    def __init__(self, cond_dim: int, dim: int):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class GeometricTokenEmbedding(nn.Module):
    """Eq. 6:  t_i <- t_i + W_o LN( MLP([b_i; c_i]) + W_p phi(X'_i) ),  W_o <- 0.

    The pooled belief (GRU state, moments, sharpness) *and* the pooled confidence are inputs: the network learns how
    far to trust each cell instead of being told by an explicit gate (Proposition 2).  Zero-initialising W_o leaves
    every visual embedding unchanged at initialisation.
    """

    def __init__(self, token_dim: int, belief_dim: int, hidden_dim: int, fourier: FourierFeatures):
        super().__init__()
        self.fourier = fourier
        self.belief_mlp = nn.Sequential(nn.Linear(belief_dim + 1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.pos_proj = nn.Linear(fourier.out_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, token_dim, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, tokens: Tensor, belief: Tensor, pos: Tensor, conf: Tensor) -> Tensor:
        """tokens [B,N,D]; belief [B,N,C]; pos [B,N,3]; conf [B,N] -> enhanced tokens [B,N,D]."""
        h = self.norm(self.belief_mlp(torch.cat((belief, conf[..., None]), -1)) + self.pos_proj(self.fourier(pos)))
        return tokens + self.out(h).to(tokens.dtype)


class RelationResampler(nn.Module):
    """Eqs. 7-8: M language- and state-conditioned queries read the native-resolution belief map in one
    cross-attention layer.

        z_j = LN( MLP([b_j; c_j]) + W_p phi(X'_j) + W_t t~_{i(j)} ),   alpha_mj = softmax_j( <W_q q_m, W_k z_j> / sqrt(d_k) - lambda (1 - c_j) )

    where t~_{i(j)} is the enhanced SigLIP token whose patch contains cell j.  The M outputs are projected by a
    zero-initialised layer and appended to the visual prefix.
    """

    def __init__(
        self,
        token_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_tokens: int,
        belief_dim: int,
        fourier: FourierFeatures,
        key_penalty: float,
        state_dim: int | None,
        semantic_keys: bool = True,
        max_keys: int = 8192,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_tokens, self.num_heads, self.head_dim = num_tokens, num_heads, hidden_dim // num_heads
        self.key_penalty = key_penalty
        self.max_keys = max_keys
        self.fourier = fourier
        self.queries = nn.Parameter(torch.randn(num_tokens, hidden_dim) * 0.02)
        self.lang_film = FiLM(token_dim, hidden_dim)
        self.state_film = FiLM(state_dim, hidden_dim) if state_dim else None
        self.key_mlp = nn.Sequential(nn.Linear(belief_dim + 1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.key_pos = nn.Linear(fourier.out_dim, hidden_dim)
        self.key_tok = nn.Linear(token_dim, hidden_dim, bias=False) if semantic_keys else None
        self.key_norm = nn.LayerNorm(hidden_dim)
        self.q_norm = nn.LayerNorm(hidden_dim)
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mlp = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 2 * hidden_dim), nn.SiLU(), nn.Linear(2 * hidden_dim, hidden_dim))
        self.out = nn.Linear(hidden_dim, token_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)  # the relation tokens start as exact zero vectors
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: Tensor,  # [B,N,D] enhanced SigLIP tokens (semantics of each patch)
        fine_feat: Tensor,  # [B,K,C] belief features on the FFS grid
        fine_pos: Tensor,  # [B,K,3] lifted coordinates
        fine_conf: Tensor,  # [B,K] confidence
        cell_tok: Tensor,  # [K] long: token index i(j) of every cell
        language: Tensor,  # [B,L,D] language token embeddings
        language_mask: Tensor,  # [B,L]
        state: Tensor | None,  # [B,S] normalised proprioceptive state
    ) -> tuple[Tensor, dict[str, Tensor]]:
        b = tokens.shape[0]
        if fine_feat.shape[1] > self.max_keys:
            stride = math.ceil(fine_feat.shape[1] / self.max_keys)
            off = torch.randint(0, stride, (1,)).item() if self.training else 0
            fine_feat, fine_pos, fine_conf, cell_tok = (
                t[..., off::stride, :] if t.dim() == 3 else t[..., off::stride] for t in (fine_feat, fine_pos, fine_conf, cell_tok)
            )
        q = self.queries[None].expand(b, -1, -1)
        q = self.lang_film(q, masked_mean(language.to(q.dtype), language_mask))
        if self.state_film is not None and state is not None:
            q = self.state_film(q, state.to(q.dtype))
        k = self.key_mlp(torch.cat((fine_feat, fine_conf[..., None]), -1)) + self.key_pos(self.fourier(fine_pos))
        if self.key_tok is not None:
            sem = self.key_tok(tokens.to(k.dtype))  # [B,N,hidden]
            k = k + sem.gather(1, cell_tok[None, :, None].expand(b, -1, sem.shape[-1]))
        kn = self.key_norm(k)
        qh = self.q(self.q_norm(q)).view(b, self.num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        kh = self.k(kn).view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        vh = self.v(kn).view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        logits = qh @ kh.transpose(-1, -2) / math.sqrt(self.head_dim)
        if self.key_penalty > 0:
            logits = logits - self.key_penalty * (1.0 - fine_conf)[:, None, None, :].to(logits.dtype)
        attn = logits.softmax(-1)
        q = q + (attn @ vh).transpose(1, 2).reshape(b, self.num_tokens, -1)
        q = q + self.mlp(q)
        a = attn.mean(1)  # [B,M,K] head-averaged attention
        aux = {"query_pos": (a @ fine_pos).detach(), "attn_entropy": (-(a * a.clamp_min(1e-8).log()).sum(-1)).mean().detach()}
        if not self.training:  # for visualisation (scripts/visualize_beliefs.py)
            aux["attn"] = a.detach()
        return self.out(self.dropout(q)), aux


class StereoLiftAdapter(nn.Module):
    """Belief lifting -> confidence-weighted pooling -> geometric token embedding -> relation tokens."""

    def __init__(self, cfg, token_dim: int, ffs_hidden_dim: int, num_bins: int, state_dim: int | None):
        super().__init__()
        self.cfg = cfg
        self.token_dim = token_dim
        self.num_bins = num_bins
        hidden = cfg.adapter_hidden_dim
        heads = cfg.adapter_num_heads
        self.belief_dim = ffs_hidden_dim + 6  # [h_K; mu; sigma; H; d_K; |d_K - mu|; pi]
        self.lifter = AffineLifter(cfg.lift_fov_deg, cfg.lift_scale)
        self.fourier = FourierFeatures(cfg.lift_fourier_bands, max_log2_freq=float(cfg.lift_fourier_max_log2))
        self.gte = GeometricTokenEmbedding(token_dim, self.belief_dim, hidden, self.fourier) if cfg.use_geometric_embedding else None
        self.relation = (
            RelationResampler(
                token_dim, hidden, heads, cfg.num_relation_tokens, self.belief_dim, self.fourier, cfg.key_penalty,
                state_dim if cfg.relation_use_state else None, bool(cfg.relation_semantic_keys), int(cfg.relation_max_keys),
                cfg.adapter_dropout,
            )
            if cfg.num_relation_tokens > 0
            else None
        )

    def _lift_grid(self, beliefs: dict[str, Tensor], native_hw: tuple[int, int], ffs_scale: float) -> tuple[Tensor, Tensor, Tensor]:
        """Returns belief features [B,C,h,w], lifted coordinates [B,3,h,w] and confidence [B,1,h,w] on the FFS grid."""
        hidden = beliefs["hidden"]
        b, _, h, w = hidden.shape
        u, v = quarter_grid_native_coords(h, w, ffs_scale, hidden.device)
        u, v = u[None].expand(b, -1, -1), v[None].expand(b, -1, -1)
        d_native = beliefs["disp"][:, 0] * 4.0 / ffs_scale  # [B,h,w] disparity in native pixels (common pixel scale)
        pos = self.lifter(u, v, d_native, float(native_hw[1]), float(native_hw[0]))
        feats = torch.cat((hidden, belief_feature_tensor(beliefs, self.num_bins)), dim=1)
        return feats, pos.permute(0, 3, 1, 2), beliefs["conf"]

    def forward(
        self,
        left: Tensor,  # [B,N,D] SigLIP tokens of the left view
        language: Tensor,  # [B,L,D]
        language_mask: Tensor,  # [B,L]
        beliefs: dict[str, Tensor],
        native_hw: tuple[int, int],
        ffs_scale: float,
        state: Tensor | None,
        H_left: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, dict[str, Tensor]]:
        """Returns (enhanced left tokens [B,N,D], relation tokens [B,M,D] | None, auxiliary tensors)."""
        language = language.to(left.dtype)
        feats, pos, conf = self._lift_grid(beliefs, native_hw, ffs_scale)
        if H_left is not None:
            # FFS ran on the rectified pair; SigLIP encodes the stored left image -> move every cell map to the
            # stored frame so that pooled geometry and patch semantics refer to the same pixels.
            c_in = feats.shape[1]
            warped = warp_grid_to_raw(torch.cat((feats, pos.to(feats.dtype), conf.to(feats.dtype)), 1), H_left, ffs_scale)
            feats, pos, conf = warped[:, :c_in], warped[:, c_in : c_in + 3], warped[:, c_in + 3 :].clamp(0.0, 1.0)
        pooled_feat, pooled_conf = pool_to_token_grid(feats, conf, native_hw)
        pooled_pos, _ = pool_to_token_grid(pos, conf, native_hw)
        tok_feat = pooled_feat.flatten(2).transpose(1, 2)  # [B,N,C]  b-bar_i
        tok_pos = pooled_pos.flatten(2).transpose(1, 2)  # [B,N,3]  X'-bar_i
        tok_conf = pooled_conf.flatten(1)  # [B,N]    c-bar_i
        fused = self.gte(left, tok_feat, tok_pos, tok_conf) if self.gte is not None else left
        aux: dict[str, Tensor] = {"token_conf": tok_conf, "token_pos": tok_pos, "mean_conf": conf.flatten(1).mean(-1)}
        extra = None
        if self.relation is not None:
            h, w = conf.shape[-2:]
            idx = cell_token_index(h, w, ffs_scale, native_hw, device=fused.device)
            extra, rel_aux = self.relation(
                fused, feats.flatten(2).transpose(1, 2), pos.flatten(2).transpose(1, 2), conf.flatten(1), idx,
                language, language_mask, state,
            )
            aux.update({f"relation_{k}": v for k, v in rel_aux.items()})
        return fused, extra, aux
