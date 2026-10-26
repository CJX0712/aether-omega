"""
AETHER-Ω 主干 —— 集 2026 年收敛配方之大成的 Transformer
=======================================================

一层 AetherBlock（Pre-Norm 残差，2026 全系统一写法）：

    h  = x + Attention( RMSNorm(x) )
    x' = h + FFN_or_MoE( RMSNorm(h) )

选择 Pre-Norm 而非 Post-Norm 的理由是数学的：Pre-Norm 下残差流是**恒等通路**，
第 L 层的输出可展开为 x_L = x_0 + Σ_l f_l(norm(x_l))，
梯度回传到 x_0 有一条不经过任何权重矩阵的直通路径（∂x_L/∂x_0 ⊇ I），
因此极深网络也能稳定训练；Post-Norm 没有这条保底通路。

注意力排布：局部 : 全局 = 3 : 1
--------------------------------
每 4 层里 3 层用滑动窗口（O(L·w)）、1 层用全注意力（O(L²)）。
Gemma 3 / Qwen 3 / Mistral Small 3.1 都采用这个比例。直觉是：
大多数语言现象是局部的（语法、搭配），少数需要长程（共指、主题）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from aether.config import ModelConfig
from aether.modules.attention import KVCache, MultiheadLatentAttention
from aether.modules.ffn import SwiGLU
from aether.modules.moe import DeepSeekMoE
from aether.modules.mtp import MTPStack
from aether.modules.norm import RMSNorm

__all__ = ["AetherBlock", "AetherModel"]


class AetherBlock(nn.Module):
    """单个 AETHER 层：MLA（局部或全局）+ SwiGLU/DeepSeekMoE。"""

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.is_global = cfg.is_global_layer(layer_idx)
        self.is_moe = cfg.is_moe_layer(layer_idx)
        self.window = None if self.is_global else int(cfg.local_window)

        self.attn_norm = RMSNorm(cfg.dim, cfg.rms_norm_eps)
        self.attn = MultiheadLatentAttention(cfg, layer_idx, mode="absorb")
        # 运行时证书采集开关：打开时缓存注意力的输入，供 MuonClip 测量 logit 上界。
        # 默认关闭，零开销；只在需要测量那一步打开。
        self.capture = False
        self.last_attn_input: Optional[torch.Tensor] = None

        self.ffn_norm = RMSNorm(cfg.dim, cfg.rms_norm_eps)
        if self.is_moe:
            self.ffn = DeepSeekMoE(cfg, cfg.dropout)
        else:
            self.ffn = SwiGLU(cfg.dim, cfg.ffn_hidden_size, cfg.dropout)

    def forward(self, x: torch.Tensor, kv_cache=None, attn_mode=None, start_pos=0):
        xn = self.attn_norm(x)
        if self.capture:
            self.last_attn_input = xn.detach()
        h = x + self.attn(
            xn, kv_cache=kv_cache, window=self.window,
            mode=attn_mode, start_pos=start_pos,
        )
        return h + self.ffn(self.ffn_norm(h))


class AetherModel(nn.Module):
    """AETHER-Ω 语言模型。

    Args:
        cfg: ModelConfig。
        attn_mode: 全局默认注意力实现（"absorb" 高效 / "naive" 参考实现）。
    """

    def __init__(self, cfg: ModelConfig, attn_mode: str = "absorb") -> None:
        super().__init__()
        self.cfg = cfg
        self.attn_mode = attn_mode

        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([AetherBlock(cfg, i) for i in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.dim, cfg.rms_norm_eps)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.head.weight = self.embed.weight  # 权重共享：省 V·D 参数，且经验上更稳

        self.mtp = MTPStack(cfg, self.embed, self.head) if cfg.mtp_depth > 0 else None
        self.mtp_loss_weight = float(cfg.mtp_loss_weight)

        self.apply(self._init_weights)
        # 残差分支按深度缩放（GPT-2 式 residual init）：稳定深层训练起点
        for name, p in self.named_parameters():
            if name.endswith("w_o.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * len(self.layers)) ** 0.5)

    # ------------------------------------------------------------------
    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[KVCache]] = None,
        attn_mode: Optional[str] = None,
        start_pos: int = 0,
    ) -> Dict[str, Any]:
        """前向。带 targets 时返回 loss 分解（主损失 + MTP 损失）。"""
        mode = attn_mode or self.attn_mode
        B, L = idx.shape
        x = self.embed(idx)

        for i, layer in enumerate(self.layers):
            x = layer(
                x,
                kv_cache=kv_caches[i] if kv_caches is not None else None,
                attn_mode=mode,
                start_pos=start_pos,
            )

        x = self.final_norm(x)
        logits = self.head(x)

        out: Dict[str, Any] = {"logits": logits}
        if targets is None:
            return out

        main = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)), targets[:, 1:].reshape(-1)
        )
        out["loss_main"] = main
        loss = main

        if self.mtp is not None and self.mtp.depth > 0:
            mtp_outs = self.mtp(x, idx)
            mtp_losses = []
            for lg, tg in mtp_outs:
                n = min(lg.shape[1], tg.shape[1])
                if n <= 0:
                    continue
                mtp_losses.append(
                    F.cross_entropy(lg[:, :n].reshape(-1, lg.size(-1)), tg[:, :n].reshape(-1))
                )
            if mtp_losses:
                lm = torch.stack(mtp_losses).mean()
                out["loss_mtp"] = lm
                loss = loss + self.mtp_loss_weight * lm

        out["loss"] = loss
        return out

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """自回归生成（带 MLA KV cache）。

        提供 seed 时采样确定可复现；temperature→0 时退化为贪心。
        """
        was_training = self.training
        self.eval()
        g = None
        if seed is not None:
            g = torch.Generator(device=idx.device).manual_seed(seed)

        B = idx.shape[0]
        caches = [
            KVCache(B, idx.shape[1] + max_new_tokens, self.cfg.kv_lora_rank,
                    self.cfg.qk_rope_head_dim, dtype=idx.dtype if idx.dtype.is_floating_point
                    else torch.float32, device=idx.device)
            for _ in self.layers
        ]

        # 先 prefill 整个 prompt（位置从 0 开始）
        start = idx.shape[1] - 1 if idx.shape[1] > 1 else 0
        if idx.shape[1] > 1:
            self.forward(idx[:, :-1], kv_caches=caches, start_pos=0)
        cur = idx[:, -1:]

        for _ in range(max_new_tokens):
            o = self.forward(cur, kv_caches=caches, start_pos=start)
            start += 1
            logits = o["logits"][:, -1, :] / max(temperature, 1e-8)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, -1:], float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            if temperature < 1e-8 or g is None:
                nxt = probs.argmax(dim=-1, keepdim=True)
            else:
                nxt = torch.multinomial(probs, 1, generator=g)
            idx = torch.cat([idx, nxt], dim=1)
            cur = nxt

        if was_training:
            self.train()
        return idx

    # ------------------------------------------------------------------
    def num_parameters(self, *, active_only: bool = False) -> int:
        """参数统计。active_only=True 时按 MoE 的 top-k 折算**每 token 激活参数**。"""
        if not active_only:
            return sum(p.numel() for p in self.parameters())
        total = 0
        for n, p in self.named_parameters():
            if "experts." in n:
                e = int(n.split("experts.")[1].split(".")[0])
                if e < self.cfg.n_routed_experts:
                    total += p.numel() * self.cfg.n_activated_experts / self.cfg.n_routed_experts
                    continue
            total += p.numel()
        return int(total)

    def layer_kind_report(self) -> List[Dict[str, Any]]:
        """每层是全局/局部、dense/MoE 的清单——供文档与证书报告核对架构正确性。"""
        rows = []
        for i, ly in enumerate(self.layers):
            rows.append(
                {
                    "layer": i,
                    "attention": "global" if ly.is_global else f"local(w={ly.window})",
                    "ffn": "MoE" if ly.is_moe else "SwiGLU",
                }
            )
        return rows
