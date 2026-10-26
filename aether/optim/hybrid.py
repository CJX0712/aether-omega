"""
混合优化器：Muon（二维隐藏矩阵）+ AdamW（其余一切）
====================================================

为什么必须混合
--------------
Muon 的正交化只对"矩阵"有意义——它利用的是矩阵的奇异值结构。
以下参数没有这种结构，硬上 Muon 只会自找麻烦：

    · embedding / output head：查表型参数，每行只在被采样到时才更新，
      正交化会破坏稀疏更新的语义（还会把没见过的 token 行搅乱）。
    · RMSNorm 的 weight / bias：一维向量，"方向"没有意义。
    · 标量 bias：同上一维。

所以工业界真实配方（Moonlight、Kimi K2、GLM-5）一律是：
    **隐藏层的 2D 权重矩阵 → Muon；其余 → AdamW。**

零重写原则
----------
AdamW 我们**不自己写**：直接复用 torch.optim.AdamW——那是 Meta 官方经过
无数次 correctness 与性能打磨的实现，重写一个只会更差。
只写"路由"这一层真正的增量逻辑。

自动路由规则
------------
    参数 → Muon   当且仅当： ndim == 2 且 不在 {embed, head} 中
                            且 min(shape) >= min_matrix_dim（太瘦的矩阵正交化收益为负）
    其余   → AdamW

"太瘦的矩阵"这个门槛是必要的：对一个 1×N 的向量式矩阵做 5 步 Newton-Schulz，
相当于把整个更新压成一个方向，比原梯度信息量还少。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn

from aether.config import OptimConfig
from aether.optim.muon import Muon

__all__ = ["split_params", "build_hybrid_optimizer", "HybridOptimizer"]


def split_params(
    model: nn.Module,
    *,
    min_matrix_dim: int = 32,
    muon_name_blacklist: Tuple[str, ...] = ("embed", "head"),
) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """把模型参数切成 (Muon 组, AdamW 组)。返回顺序稳定，保证跨次运行一致。"""
    muon_p: List[nn.Parameter] = []
    adamw_p: List[nn.Parameter] = []
    seen = set()

    for name, p in sorted(model.named_parameters()):
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        is_matrix = p.dim() == 2
        blacklisted = any(k in name for k in muon_name_blacklist)
        too_thin = is_matrix and min(p.shape) < min_matrix_dim
        if is_matrix and not blacklisted and not too_thin:
            muon_p.append(p)
        else:
            adamw_p.append(p)
    return muon_p, adamw_p


class HybridOptimizer:
    """把 Muon 与 AdamW 打包成单一入口（step / zero_grad / lr 调度 / state_dict）。"""

    def __init__(self, muon: Muon, adamw: torch.optim.Optimizer) -> None:
        self.muon = muon
        self.adamw = adamw

    @torch.no_grad()
    def step(self) -> None:
        self.muon.step()
        self.adamw.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def set_lr_scale(self, scale: float, *, base_muon: float, base_adamw: float) -> None:
        """按 schedule 因子同步缩放两个优化器的学习率。"""
        for g in self.muon.param_groups:
            g["lr"] = base_muon * scale
        for g in self.adamw.param_groups:
            g["lr"] = base_adamw * scale

    def state_dict(self) -> Dict[str, Any]:
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.muon.load_state_dict(sd["muon"])
        self.adamw.load_state_dict(sd["adamw"])

    def info(self) -> Dict[str, Any]:
        n_muon = sum(p.numel() for g in self.muon.param_groups for p in g["params"])
        n_adamw = sum(p.numel() for g in self.adamw.param_groups for p in g["params"])
        return {
            "muon_params": int(n_muon),
            "adamw_params": int(n_adamw),
            "muon_groups": len(self.muon.param_groups),
            "adamw_groups": len(self.adamw.param_groups),
            "optimizer_state_ratio": round(n_adamw / max(n_muon + n_adamw, 1), 4),
        }


def build_hybrid_optimizer(model: nn.Module, cfg: OptimConfig) -> HybridOptimizer:
    """按 OptimConfig 构建混合优化器。"""
    muon_p, adamw_p = split_params(model)

    muon = Muon(
        muon_p,
        lr=cfg.lr_muon,
        momentum=cfg.muon_momentum,
        nesterov=True,
        ns_steps=cfg.ns_steps,
        backend="gram" if cfg.use_gram_ns else "original",
        gram_reset_every=cfg.gram_reset_interval if cfg.use_gram_ns else 0,
        weight_decay=cfg.weight_decay,
        eps=cfg.ns_eps,
    )
    # AdamW 直接用 PyTorch 官方实现——零重写原则：不重造已被打磨到极致的东西
    adamw = torch.optim.AdamW(
        adamw_p,
        lr=cfg.lr_adamw,
        betas=cfg.betas,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )
    return HybridOptimizer(muon, adamw)
