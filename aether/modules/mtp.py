"""
MTP —— Multi-Token Prediction（多 token 预测）
==============================================

来源：Gloeckle et al. (arXiv:2404.19737)，DeepSeek-V3 首次在旗舰规模上验证，
随后 Qwen3 / GLM-4.5 / GLM-5 跟进。

原理
----
标准语言模型在每个位置只预测下一个 token，监督信号是"稠密但短视"的。
MTP 额外挂若干个轻量预测头，让第 t 个位置的表示去预测 t+1+d：

    主头：   h_t  →  logits(t+1)
    MTP d：  h_t  →  logits(t+1+d)     d = 1..D

MTP 的价值（两点，第二点常被忽略）：
1. **训练信号增密**：一个位置同时被 D+1 个目标监督，样本效率提升。
2. **表示前置**：迫使 h_t 提前编码"未来 d 步还需要的信息"，
   这被认为是 MTP 能改善长程推理与猜测解码接受率的根因。

关键实现约束（DeepSeek-V3 的做法，本文件严格遵守）
--------------------------------------------------
· MTP 头共享主模型的 embedding 与 output head —— 不引入独立词表，零额外参数量级。
· 每个 MTP 头先做一次**跨步融合**：
      h'_t = RMSNorm(h_t) 与 RMSNorm(emb(y_{t+d-1})) 拼接后线性投影回 dim
  即把"上一步的真实 token"喂进来做 teacher forcing。
· **因果链**：MTP_d 的输入依赖 MTP_{d-1} 的表示（顺序执行），不是各自独立。

推理期的用途
------------
MTP 头天然就是**自投机解码（self-speculative decoding）的草稿模型**：
先用 MTP 头快速猜 D 个 token，再用主模型一次前向并行验证。
本文件提供 draft() 接口，scripts/benchmark.py 中给出接受率与加速比。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from aether.modules.norm import RMSNorm

__all__ = ["MTPHead", "MTPStack"]


class MTPHead(nn.Module):
    """单个 MTP 预测头。

    Args:
        dim: 模型维度。
        shared_embed: 主模型的 token embedding（共享，不新增参数）。
        shared_out: 主模型的输出头（共享）。
    """

    def __init__(self, dim: int, shared_embed: nn.Module, shared_out: nn.Module, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.embed = shared_embed
        self.out = shared_out
        self.norm_h = RMSNorm(dim, eps)
        self.norm_e = RMSNorm(dim, eps)
        # 融合：把 [归一化后的隐状态 ; 归一化后的上一步 token 嵌入] 投影回 dim
        self.proj = nn.Linear(2 * dim, dim, bias=False)

    def forward(
        self,
        h: torch.Tensor,
        prev_hidden: torch.Tensor,
        prev_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """h: 主干隐状态 [B,L,D]；prev_hidden: 前一个 MTP 头（或主干）的表示；
        prev_tokens: 目标位移 d-1 处的 token id [B,L]。"""
        e = self.embed(prev_tokens)
        fused = self.proj(torch.cat([self.norm_h(prev_hidden), self.norm_e(e)], dim=-1))
        return self.out(fused)


class MTPStack(nn.Module):
    """顺序 MTP 头栈。depth=0 时该模块为空操作（配置里关掉 MTP 即可）。"""

    def __init__(self, cfg, shared_embed: nn.Module, shared_out: nn.Module) -> None:
        super().__init__()
        self.depth = int(cfg.mtp_depth)
        self.heads = nn.ModuleList(
            [MTPHead(cfg.dim, shared_embed, shared_out, cfg.rms_norm_eps) for _ in range(self.depth)]
        )

    def forward(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """返回 [(logits_d, target_d)]，d = 1..depth。

        对齐关系（设序列长 L）：
            位置 t 用主干隐状态 h_t，teacher forcing 喂入真实 token y_{t+d}，
            预测目标是 y_{t+d+1}。
            为使下标合法需 t + d + 1 ≤ L−1 ⇒ 可用位置数 n = L − d − 1。
        """
        out: List[Tuple[torch.Tensor, torch.Tensor]] = []
        L = hidden.shape[1]
        prev = hidden  # d=1 时前一表示就是主干隐状态；之后链式替换为各 MTP 头的输出
        for d, head in enumerate(self.heads, start=1):
            n = L - d - 1
            if n <= 0:
                break
            h_cut = hidden[:, :n]
            prev_cut = prev[:, :n]
            prev_tokens = input_ids[:, d : d + n]        # y_{t+d}
            logits = head(h_cut, prev_cut, prev_tokens)  # → 预测 y_{t+d+1}
            target = input_ids[:, d + 1 : d + 1 + n]
            out.append((logits, target))
            # 因果链：本头的融合表示成为下一头的输入
            prev = head.proj(
                torch.cat([head.norm_h(prev_cut), head.norm_e(head.embed(prev_tokens))], dim=-1)
            )
        return out

    @torch.no_grad()
    def draft(self, hidden: torch.Tensor, last_token: torch.Tensor) -> List[torch.Tensor]:
        """推理期草稿：给主模型做自投机解码用。返回每个头预测的 token id。"""
        seq: List[torch.Tensor] = []
        prev = hidden
        tok = last_token
        for head in self.heads:
            logits = head(prev[:, -1:], prev[:, -1:], tok)
            nxt = logits.argmax(dim=-1)
            seq.append(nxt)
            tok = nxt
            prev = head.proj(
                torch.cat([head.norm_h(prev[:, -1:]), head.norm_e(head.embed(nxt))], dim=-1)
            )
        return seq
