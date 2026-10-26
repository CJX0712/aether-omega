"""
DeepSeekMoE —— 细粒度专家 + 共享专家 + 无辅助损失偏置路由
=========================================================

来源：DeepSeek-V2 (arXiv:2405.04434) / DeepSeek-V3 (arXiv:2412.19437, 无辅助损失版)。
同款设计见于 Kimi K2、GLM-5、Qwen3-MoE。

三个关键设计，每一条都在解决 MoE 的一个具体病症
------------------------------------------------
1. **细粒度专家切分（Fine-grained）**
   把 FFN 中间层切成更多更小的专家。同样的激活参数量下，专家数越多、
   路由的组合空间越大（从 C(n,k) 的意义上说）， specialization 更充分。

2. **共享专家（Shared Expert）**
   恒定激活 1~2 个专家，承载**所有 token 都必须经过的通用变换**。
   这样路由专家就不必浪费容量去重复学习"I am a token"这种公共知识，
   可以专心做差异化。Kimi K2/DeepSeek 均验证有效。

3. **无辅助损失的偏置路由（Auxiliary-loss-free）**  ← DeepSeek-V3 的标志性改动
   传统 MoE 用 auxiliary loss 强制负载均衡，但 aux loss 会**污染梯度**——
   它跟语言建模目标是竞争关系，α 调大了损害模型质量，调小了负载失衡。
   V3 的做法：给每个专家维护一个**可在线更新的偏置 b_i**，只用于 top-k 选择，
   不进梯度图：

        s_i    = sigmoid(w_gate(x)ᵀ)_i            路由分数（进梯度）
        ŝ_i    = s_i + b_i                        仅用于挑选（不进梯度）
        top-k  = argtop_k(ŝ)
        每步后：b_i ← b_i + u·sign(ē_i − e_i)     过载则降偏置，欠载则升偏置

   偏置的更新是**纯函数式**的（只依赖当前 batch 的负载统计），因此不破坏
   位级确定性——这一点对 AETHER-Ω 的自证式目标至关重要。

输出
----
    y = Σ_{i∈top-k} g_i · Expert_i(x)  +  SharedExpert(x)
    g_i = s_i / Σ_{j∈top-k} s_j        （V3 用 sigmoid 分数做归一化权重）

确定性注意
----------
· top-k 选择使用 stable sort，保证同分时索引顺序稳定（避免跨机漂移）。
· 专家计算按 expert_id 升序循环，float 累加顺序固定。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Expert", "DeepSeekMoE"]


class Expert(nn.Module):
    """单个 SwiGLU 专家。参数布局与 FFN 一致：gate/up 合并为一次 matmul 省一次 kernel 启动。"""

    def __init__(self, dim: int, inter: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w_gate_up = nn.Linear(dim, 2 * inter, bias=False)
        self.w_down = nn.Linear(inter, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gu = self.w_gate_up(x)
        g, u = gu.chunk(2, dim=-1)
        return self.w_down(self.dropout(F.silu(g) * u))


class DeepSeekMoE(nn.Module):
    """细粒度 MoE 层（含共享专家与偏置路由）。"""

    def __init__(self, cfg, dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = cfg.dim
        self.n_routed = int(cfg.n_routed_experts)
        self.n_shared = int(cfg.n_shared_experts)
        self.top_k = int(cfg.n_activated_experts)
        self.inter = int(cfg.moe_intermediate_size)
        self.bias_update_rate = 1e-3

        if self.top_k > self.n_routed:
            raise ValueError("n_activated_experts 不能大于 n_routed_experts")

        self.experts = nn.ModuleList(
            [Expert(self.dim, self.inter, dropout) for _ in range(self.n_routed)]
        )
        self.shared = (
            nn.ModuleList([Expert(self.dim, self.inter * 2, dropout) for _ in range(self.n_shared)])
            if self.n_shared > 0
            else None
        )
        self.gate = nn.Linear(self.dim, self.n_routed, bias=False)

        # 偏置：不参与梯度，仅用于 top-k 挑选；注册为 buffer 以便进 state_dict 指纹
        self.register_buffer("expert_bias", torch.zeros(self.n_routed))
        self.register_buffer("expert_load", torch.zeros(self.n_routed))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_bias(self, counts: torch.Tensor, n_tokens: int) -> None:
        """按负载更新偏置：过载（超过均值）→ 降偏置；欠载 → 升偏置。

        counts: 每个专家在本 batch 中被选中的次数 [E]
        """
        avg = n_tokens * self.top_k / self.n_routed
        err = avg - counts.float()  # >0 表示欠载，需要提高偏置
        self.expert_bias += self.bias_update_rate * torch.sign(err)
        self.expert_load.mul_(0.9).add_(0.1 * counts.float())

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        T = B * L
        xf = x.reshape(T, D)

        # ---- 路由 ----
        score = torch.sigmoid(self.gate(xf))                 # [T, E] 进梯度
        biased = score + self.expert_bias.unsqueeze(0)       # 仅用于挑选
        topv, topi = torch.topk(biased, self.top_k, dim=-1, sorted=True)  # [T, k]

        # 权重用**未加偏置**的分数做归一化，保证梯度不被偏置污染
        w = score.gather(1, topi)
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)

        # ---- 按专家分组计算（稀疏：只算被选中的）----
        out = torch.zeros_like(xf)
        flat_e = topi.reshape(-1)
        flat_t = torch.arange(T, device=x.device).unsqueeze(1).expand(T, self.top_k).reshape(-1)
        flat_w = w.reshape(-1)

        # stable sort：同分时索引顺序稳定 → 位级可复现
        order = torch.argsort(flat_e, stable=True)
        flat_e = flat_e[order]
        flat_t = flat_t[order]
        flat_w = flat_w[order]

        counts = torch.bincount(flat_e, minlength=self.n_routed)
        # cumsum 给出每个专家的连续区段（因已排序）
        bounds = torch.cumsum(counts, 0)
        starts = bounds - counts

        for e in range(self.n_routed):
            s, c = int(starts[e]), int(counts[e])
            if c == 0:
                continue
            idx_t = flat_t[s : s + c]
            xe = xf.index_select(0, idx_t)                   # [c, D]
            ye = self.experts[e](xe)                         # [c, D]
            ww = flat_w[s : s + c].unsqueeze(1)
            out.index_add_(0, idx_t, ye * ww)

        if self.shared is not None:
            for sh in self.shared:
                out = out + sh(xf)

        # ⚠️ 偏置只在训练期更新。
        # 推理（eval）时若继续更新，同一个 batch 第二次前向就会得到不同的路由结果，
        # 直接破坏两条性质：(1) KV cache 增量解码 ≡ 全序列重算（EQUIV-03）
        #                  (2) 因果性：未来 token 不影响过去（INVAR-06）
        # 这个 bug 正是由这两条证书抓出来的——它不会让程序崩溃，只会让结果静默漂移。
        if self.training:
            self._update_bias(counts, T)
        return out.view(B, L, D)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def load_balance_stats(self) -> Tuple[float, float]:
        """返回 (最大负载/平均负载, 专家利用率)。用于训练期的健康度证书。"""
        load = self.expert_load
        avg = load.mean().item()
        if avg <= 0:
            return 1.0, 0.0
        return float(load.max().item() / avg), float((load > avg * 0.1).float().mean().item())
