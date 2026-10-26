"""
MLA —— Multi-head Latent Attention（多头潜在注意力）
====================================================

来源：DeepSeek-V2 (arXiv:2405.04434)，被 DeepSeek-V3/V4、Kimi K2/K3、GLM-5 沿用。

为什么 MLA 是 2026 年 KV cache 问题的正解
----------------------------------------
GQA 的思路是"多个 query 头共用一组 KV 头"，把 KV cache 从 n_h 降到 n_kv；
但它仍要为**每个 KV 头**存完整的 d_h 维键与值。

MLA 走得更远：它压根不缓存展开后的 K/V，而是缓存一个**低秩潜在向量** c^KV ∈ R^{d_c}
（d_c 通常只有 d_h 的 1/4 ~ 1/8），加上一小段解耦的 RoPE 键 k^R ∈ R^{d_h^R}：

    c_t^{KV} = W^{DKV} h_t                    缓存它
    k_t^{R}  = RoPE(W^{KR} h_t)               缓存它（所有头共享一份）
    k_t^{C}  = W^{UK} c_t^{KV}                推理时不必真正展开
    v_t      = W^{UV} c_t^{KV}                推理时不必真正展开

于是每个 token 的缓存量是 (d_c + d_h^R)，**与头数 n_h 完全无关**。
DeepSeek-V3: 每 token 约 68.6 KiB；同规模 MHA 会高一个量级。头越多、上下文越长，省得越狠。

解耦 RoPE（Decoupled RoPE）
--------------------------
MLA 有个陷阱：如果直接把 RoPE 加到 c^KV 上，那么"吸收"技巧就失效了——
因为 RoPE 会在 c^KV 与 W^{UK} 之间插入一个位置相关的旋转矩阵，不可交换：

    RoPE_m(W^{UK} c) ≠ W^{UK} RoPE_m(c)

DeepSeek 的解法是把位置信息**剥离出去**：只让一小段独立的 k^R 携带位置，
主体 k^C 保持位置无关，从而吸收技巧继续成立。这就是"解耦"的含义。

两条数学等价的计算路径（本文件的核心）
--------------------------------------
【naive】显式展开 K/V 再做多头注意力——直观、易读、可当参考实现：
    k = [W^{UK} c^{KV} ; k^R]      v = W^{UV} c^{KV}
    o = softmax(q kᵀ / √d + mask) v

【absorb】把 W^{UK} 吸收进 q、把 W^{UV} 吸收进输出——推理期高效形式：
    q̂ = q^C W^{UK}                              ∈ R^{d_c}
    logit = q̂ · (c^{KV})ᵀ + q^R · (k^R)ᵀ       ← 注意这里已经没有 head 维
    ctx   = softmax(logit) @ c^{KV}             ∈ R^{d_c}
    o     = ctx W^{UVᵀ}                         ∈ R^{n_h·d_v}

严格推导（这就是 EQUIV 证书要机器检查的那条恒等式）：
    q·kᵀ = q^C·(W^{UK}c)^ᵀ + q^R·(k^R)^ᵀ
         = (q^C W^{UK})·c^ᵀ + q^R·(k^R)^ᵀ          ← 结合律，W^{UK} 左移到 q 侧
    Σ_t a_t v_t = Σ_t a_t W^{UV} c_t = W^{UV} (Σ_t a_t c_t)  ← 线性，W^{UV} 外移

两条路径在数学上**逐位相等**（差仅来自浮点结合律）。

⚠️ 一个常见坑：W^{UK} 与 W^{UV} 的形状
    吸收时 q̂ 的维度必须是 d_c（KV 潜在维度），不是 n_h·d_h。
    写反了模型不会报错，只会静默学得慢——这正是"必须有等价性证书"的理由。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from aether.modules.norm import QKNorm, RMSNorm
from aether.modules.rope import apply_rope, build_rope_cache

__all__ = ["MultiheadLatentAttention", "KVCache"]


class KVCache:
    """MLA 专用的 KV cache。

    只存两样东西：低秩潜在 c^{KV} 与解耦 RoPE 键 k^R。
    相比存 (K, V) 的标准 cache，内存是 (d_c + d_h^R) : (2·n_kv·d_h)。
    """

    __slots__ = ("max_len", "c_kv", "k_rope", "length")

    def __init__(
        self,
        batch_size: int,
        max_len: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        self.max_len = int(max_len)
        self.c_kv = torch.zeros(batch_size, max_len, kv_lora_rank, dtype=dtype, device=device)
        self.k_rope = torch.zeros(
            batch_size, max_len, qk_rope_head_dim, dtype=dtype, device=device
        )
        self.length = 0

    @torch.no_grad()
    def update(self, c_kv: torch.Tensor, k_rope: torch.Tensor) -> None:
        """追加写入。c_kv:[B,L,d_c]，k_rope:[B,L,d_h^R]。"""
        L = c_kv.shape[1]
        if self.length + L > self.max_len:
            raise RuntimeError(
                f"KV cache 溢出：已用 {self.length} + 新增 {L} > 容量 {self.max_len}"
            )
        s = self.length
        self.c_kv[:, s : s + L] = c_kv
        self.k_rope[:, s : s + L] = k_rope
        self.length = s + L

    @torch.no_grad()
    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.c_kv[:, : self.length], self.k_rope[:, : self.length]


class MultiheadLatentAttention(nn.Module):
    """多头潜在注意力。

    Args:
        cfg: ModelConfig。
        layer_idx: 层号（用于决定局部/全局）。
        mode: "naive"（参考实现，展开 K/V）或 "absorb"（推理高效，吸收矩阵）。
              两者数学等价，由 cert.equivalence 机器验证。
    """

    def __init__(self, cfg, layer_idx: int = 0, mode: str = "absorb") -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.mode = mode

        d = cfg.dim
        nh = cfg.n_heads
        self.n_heads = nh
        self.d_c = cfg.kv_lora_rank          # KV 潜在维度
        self.d_cq = cfg.q_lora_rank          # Q 潜在维度
        self.d_nope = cfg.qk_nope_head_dim   # 非位置部分头维
        self.d_rope = cfg.qk_rope_head_dim   # 位置部分头维
        self.d_v = cfg.v_head_dim
        self.scale = (self.d_nope + self.d_rope) ** -0.5

        # ---- 下投影（压缩）----
        self.w_dq = nn.Linear(d, self.d_cq, bias=False)     # h → c^Q
        self.w_dkv = nn.Linear(d, self.d_c, bias=False)     # h → c^KV

        # ---- Q 上投影 ----
        self.w_uq = nn.Linear(self.d_cq, nh * self.d_nope, bias=False)   # c^Q → q^C
        self.w_qr = nn.Linear(self.d_cq, nh * self.d_rope, bias=False)   # c^Q → q^R(待 RoPE)

        # ---- KV 上投影 ----
        self.w_uk = nn.Linear(self.d_c, nh * self.d_nope, bias=False)    # c^KV → k^C
        self.w_uv = nn.Linear(self.d_c, nh * self.d_v, bias=False)       # c^KV → v
        self.w_kr = nn.Linear(d, self.d_rope, bias=False)                # h → k^R(待 RoPE)

        self.w_o = nn.Linear(nh * self.d_v, d, bias=False)

        # ---- QK-Norm（Qwen3 / Gemma3 / OLMo2 稳定性标配）----
        #
        # ⚠️ 一个必须诚实交代的工程冲突：
        # 标准 QK-Norm 作用在**展开后**的 k^C = W^{UK} c^{KV} 上，
        # 但 RMSNorm 是非线性的（除以依赖数据的 RMS），无法吸收进 q 侧——
        # 一旦这样做，"naive 与 absorb 数学等价"这条性质就被破坏了，
        # 而这条性质正是 MLA 推理加速（也是本仓库的等价性证书）的根基。
        #
        # 解法：把 K 侧的归一化下移到**潜在空间**，分别对 c^{KV} 与解耦键 k^R 做 RMSNorm。
        # 数学上依然是对 key 范数的硬约束（‖k‖ 由 ‖c^{KV}‖ 与 ‖k^R‖ 共同控制），
        # 但由于归一化作用在**缓存的那份向量**上，吸收技巧继续严格成立。
        # 这是 MLA + QK-Norm 组合下唯一能同时保住"稳定"与"可吸收"的写法。
        self.use_qk_norm = bool(cfg.use_qk_norm)
        if self.use_qk_norm:
            self.q_norm = QKNorm(self.d_nope)
            self.kv_latent_norm = RMSNorm(self.d_c, cfg.rms_norm_eps)
            self.k_rope_norm = RMSNorm(self.d_rope, cfg.rms_norm_eps)

        # RoPE 表（按配置预生成，非 persistent buffer 以免污染 state_dict 指纹）
        cos, sin = build_rope_cache(
            cfg.max_seq_len, self.d_rope, cfg.rope_theta, cfg.yarn_scale,
            cfg.yarn_original_max_pos,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    # ------------------------------------------------------------------
    def _project(self, x: torch.Tensor, start_pos: int = 0):
        B, L, _ = x.shape
        nh = self.n_heads

        # ── RoPE 必须使用**绝对位置**，不是"本段内的相对位置" ──
        # 增量解码时每次只喂 1 个 token，若仍取 cos[:L]，则每个新 token 都被当成位置 0，
        # 整段生成的位置编码全部塌缩。这是 KV cache 实现里最经典的静默错误：
        # 模型照样能说话，只是长程位置关系全错。由 EQUIV-03 证书抓出。
        if start_pos + L > self.cos.shape[0]:
            raise RuntimeError(
                f"位置 {start_pos + L} 超出 RoPE 表容量 {self.cos.shape[0]}（cfg.max_seq_len）。"
                "请增大 max_seq_len 或减少生成长度——静默截断位置编码是绝对不能接受的。"
            )
        cos = self.cos[start_pos : start_pos + L]
        sin = self.sin[start_pos : start_pos + L]

        c_q = self.w_dq(x)                                    # [B,L,d_cq]
        c_kv = self.w_dkv(x)                                  # [B,L,d_c]

        q_c = self.w_uq(c_q).view(B, L, nh, self.d_nope)      # [B,L,H,d_nope]
        q_r = self.w_qr(c_q).view(B, L, nh, self.d_rope)      # [B,L,H,d_rope]
        k_r = self.w_kr(x)                                    # [B,L,d_rope] 所有头共享

        # RoPE 只作用于位置部分（解耦的关键）
        q_r = apply_rope(q_r, cos, sin)                       # [B,L,H,d_rope]
        k_r = apply_rope(k_r, cos, sin)                         # [B,L,d_rope]

        if self.use_qk_norm:
            q_c = self.q_norm(q_c)
            c_kv = self.kv_latent_norm(c_kv)   # 潜在空间归一化 → 保持可吸收
            k_r = self.k_rope_norm(k_r)
        return c_q, c_kv, q_c, q_r, k_r

    # ------------------------------------------------------------------
    def _naive(self, x, kv_cache=None, window=None, start_pos=0):
        """参考实现：显式展开 K/V。慢但直白，作为等价性证书的黄金参照。"""
        B, L, _ = x.shape
        nh = self.n_heads
        _, c_kv, q_c, q_r, k_r = self._project(x, start_pos)

        # 归一化已在 _project 中作用于潜在空间，此处直接展开即可
        k_c = self.w_uk(c_kv).view(B, L, nh, self.d_nope)     # [B,L,H,d_nope]
        v = self.w_uv(c_kv).view(B, L, nh, self.d_v)          # [B,L,H,d_v]

        if kv_cache is not None:
            kv_cache.update(c_kv.detach(), k_r.detach())
            c_all, kr_all = kv_cache.get()
            k_c = self.w_uk(c_all).view(B, -1, nh, self.d_nope)
            v = self.w_uv(c_all).view(B, -1, nh, self.d_v)
            k_r_full = kr_all
            S = c_all.shape[1]
        else:
            k_r_full = k_r
            S = L

        # k^R 广播到每个头后与 k^C 拼接 → [B,H,S,d_nope+d_rope]
        k_r_exp = k_r_full.unsqueeze(2).expand(B, S, nh, self.d_rope).transpose(1, 2)
        k = torch.cat([k_c.transpose(1, 2), k_r_exp], dim=-1)
        q = torch.cat([q_c.transpose(1, 2), q_r.transpose(1, 2)], dim=-1)  # [B,H,L,dqk]

        logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale         # [B,H,L,S]
        logits = self._mask(logits, L, S, window, kv_cache is not None)
        attn = torch.softmax(logits, dim=-1)
        out = torch.matmul(attn, v.transpose(1, 2))                        # [B,H,L,d_v]
        out = out.transpose(1, 2).reshape(B, L, nh * self.d_v)
        return self.w_o(out)

    # ------------------------------------------------------------------
    def _absorb(self, x, kv_cache=None, window=None, start_pos=0):
        """高效实现：吸收 W^{UK}/W^{UV}，KV cache 只存潜在向量。"""
        B, L, _ = x.shape
        nh = self.n_heads
        _, c_kv, q_c, q_r, k_r = self._project(x, start_pos)

        # 关键一步：把 W^{UK} 左移到 q 侧 → q̂ ∈ R^{d_c}（head 维消失）
        #
        # 性能说明：这里刻意用 matmul 的**广播**而不是 einsum。
        # 4D einsum 在 CPU 后端会插入 permute + reshape + 拷贝，实测比 matmul 慢数倍；
        # 而 matmul 对 [B,H,L,d] @ [1,H,d,d'] 的 batch 维广播是零拷贝的。
        # 数学上完全等价，只是调用的是更快的 BLAS 路径。
        w_uk = self.w_uk.weight.view(nh, self.d_nope, self.d_c)           # [H,d_nope,d_c]
        q_hat = torch.matmul(q_c.transpose(1, 2), w_uk.unsqueeze(0))      # [B,H,L,d_c]

        if kv_cache is not None:
            kv_cache.update(c_kv.detach(), k_r.detach())
            c_all, kr_all = kv_cache.get()
        else:
            c_all, kr_all = c_kv, k_r
        S = c_all.shape[1]

        # 拆成两段点积和：非位置段在潜在空间算，位置段单独算
        # [B,H,L,d] @ [B,1,d,S] → [B,H,L,S]（广播 batch 头维，零拷贝）
        kr_t = kr_all.unsqueeze(1).transpose(-1, -2)                      # [B,1,d_rope,S]
        logits = (
            torch.matmul(q_hat, c_all.unsqueeze(1).transpose(-1, -2))
            + torch.matmul(q_r.transpose(1, 2), kr_t)
        ) * self.scale

        logits = self._mask(logits, L, S, window, kv_cache is not None)
        attn = torch.softmax(logits, dim=-1)

        # 关键第二步：先对 c^{KV} 加权求和，再一次性乘 W^{UV}
        ctx = torch.matmul(attn, c_all.unsqueeze(1))                      # [B,H,L,d_c]
        w_uv = self.w_uv.weight.view(nh, self.d_v, self.d_c)              # [H,d_v,d_c]
        out = torch.matmul(ctx, w_uv.transpose(-1, -2).unsqueeze(0))      # [B,H,L,d_v]
        out = out.transpose(1, 2).reshape(B, L, nh * self.d_v)
        return self.w_o(out)

    # ------------------------------------------------------------------
    def _mask(self, logits, L, S, window, is_cached):
        """因果掩码 + 可选滑动窗口。返回被屏蔽位置为 -inf 的 logits。"""
        dev = logits.device
        if not is_cached:
            q_pos = torch.arange(L, device=dev)[:, None]
            k_pos = torch.arange(S, device=dev)[None, :]
            causal = k_pos <= q_pos
            if window is not None and window > 0:
                causal = causal & (k_pos > q_pos - window)
        else:
            # 增量解码：query 只有最后 L 个位置，其绝对位置为 S-L ... S-1
            q_pos = torch.arange(S - L, S, device=dev)[:, None]
            k_pos = torch.arange(S, device=dev)[None, :]
            causal = k_pos <= q_pos
            if window is not None and window > 0:
                causal = causal & (k_pos > q_pos - window)
        return logits.masked_fill(~causal, float("-inf"))

    # ------------------------------------------------------------------
    def forward(self, x, kv_cache=None, window=None, mode=None, start_pos=0):
        mode = mode or self.mode
        if mode not in ("naive", "absorb"):
            raise ValueError(f"mode 必须是 'naive' 或 'absorb'，收到 {mode!r}")
        # 有 KV cache 时，绝对位置由 cache 已写入的长度唯一确定——
        # 这比让调用方手动传 start_pos 更不容易出错。
        sp = kv_cache.length if kv_cache is not None else int(start_pos)
        fn = self._naive if mode == "naive" else self._absorb
        return fn(x, kv_cache, window, sp)

    @torch.no_grad()
    def max_abs_logit(self, x: torch.Tensor) -> float:
        """当前 batch 上注意力 logit 的最大绝对值。

        MuonClip 用它做 QK-clip 的触发信号：logit 一旦越过阈值 τ，
        就对产生它的 q/k 投影矩阵做一次缩放，防止训练中后期 logit 爆炸。
        """
        B, L, _ = x.shape
        nh = self.n_heads
        _, c_kv, q_c, q_r, k_r = self._project(x)
        w_uk = self.w_uk.weight.view(nh, self.d_nope, self.d_c)
        q_hat = torch.matmul(q_c.transpose(1, 2), w_uk.unsqueeze(0))
        logits = (
            torch.matmul(q_hat, c_kv.unsqueeze(1).transpose(-1, -2))
            + torch.matmul(q_r.transpose(1, 2), k_r.unsqueeze(1).transpose(-1, -2))
        ) * self.scale
        return float(logits.abs().max().item())
