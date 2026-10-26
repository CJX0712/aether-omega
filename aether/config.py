"""
AETHER-Ω 模型配置
=================

全部超参都对齐 2026 年已公开的旗舰开源模型共识配方，每一处都能追溯到具体模型：

    MLA 低秩 KV 压缩             → DeepSeek-V2/V3（KV cache 降一个量级）
    局部:全局 = 3:1 混合注意力    → Gemma 3 / Qwen 3 / Mistral Small 3.1
    DeepSeekMoE（细粒度+共享专家）→ DeepSeek-V3 / Kimi K2 / GLM-5
    RMSNorm（无 LayerNorm）      → 2026 全系，LayerNorm 已彻底退场
    SwiGLU FFN                   → 2026 全系
    RoPE + YaRN 长上下文外推      → 2026 全系（无人再用可学习位置）
    QK-Norm                      → Qwen 3 / Gemma 3 / OLMo 2（训练稳定性）
    MTP 多 token 预测            → DeepSeek-V3 / Qwen3 / GLM-4.5

预设档位按"能在什么机器上真跑起来"划分，而不是拍脑袋的数字：
    nano    ~15M  →  笔记本 CPU，几分钟出结果（默认，CI 与教学用）
    micro   ~60M  →  16 核 CPU / 单卡几小时
    base    ~350M →  单张 24G 卡
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict

__all__ = ["ModelConfig", "OptimConfig", "TrainConfig", "PRESETS", "get_preset"]


@dataclass
class ModelConfig:
    """架构配置。字段名与主流 HF config 对齐，便于互操作。"""

    # ---- 主干 ----
    vocab_size: int = 256
    dim: int = 256
    n_layers: int = 8
    max_seq_len: int = 512
    dropout: float = 0.0

    # ---- MLA 多头潜在注意力 ----
    # 约定（沿用 DeepSeek-V2 记号）：
    #   c_t^Q = W^{DQ} h_t          —— query 低秩压缩
    #   c_t^KV = W^{DKV} h_t        —— KV 低秩压缩（进 KV cache 的就是它）
    #   q_t^R = RoPE(W^{QR} c_t^Q)  —— 仅这一小段携带位置信息
    #   k_t^R = RoPE(W^{KR} h_t)    —— 解耦式 RoPE 键
    n_heads: int = 8
    q_lora_rank: int = 64
    kv_lora_rank: int = 32
    qk_nope_head_dim: int = 32  # 非位置部分
    qk_rope_head_dim: int = 16  # 位置（RoPE）部分
    v_head_dim: int = 32

    # ---- 局部/全局混合注意力（3:1）----
    # 每 4 层里 3 层滑动窗口（局部）、1 层全注意力（全局）
    local_window: int = 128
    global_every: int = 4

    # ---- DeepSeekMoE ----
    # 细粒度专家 + 共享专家 + 无辅助损失的偏置路由（DeepSeek-V3 方案）
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    moe_intermediate_size: int = 128
    first_moe_layer: int = 4  # 前几层用 dense FFN 打底（业界共识：底层放 dense）
    moe_layer_pattern: int = 1  # 1=全部 MoE，2=隔层 MoE

    # ---- Dense FFN ----
    ffn_hidden_size: int = 512

    # ---- 归一化 / 稳定化 ----
    rms_norm_eps: float = 1e-6
    use_qk_norm: bool = True
    qk_norm_clip: float = 64.0  # MuonClip 的 QK-clip 阈值 τ

    # ---- RoPE / YaRN ----
    rope_theta: float = 10000.0
    yarn_scale: float = 1.0  # >1 时启用 YaRN 外推
    yarn_original_max_pos: int = 512

    # ---- MTP 多 token 预测 ----
    mtp_depth: int = 1  # 预测未来几个 token（0=关闭）
    mtp_loss_weight: float = 0.1

    # ---- 推理 ----
    tie_embeddings: bool = True

    # ------------------------------------------------------------------
    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def kv_cache_dim(self) -> int:
        """每个 token 进 KV cache 的浮点数量。

        这是 MLA 相对 GQA/MHA 的杀手锏：cache 的是低秩潜在 c^KV(d_c 维) 加上
        解耦 RoPE 键 k^R(d_h^R 维)，与注意力头数 n_h 完全无关。
        GQA 需要 2·n_kv·d_h·L，MLA 只需 (d_c + d_h^R)·L —— 头数越多省得越狠。
        """
        return self.kv_lora_rank + self.qk_rope_head_dim

    def is_moe_layer(self, layer_idx: int) -> bool:
        if layer_idx < self.first_moe_layer:
            return False
        if self.moe_layer_pattern <= 1:
            return True
        return (layer_idx - self.first_moe_layer) % self.moe_layer_pattern == 0

    def is_global_layer(self, layer_idx: int) -> bool:
        return (layer_idx + 1) % self.global_every == 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, **kw: Any) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, **kw)


@dataclass
class OptimConfig:
    """优化器配置：Muon(2D 隐藏矩阵) + AdamW(其余) 混合路由。"""

    lr_muon: float = 0.02
    lr_adamw: float = 3e-4
    weight_decay: float = 0.01
    muon_momentum: float = 0.95
    betas: tuple = field(default_factory=lambda: (0.9, 0.95))
    eps: float = 1e-8

    # Gram Newton-Schulz（2026 最新：在 Gram 矩阵上正交化，数学等价但快 40–50%）
    ns_steps: int = 5
    use_gram_ns: bool = True
    gram_reset_interval: int = 32  # Gram 矩阵定期重算，抑制中精度累积漂移
    ns_eps: float = 1e-7

    # MuonClip：QK-clip 抑制注意力 logit 爆炸（Kimi K2 同款机制）
    use_muon_clip: bool = True
    qk_clip_threshold: float = 64.0

    # 学习率 schedule
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainConfig:
    """训练配置。默认按 CPU 友好设定，保证任何机器上都能真跑完。"""

    batch_size: int = 8
    grad_accum_steps: int = 2
    max_steps: int = 1200
    eval_every: int = 150
    eval_iters: int = 10
    log_every: int = 50
    seed: int = 1337
    # CPU 上线程数**不是越多越好**：本仓库实测（16 核 / 超线程）
    # 6 线程比 16 线程快约 1.8×——过多线程会在小矩阵上产生调度与缓存争用。
    # 可用 scripts/benchmark.py --scan-threads 在自己的机器上校准。
    threads: int = 6
    device: str = "cpu"
    dtype: str = "float32"
    checkpoint_dir: str = "checkpoints"
    sample_every: int = 300
    sample_tokens: int = 200

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


PRESETS: Dict[str, ModelConfig] = {
    "nano": ModelConfig(
        vocab_size=256,
        dim=256,
        n_layers=8,
        max_seq_len=256,
        n_heads=8,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
        local_window=128,
        global_every=4,
        n_routed_experts=8,
        n_shared_experts=1,
        n_activated_experts=2,
        moe_intermediate_size=128,
        first_moe_layer=4,
        moe_layer_pattern=1,
        ffn_hidden_size=512,
        mtp_depth=1,
    ),
    "micro": ModelConfig(
        vocab_size=256,
        dim=384,
        n_layers=12,
        max_seq_len=768,
        n_heads=8,
        q_lora_rank=96,
        kv_lora_rank=48,
        qk_nope_head_dim=40,
        qk_rope_head_dim=20,
        v_head_dim=40,
        local_window=192,
        global_every=4,
        n_routed_experts=12,
        n_shared_experts=1,
        n_activated_experts=3,
        moe_intermediate_size=192,
        first_moe_layer=4,
        moe_layer_pattern=1,
        ffn_hidden_size=768,
        mtp_depth=1,
    ),
    "base": ModelConfig(
        vocab_size=1024,
        dim=768,
        n_layers=20,
        max_seq_len=1024,
        n_heads=12,
        q_lora_rank=192,
        kv_lora_rank=96,
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        v_head_dim=64,
        local_window=256,
        global_every=4,
        n_routed_experts=16,
        n_shared_experts=1,
        n_activated_experts=4,
        moe_intermediate_size=384,
        first_moe_layer=2,
        moe_layer_pattern=1,
        ffn_hidden_size=1536,
        mtp_depth=1,
    ),
}


def get_preset(name: str = "nano", **overrides: Any) -> ModelConfig:
    """取预设并局部覆盖。未知档位直接报错，不静默回退——静默回退是实验复现的头号杀手。"""
    if name not in PRESETS:
        raise KeyError(f"未知档位 {name!r}，可选：{sorted(PRESETS)}")
    base = PRESETS[name].to_dict()
    unknown = set(overrides) - set(base)
    if unknown:
        raise KeyError(f"未知配置项：{sorted(unknown)}")
    base.update(overrides)
    return ModelConfig(**base)
