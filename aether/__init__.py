"""
AETHER-Ω —— 自证式（Self-Certifying）旗舰 AI 基础栈
====================================================

作者：晨星 (Chenxing)   |   许可：MIT

核心命题
--------
2026 年前沿大模型的 backbone 已经高度收敛（MLA/MoE + RMSNorm + SwiGLU + QK-Norm + MTP），
架构层的差异化空间几乎耗尽；而真正的空白在另一处——**可验证性**：

    · 训练不可复现（同一份代码两次跑出不同权重）
    · 梯度不可校验（反向传播写错了没人知道）
    · 加速算子不可证明（"数学等价"只写在 README 里，没有机器检查）
    · 数值误差无界（没人能给出前向/反向的误差上界）

AETHER-Ω 的回答是：让**每一项计算都自带机器可验证的数学证书（Certificate）**。
不是"我们相信它对"，而是"跑一条命令，机器证明它对"。

三根支柱
--------
1. ARCHITECTURE —— 2026 收敛配方集大成
   MLA(DeepSeek V2/V3) · 局部/全局 3:1 混合注意力(Gemma 3 / Qwen 3)
   DeepSeekMoE(fine-grained + shared expert + 无辅助损失偏置路由)
   RMSNorm · SwiGLU · RoPE(YaRN) · QK-Norm · MTP(DeepSeek V3)

2. OPTIMIZATION —— 2026 优化器前沿
   GramMuon（Gram Newton-Schulz 正交化，数学等价、比原版 NS 快 40–50%）
   MuonClip（QK-clip 抑制注意力 logit 爆炸，Kimi K2 同款机制）
   Muon/AdamW 按参数结构自动路由（2D 隐藏层矩阵走 Muon，其余走 AdamW）

3. CERTIFICATION —— 差异化核心：自证式证书层
   GRAD（解析梯度 vs 中心差分）· EQUIV（加速算子 vs 朴素参考实现的数学等价）
   INVAR（正交性/低秩/守恒/单调等代数不变量）· DETERM（位级确定性复现）
   BOUND（前向/反向浮点误差上界）

快速开始
--------
    pip install -r requirements.txt
    python scripts/download_data.py
    python scripts/certify.py          # 一键全量验真（不训练，纯验真）
    python scripts/train_nano.py       # 真实训练 + 运行时证书采集
    python scripts/reproduce.py        # 位级确定性复现验证
    node  verify/aether-verify.js      # 零依赖验真（无需 Python/PyTorch）

版本
----
__version__ = "1.0.0"
"""

__version__ = "1.0.0"
__author__ = "晨星 (Chenxing)"

from aether.config import ModelConfig, PRESETS  # noqa: F401

__all__ = ["ModelConfig", "PRESETS", "__version__", "__author__"]
