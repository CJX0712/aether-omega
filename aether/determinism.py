"""
位级确定性（Bitwise Determinism）基座
=====================================

为什么这是第一等公民？
----------------------
科学的最低门槛是**可复现**。如果一个训练过程两次跑出不同的权重，
那么围绕它做的任何 ablation、任何结论、任何"提升了 3 个点"都不可信。
当前主流训练栈默认是非确定的：原子加、cuBLAS 归约顺序、多线程并行归约、
Dropout 采样顺序……任何一环漂移都让结果无法被第三方复现。

本模块把确定性变成**可开关、可检查、可证明**的一件事：
设置全局确定性 → 产出 bit-exact 的权重指纹 → 由 cert.determinism 交叉验证。

数学与工程依据
--------------
· torch.use_deterministic_algorithms(True)：强制走确定实现，非确定算子直接抛错而非静默漂移。
· CUBLAS_WORKSPACE_CONFIG=:4096:8 / :16:8：cuBLAS 在确定性模式下必须固定 workspace，
  否则 split-K 归约顺序不确定（NVIDIA 官方要求，见 PyTorch reproducibility 文档）。
· 固定 CPU 线程数：并行归约的分块顺序在固定线程数下稳定，避免 OMP 动态调度引入漂移。
· 固定 seed + 独立生成器：数据采样与模型初始化使用互不干扰的 RNG 流，
  保证"改了数据顺序不会悄悄改变初始化"。

用法
----
    from aether.determinism import enable_determinism, seed_all
    enable_determinism(seed=1337, threads=8)
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any, Dict, Iterable, List

import torch

__all__ = [
    "enable_determinism",
    "seed_all",
    "fingerprint_tensor",
    "fingerprint_module",
    "fingerprint_bytes",
    "determinism_report",
]


def seed_all(seed: int) -> None:
    """为 Python / NumPy(可选) / Torch(CPU+CUDA) 播下同一颗种子。"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - 无 GPU 环境下跳过
        torch.cuda.manual_seed_all(seed)
    try:  # numpy 为可选依赖，缺失不影响主链路
        import numpy as np

        np.random.seed(seed % (2**32))
    except Exception:  # pragma: no cover
        pass


def enable_determinism(
    seed: int = 1337,
    threads: int = 8,
    *,
    warn_only: bool = False,
    cudnn: bool = True,
) -> Dict[str, Any]:
    """打开全局确定性开关，并返回一份可审计的环境快照。

    Args:
        seed: 全局随机种子。
        threads: 固定 CPU 线程数（避免并行归约顺序漂移）。
        warn_only: True 时，遇到只能用非确定实现的算子仅告警；False 时直接抛错（推荐 CI 用）。
        cudnn: 是否强制 cuDNN 确定性（无 GPU 时无副作用）。

    Returns:
        环境快照 dict，供证书报告记录。
    """
    seed_all(seed)

    # cuBLAS 确定性 workspace：必须在 import torch 之后、首次 matmul 之前设置
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    if cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # benchmark 会挑选非确定算法

    torch.set_num_threads(int(threads))
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except Exception:  # 已在运行中调用过则忽略
            pass

    return {
        "seed": int(seed),
        "threads": int(threads),
        "warn_only": bool(warn_only),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "torch_version": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def fingerprint_bytes(data: bytes) -> str:
    """SHA-256 指纹。选 SHA-256 而非 CRC：CRC 只查意外损坏，SHA-256 抗碰撞，
    可作为"bit-exact 复现"的可公开比对凭据。"""
    return hashlib.sha256(data).hexdigest()


def fingerprint_tensor(t: torch.Tensor) -> str:
    """张量 bit-exact 指纹：取底层连续字节做 SHA-256。

    注意：先看 dtype/stride 再取字节，避免"数值相同但布局不同"被误判为一致。
    """
    if not t.is_contiguous():
        t = t.contiguous()
    header = f"{tuple(t.shape)}|{t.dtype}|{t.layout}".encode()
    body = t.detach().cpu().numpy().tobytes()
    return fingerprint_bytes(header + body)


def fingerprint_module(m: torch.nn.Module, *, digits: int = 8) -> str:
    """整个模型权重的 bit-exact 指纹（参数名有序，避免 dict 顺序漂移）。"""
    h = hashlib.sha256()
    for name, p in sorted(m.named_parameters()):
        h.update(name.encode())
        h.update(fingerprint_tensor(p.data).encode())
    for name, b in sorted(m.named_buffers()):
        h.update(b"buf:" + name.encode())
        h.update(fingerprint_tensor(b.data).encode())
    return h.hexdigest()[:digits]


def determinism_report(values: Iterable[float]) -> Dict[str, Any]:
    """对一组"重复运行的指纹/损失值"给出复现性判定。

    判定标准：全部取值完全相同 → bit-exact；否则给出最大漂移。
    """
    vals: List[float] = list(values)
    if not vals:
        return {"n": 0, "identical": False, "max_abs_drift": None}
    return {
        "n": len(vals),
        "identical": all(v == vals[0] for v in vals),
        "max_abs_drift": float(max(vals) - min(vals)),
        "first": vals[0],
    }
