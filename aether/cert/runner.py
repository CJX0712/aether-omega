"""
证书总入口 —— 一条命令跑完全部自证
===================================

    from aether.cert.runner import run_all
    reg = run_all()
    print(reg.to_markdown())
"""

from __future__ import annotations

import sys
import time
from typing import Optional

from aether.cert.registry import CertRegistry
from aether.config import ModelConfig, get_preset

__all__ = ["run_all"]


def run_all(
    preset: str = "nano",
    cfg: Optional[ModelConfig] = None,
    *,
    verbose: bool = True,
    stream=sys.stdout,
) -> CertRegistry:
    """运行全部证书。"""
    from aether.cert.bounds import bound_certificates, monotonicity_certificates
    from aether.cert.determinism import determinism_certificates
    from aether.cert.equivalence import equivalence_certificates
    from aether.cert.gradcheck import grad_certificates
    from aether.cert.invariants import invariant_certificates

    cfg = cfg or get_preset(preset)
    reg = CertRegistry(name=f"AETHER-Ω Certificate Bundle ({preset})")

    stages = [
        ("GRAD   梯度正确性", lambda: grad_certificates(reg, cfg)),
        ("EQUIV  数学等价性", lambda: equivalence_certificates(reg, cfg)),
        ("INVAR  代数不变量", lambda: invariant_certificates(reg, cfg)),
        ("DETERM 位级确定性", lambda: determinism_certificates(reg, cfg)),
        ("BOUND  误差界 / MONO 训练动态", lambda: (bound_certificates(reg, cfg), monotonicity_certificates(reg, cfg))),
    ]

    for label, fn in stages:
        t0 = time.perf_counter()
        if verbose:
            print(f"  · 运行 {label} …", file=stream, flush=True)
        fn()
        if verbose:
            n = len(reg.certs)
            print(f"    完成，耗时 {time.perf_counter() - t0:.2f}s（累计 {n} 条）", file=stream, flush=True)
    return reg
