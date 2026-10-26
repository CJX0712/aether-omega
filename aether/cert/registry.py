"""
证书注册表 —— AETHER-Ω 的"自证"中枢
=====================================

设计哲学
--------
普通项目的测试回答的是"代码有没有崩"；
证书（Certificate）回答的是更强的命题：**"这个计算在数学上是否成立"**。

区别在于，每条证书都必须明确写出三件事：
    1. STATEMENT —— 被验证的**数学命题**（不是"跑一下看看"，是"应当成立什么"）
    2. VALUE     —— 实测残差
    3. TOLERANCE —— 允许的偏差上界，以及**这个上界从哪来**（浮点误差分析，不是拍脑袋）

于是"通过"不再是"没报错"，而是：
        实测残差 ≤ 由浮点分析推导出的理论误差上界

证书分类
--------
    GRAD   梯度证书   解析梯度 vs 中心差分            （反向传播写错了能被抓住）
    EQUIV  等价证书   加速实现 vs 朴素参考实现        （"数学等价"不再只是 README 里的一句话）
    INVAR  不变量证书 代数性质（正交/相对性/守恒/归一）
    DETERM 确定性证书 位级可复现
    BOUND  误差界证书 前向/反向的浮点误差上界
    MONO   单调证书   训练动态应满足的序关系
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

__all__ = ["Certificate", "CertRegistry", "certificate", "check"]


@dataclass
class Certificate:
    """一条证书的完整记录。"""

    id: str                  # 稳定编号，如 "EQUIV-01"
    name: str                # 中文名
    category: str            # GRAD / EQUIV / INVAR / DETERM / BOUND / MONO
    statement: str           # 被验证的数学命题（人话 + 公式）
    passed: bool             # 是否通过
    value: Optional[float] = None       # 实测残差
    tolerance: Optional[float] = None   # 允许上界
    relation: str = "<="               # 比较关系，默认 value <= tolerance
    detail: str = ""                   # 补充说明
    reference: str = ""                # 依据（论文/定理）
    elapsed_ms: float = 0.0

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"

    def ratio(self) -> Optional[float]:
        """残差 / 上界。<1 表示有余量，越接近 1 说明越紧。"""
        if self.value is None or self.tolerance in (None, 0):
            return None
        return self.value / self.tolerance

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status
        d["ratio"] = self.ratio()
        return d


@dataclass
class CertRegistry:
    """证书集合 + 汇总输出。"""

    name: str = "AETHER-Ω Certificate Bundle"
    certs: List[Certificate] = field(default_factory=list)

    def add(self, c: Certificate) -> Certificate:
        self.certs.append(c)
        return c

    # ---- 汇总 ----
    @property
    def n(self) -> int:
        return len(self.certs)

    @property
    def n_pass(self) -> int:
        return sum(1 for c in self.certs if c.passed)

    @property
    def n_fail(self) -> int:
        return sum(1 for c in self.certs if not c.passed)

    def by_category(self) -> Dict[str, List[Certificate]]:
        out: Dict[str, List[Certificate]] = {}
        for c in self.certs:
            out.setdefault(c.category, []).append(c)
        return out

    def summary(self) -> Dict[str, Any]:
        cats = self.by_category()
        return {
            "name": self.name,
            "total": self.n,
            "passed": self.n_pass,
            "failed": self.n_fail,
            "all_green": self.n_fail == 0 and self.n > 0,
            "by_category": {k: {"n": len(v), "pass": sum(1 for c in v if c.passed)} for k, v in cats.items()},
        }

    # ---- 输出 ----
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            {"summary": self.summary(), "certificates": [c.to_dict() for c in self.certs]},
            ensure_ascii=False, indent=indent,
        )

    def to_markdown(self) -> str:
        s = self.summary()
        lines = [
            f"# 证书报告 · {s['name']}",
            "",
            f"**总计 {s['total']} 条 · 通过 {s['passed']} · 失败 {s['failed']} · "
            f"{'✅ 全绿' if s['all_green'] else '❌ 存在失败'}**",
            "",
            "| 编号 | 类别 | 命题 | 实测残差 | 允许上界 | 余量 | 状态 |",
            "|---|---|---|---|---|---|---|",
        ]
        for c in self.certs:
            v = "—" if c.value is None else f"{c.value:.3e}"
            t = "—" if c.tolerance is None else f"{c.tolerance:.3e}"
            r = c.ratio()
            rr = "—" if r is None else f"{r:.2f}×"
            lines.append(
                f"| `{c.id}` | {c.category} | {c.statement} | {v} | {t} | {rr} | {'✅' if c.passed else '❌'} |"
            )
        lines.append("")
        lines.append("## 命题详情")
        for c in self.certs:
            lines.append(f"### `{c.id}` · {c.name}")
            lines.append(f"- **命题**：{c.statement}")
            lines.append(f"- **判据**：value {c.relation} tolerance")
            if c.detail:
                lines.append(f"- **说明**：{c.detail}")
            if c.reference:
                lines.append(f"- **依据**：{c.reference}")
            lines.append(f"- **结果**：{c.status}")
            lines.append("")
        return "\n".join(lines)


def certificate(
    registry: CertRegistry,
    *,
    id: str,
    name: str,
    category: str,
    statement: str,
    fn: Callable[[], Dict[str, Any]],
) -> Certificate:
    """执行一个证书函数并登记结果。

    fn 需返回 dict，可含：value / tolerance / relation / detail / reference / passed。
    若未显式给 passed，则按 `value <= tolerance` 判定。
    任何异常都会被捕获并登记为 FAIL —— 证书运行失败本身就是一个必须暴露的事实。
    """
    t0 = time.perf_counter()
    try:
        res = fn() or {}
    except Exception as e:  # noqa: BLE001 - 证书必须永不崩溃，失败要被记录而非抛出
        res = {"passed": False, "detail": f"证书执行异常：{type(e).__name__}: {e}"}

    value = res.get("value")
    tol = res.get("tolerance")
    passed = res.get("passed")
    if passed is None:
        passed = (value is not None and tol is not None and value <= tol) or (
            value is None and tol is None and res.get("detail") is None
        )
        if passed is None:
            passed = False

    c = Certificate(
        id=id, name=name, category=category, statement=statement,
        passed=bool(passed),
        value=None if value is None else float(value),
        tolerance=None if tol is None else float(tol),
        relation=str(res.get("relation", "<=")),
        detail=str(res.get("detail", "")),
        reference=str(res.get("reference", "")),
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )
    return registry.add(c)


def check(value: float, tolerance: float, **kw: Any) -> Dict[str, Any]:
    """构造标准判定结果的小工具。"""
    out = {"value": float(value), "tolerance": float(tolerance)}
    out.update(kw)
    return out
