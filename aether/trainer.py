"""
训练器
======

包含 2026 旗舰训练栈的标准件，每一件都注明来源与取舍理由：

    · Warmup + Cosine 衰减        —— 前沿训练的默认 schedule
    · 梯度裁剪（全局范数）        —— 防梯度爆炸的最后一道闸
    · MuonClip（QK-clip）         —— Kimi K2 的 logit 上界机制，防中后期 spike
    · 混合优化器自动路由          —— 2D 隐藏矩阵走 Muon，其余走 AdamW
    · 训练期运行时证书采集        —— 不只在训完才验真，训练全程持续体检

为什么要在**训练期**采集证书
----------------------------
训完再验真只能证明"最后一刻的模型没坏"。真正有价值的信号是**全程**的：
    · 梯度范数是否稳定（爆炸/消失）
    · 注意力 logit 是否越过阈值（QK-clip 是否被触发、触发了几次）
    · MoE 负载是否失衡（是否出现"专家死亡"）
    · 每步 update 的正交化质量
这些曲线能让人一眼看出训练是健康的，还是"看起来 loss 在降但内部已经烂了"。
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, List, Optional

import torch

from aether.config import ModelConfig, OptimConfig, TrainConfig
from aether.determinism import fingerprint_module
from aether.optim.hybrid import HybridOptimizer, build_hybrid_optimizer
from aether.optim.muon import MuonClipController

__all__ = ["Trainer", "lr_schedule"]


def lr_schedule(step: int, total: int, warmup: int, min_ratio: float = 0.1) -> float:
    """Warmup（线性）→ Cosine 衰减到 min_ratio。返回 [min_ratio, 1] 的缩放因子。"""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    prog = min(max(prog, 0.0), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog))


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        cfg: ModelConfig,
        ocfg: OptimConfig,
        tcfg: TrainConfig,
        dataset,
        *,
        capture_every: int = 50,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.ocfg = ocfg
        self.tcfg = tcfg
        self.ds = dataset
        self.opt: HybridOptimizer = build_hybrid_optimizer(model, ocfg)
        self.clip = (
            MuonClipController(model, ocfg.qk_clip_threshold, every=50)
            if ocfg.use_muon_clip
            else None
        )
        self.capture_every = capture_every

        self.history: List[Dict[str, Any]] = []
        self.clip_events: List[Dict[str, Any]] = []
        self.best_val = float("inf")
        os.makedirs(tcfg.checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def _set_phase(self, capture: bool) -> None:
        for ly in self.model.layers:
            ly.capture = capture

    # ------------------------------------------------------------------
    @torch.no_grad()
    def grad_norm(self) -> float:
        """全局梯度范数 ‖∇‖₂ = sqrt(Σ_p ‖∇_p‖²)（跨所有参数）。

        必须在 optimizer.step() **之前**调用——step 之后紧接着 zero_grad，
        梯度就被清成 None 了，此时再采只會得到 0（一个看起来"很健康"的假信号）。
        """
        gsq = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                gsq += float(p.grad.detach().pow(2).sum())
        return math.sqrt(gsq)

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        tot, n = 0.0, 0
        for _ in range(self.tcfg.eval_iters):
            x, y = self.ds.get_batch("val", self.tcfg.batch_size)
            out = self.model(x, y)
            tot += float(out["loss"].detach())
            n += 1
        self.model.train()
        return {"val_loss": tot / max(n, 1)}

    # ------------------------------------------------------------------
    def _runtime_certificates(self, step: int, grad_norm: float) -> Dict[str, Any]:
        """轻量运行时体检（每 log_every 步一次，开销 < 1%）。

        Args:
            grad_norm: 由调用方在 step() 之前采集并传入（见 grad_norm() 的说明）。
        """
        rec: Dict[str, Any] = {}
        # 梯度范数：爆炸/消失的第一指标
        rec["grad_norm"] = grad_norm

        # MoE 负载均衡
        ratios, utils = [], []
        for ly in self.model.layers:
            if hasattr(ly.ffn, "load_balance_stats"):
                r, u = ly.ffn.load_balance_stats()
                ratios.append(r)
                utils.append(u)
        if ratios:
            rec["moe_max_load_ratio"] = max(ratios)
            rec["moe_expert_util"] = sum(utils) / len(utils)
        return rec

    # ------------------------------------------------------------------
    def _maybe_muon_clip(self, step: int) -> None:
        if self.clip is None:
            return
        if step % self.clip.every != 0:
            return
        smax: List[float] = []
        for ly in self.model.layers:
            xn = getattr(ly, "last_attn_input", None)
            if xn is None:
                return
            smax.append(ly.attn.max_abs_logit(xn))
        rec = self.clip.clip(smax)
        rec["step"] = step
        if rec["n_clipped"] > 0:
            self.clip_events.append(rec)

    # ------------------------------------------------------------------
    def run(self, *, verbose: bool = True) -> Dict[str, Any]:
        m, t = self.model, self.tcfg
        m.train()
        base_muon, base_adamw = self.ocfg.lr_muon, self.ocfg.lr_adamw
        t0 = time.perf_counter()
        running = 0.0
        loss_count = 0  # 当前窗口内已累加的 micro-batch 数（用于正确的平均 loss）

        for step in range(t.max_steps):
            capture = (step % self.capture_every == 0) or (
                self.clip is not None and step % self.clip.every == 0
            )
            self._set_phase(capture)

            scale = lr_schedule(step, t.max_steps, self.ocfg.warmup_steps, self.ocfg.min_lr_ratio)
            self.opt.set_lr_scale(scale, base_muon=base_muon, base_adamw=base_adamw)

            for _ in range(t.grad_accum_steps):
                x, y = self.ds.get_batch("train", t.batch_size)
                out = m(x, y)
                (out["loss"] / t.grad_accum_steps).backward()

            # 必须在 step() 之前采集：step 之后立刻 zero_grad，梯度归零。
            # 而且必须采集**裁剪前**的范数——clip 之后范数恒等于阈值，
            # 曲线会变成一条毫无信息量的水平线（这个坑在开发中真实踩过：
            # 日志里 |∇| 恒为 1.000，看起来"很稳定"，其实什么都没说明）。
            if self.ocfg.grad_clip > 0:
                # clip_grad_norm_ 返回的是**裁剪前**的总范数，正好是我们要的信号
                gnorm = float(torch.nn.utils.clip_grad_norm_(m.parameters(), self.ocfg.grad_clip))
            else:
                gnorm = self.grad_norm()
            self._maybe_muon_clip(step)
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            self._set_phase(False)

            running += float(out["loss"].detach())
            loss_count += 1
            if verbose and (step % t.log_every == 0 or step == t.max_steps - 1):
                rec = self._runtime_certificates(step, gnorm)
                rec.update(
                    {
                        "step": step,
                        "train_loss": running / loss_count,
                        "lr_scale": scale,
                        "grad_clipped": gnorm > self.ocfg.grad_clip if self.ocfg.grad_clip > 0 else False,
                    }
                )
                self.history.append(rec)
                print(
                    f"  step {step:5d}/{t.max_steps}  loss {float(out['loss'].detach()):.4f}"
                    f"  |∇| {rec['grad_norm']:.3f}{'✂' if rec['grad_clipped'] else ' '}"
                    + (f"  MoE负载 {rec['moe_max_load_ratio']:.2f}" if "moe_max_load_ratio" in rec else ""),
                    flush=True,
                )
                running = 0.0
                loss_count = 0

            if step % t.eval_every == 0 or step == t.max_steps - 1:
                v = self.evaluate()
                self.best_val = min(self.best_val, v["val_loss"])
                if verbose:
                    print(
                        f"    ↳ val_loss {v['val_loss']:.4f}  "
                        f"ppl {math.exp(min(v['val_loss'], 20)):.2f}  (best {self.best_val:.4f})",
                        flush=True,
                    )
                self.history[-1].update(v)

        elapsed = time.perf_counter() - t0
        fp = fingerprint_module(m, digits=16)
        return {
            "elapsed_sec": round(elapsed, 2),
            "steps_per_sec": round(t.max_steps / max(elapsed, 1e-9), 3),
            "final_val_loss": self.best_val,
            "final_perplexity": math.exp(min(self.best_val, 20)),
            "weight_fingerprint": fp,
            "n_clip_events": len(self.clip_events),
            "history": self.history,
        }

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "opt": self.opt.state_dict(),
                "cfg": self.cfg.to_dict(),
                "ocfg": self.ocfg.to_dict(),
                "tcfg": self.tcfg.to_dict(),
            },
            path,
        )

    def save_report(self, path: str, result: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
