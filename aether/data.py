"""
数据管线
========

设计取舍：字节级（byte-level）UTF-8 分词
----------------------------------------
不用 BPE：BPE 需要额外的词表文件与合并规则，引入一条"必须联网下载某个特定文件"
的依赖，直接破坏**可复现**这个第一目标。字节级分词的好处是：

    · 词表固定 256，零外部依赖，任何机器跑出来完全一致
    · 中英文混排天然支持（UTF-8 多字节序列）
    · 与 GPT-2 的 byte-level BPE 在信息论上等价（都是无损编码）

代价是序列变长（同等文本 token 数更多），对 nano 档教学实验完全够用。

确定性采样
----------
数据顺序由独立的 torch.Generator 决定，与模型初始化的 RNG 流互不干扰——
这样"改数据种子不会悄悄改变模型初始化"，ablation 才有意义。
"""

from __future__ import annotations

import os
import sys
import urllib.request
from typing import Dict, List, Optional, Tuple

import torch

__all__ = ["TextDataset", "load_text", "download_tinyshakespeare", "encode", "decode"]

# 语料已随仓库发布（data/tinyshakespeare.txt，1.1MB，莎士比亚公共领域文本），
# 目的是让第三方 clone 后**无需联网**即可完整复现实验——网络是复现的头号敌人。
# 下面的多源列表只在用户显式要求重新拉取时才用到。
MIRRORS = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "https://cdn.jsdelivr.net/gh/karpathy/char-rnn@master/data/tinyshakespeare/input.txt",
    "https://gh-proxy.com/https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
)


def download_tinyshakespeare(path: str = "data/tinyshakespeare.txt", force: bool = False) -> str:
    """下载 Tiny Shakespeare（约 1.1 MB）。已存在且非 force 时直接返回。

    逐个尝试镜像源：部分网络环境下 raw.githubusercontent.com 会被重置连接
    （Windows 上表现为 WinError 10054），因此必须有多源兜底。
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not force and os.path.exists(path) and os.path.getsize(path) > 100_000:
        return path

    last_err: Optional[Exception] = None
    for url in MIRRORS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            if len(data) < 100_000:
                raise RuntimeError(f"下载内容异常（{len(data)} 字节）")
            with open(path, "wb") as f:
                f.write(data)
            return path
        except Exception as e:  # noqa: BLE001 - 逐个尝试，全部失败才抛出
            last_err = e
            print(f"  · 镜像不可用（{type(e).__name__}）：{url}", file=sys.stderr)
    raise RuntimeError(f"所有镜像均下载失败，最后一个错误：{last_err}")


def encode(text: str) -> List[int]:
    """UTF-8 字节级编码 → [0, 255]。"""
    return list(text.encode("utf-8", errors="surrogatepass"))


def decode(ids: List[int]) -> str:
    return bytes(ids).decode("utf-8", errors="replace")


def load_text(path: Optional[str] = None) -> str:
    if path is None:
        path = download_tinyshakespeare()
    if not os.path.exists(path):
        raise FileNotFoundError(f"语料不存在：{path}。请先运行 python scripts/download_data.py")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


class TextDataset:
    """把文本切成 train/val 两段，并提供确定性 batch 采样。"""

    def __init__(
        self,
        text: str,
        seq_len: int,
        *,
        train_ratio: float = 0.9,
        seed: int = 1337,
        device: str = "cpu",
    ) -> None:
        ids = encode(text)
        n = int(len(ids) * train_ratio)
        self.train = torch.tensor(ids[:n], dtype=torch.long, device=device)
        self.val = torch.tensor(ids[n:], dtype=torch.long, device=device)
        self.seq_len = int(seq_len)
        self.seed = int(seed)
        # 独立 RNG 流：数据采样的随机性与模型初始化解耦
        self._gen = torch.Generator(device="cpu").manual_seed(seed)

    @property
    def vocab_size(self) -> int:
        return 256

    def get_batch(self, split: str, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.train if split == "train" else self.val
        if len(data) <= self.seq_len + 1:
            raise ValueError("语料太短，无法构造一个完整序列")
        hi = len(data) - self.seq_len - 1
        ix = torch.randint(0, hi, (batch_size,), generator=self._gen)
        x = torch.stack([data[i: i + self.seq_len] for i in ix.tolist()])
        y = torch.stack([data[i + 1: i + 1 + self.seq_len] for i in ix.tolist()])
        return x.to(self.train.device), y.to(self.train.device)

    def stats(self) -> Dict[str, int]:
        return {
            "train_tokens": int(self.train.numel()),
            "val_tokens": int(self.val.numel()),
            "seq_len": self.seq_len,
            "vocab_size": self.vocab_size,
        }
