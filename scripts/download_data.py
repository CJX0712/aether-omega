#!/usr/bin/env python
"""下载训练语料（Tiny Shakespeare, ~1.1MB）。

    python scripts/download_data.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aether.data import download_tinyshakespeare, load_text  # noqa: E402


def main() -> None:
    path = download_tinyshakespeare("data/tinyshakespeare.txt")
    text = load_text(path)
    print(f"语料已就绪：{path}")
    print(f"  字符数：{len(text):,}")
    print(f"  字节数：{len(text.encode('utf-8')):,}")
    print(f"  前 120 字：{text[:120]!r}")


if __name__ == "__main__":
    main()
