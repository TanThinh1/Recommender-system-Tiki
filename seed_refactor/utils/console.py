from __future__ import annotations

import sys
from datetime import datetime

# ── ANSI escape codes ────────────────────────────────────────────────
C_RESET = "\033[0m"
C_BOLD  = "\033[1m"
C_DIM   = "\033[2m"

C_GREEN = "\033[92m"
C_CYAN  = "\033[96m"
C_YEL   = "\033[93m"
C_RED   = "\033[91m"
C_MAG   = "\033[95m"
C_BLUE  = "\033[94m"

# Alias ngắn dùng trong reporting.py (seed_refactor.config cũ)
C_BOLD_CYAN = C_BOLD + C_CYAN


def log(msg: str, color: str = "", indent: int = 0) -> None:
    """In log có timestamp, màu tuỳ chọn, và thụt đầu dòng.

    Args:
        msg:    Nội dung cần in.
        color:  ANSI escape code (dùng các hằng C_* ở trên). Mặc định không màu.
        indent: Số mức thụt đầu dòng (mỗi mức = 2 dấu cách).
    """
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"{C_DIM}[{ts}]{C_RESET} {'  ' * indent}{color}{msg}{C_RESET}", flush=True)


def log_section(title: str, width: int = 60) -> None:
    """In tiêu đề section có đường kẻ — dùng trong reporting."""
    print(f"\n{C_BOLD}{C_CYAN}{'═' * width}{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}  {title}{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}{'═' * width}{C_RESET}")
