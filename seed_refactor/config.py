"""
config.py — Toàn bộ hằng số, cấu hình danh mục, và tiện ích log.
"""

import os
import re
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════
# ⚙️  HẰNG SỐ CHÍNH
# ══════════════════════════════════════════════════════════════════════
TARGET_TOTAL = 5_400
MAX_PAGES_PER_CAT = 3
MAX_REVIEW_PAGES_PER_PROD = 5          # FIX BUG-05: đổi tên cho rõ
MIN_REVIEWS_FOR_USER = 1
CART_WEIGHT_RATIO = 0.65               # add_to_cart weight = purchase_weight × 0.65

# FIX DESIGN-03: normalize giá trị env đúng cách
_headless_env = os.getenv("HEADLESS", "1").lower()
HEADLESS = _headless_env not in ("0", "false", "no")

# Đặt None để không giới hạn, hoặc đặt số nguyên để cap
MAX_INTERACTIONS: int | None = None

# ══════════════════════════════════════════════════════════════════════
# 📊 RATING TIERS
# ══════════════════════════════════════════════════════════════════════
RATING_TIERS: dict[str, dict] = {
    "1-2": {
        "view_ratio": 50,
        "cart_ratio": 6,
        "review_rate": 0.35,
        "purchase_weight": 0.2,
    },
    "2-3": {
        "view_ratio": 40,
        "cart_ratio": 5,
        "review_rate": 0.25,
        "purchase_weight": 0.4,
    },
    "3-4": {
        "view_ratio": 30,
        "cart_ratio": 4,
        "review_rate": 0.18,
        "purchase_weight": 0.7,
    },
    "4-5": {
        "view_ratio": 20,
        "cart_ratio": 3,
        "review_rate": 0.15,
        "purchase_weight": 1.0,
    },
}

# ══════════════════════════════════════════════════════════════════════
# 🗂️  DANH MỤC CRAWL
# ══════════════════════════════════════════════════════════════════════
CATEGORY_CONFIG: dict[str, dict] = {
    "Thời trang": {
        "urls": [
            "https://tiki.vn/dam-vay-lien/c941",
            "https://tiki.vn/trang-phuc-the-thao-nam/c6140",
            "https://tiki.vn/trang-phuc-the-thao-nu/c6141",
            "https://tiki.vn/thoi-trang-nam/c915",
            "https://tiki.vn/thoi-trang/c21442",
            "https://tiki.vn/giay-the-thao-nam/c27572",
            "https://tiki.vn/giay-tay-cong-so/c1581",
            "https://tiki.vn/giay-sandals-nam/c5341",
            "https://tiki.vn/giay-cao-got/c8355",
            "https://tiki.vn/giay-bup-be/c1192",
            "https://tiki.vn/dep-guoc-nu/c984",
            "https://tiki.vn/mat-kinh/c8370",
            "https://tiki.vn/dong-ho-nam/c1778",
            "https://tiki.vn/dong-ho-nu/c977",
            "https://tiki.vn/trang-suc/c8374",
            "https://tiki.vn/tui-vi-nu/c976",
            "https://tiki.vn/tui-thoi-trang-nam/c27616",
        ],
        "icon": "👗",
    },
    "Thiết bị điện tử": {
        "urls": [
            "https://tiki.vn/dien-thoai-may-tinh-bang/c1789",
            "https://tiki.vn/thiet-bi-kts-phu-kien-so/c1815",
            "https://tiki.vn/laptop-may-vi-tinh-linh-kien/c1846",
            "https://tiki.vn/dien-tu-dien-lanh/c4221",
            "https://tiki.vn/may-anh/c1801",
            "https://tiki.vn/dien-gia-dung/c1882",
            "https://tiki.vn/dien-thoai-smartphone/c1795",
            "https://tiki.vn/thiet-bi-am-thanh-va-phu-kien/c8215",
            "https://tiki.vn/laptop/c8095",
            "https://tiki.vn/tivi/c5015",
            "https://tiki.vn/may-anh/c28806",
            "https://tiki.vn/may-tinh-bang/c1794",
            "https://tiki.vn/thiet-bi-choi-game-va-phu-kien/c2667",
            "https://tiki.vn/thiet-bi-van-phong-thiet-bi-ngoai-vi/c12884",
            "https://tiki.vn/may-giat/c3862",
            "https://tiki.vn/thiet-bi-deo-thong-minh-va-phu-kien/c8039",
            "https://tiki.vn/pc-may-tinh-bo/c8093",
            "https://tiki.vn/may-lanh-may-dieu-hoa/c3865",
            "https://tiki.vn/tu-lanh/c2328",
            "https://tiki.vn/quat-dien/c2001",
        ],
        "icon": "📱",
    },
    "Làm đẹp - Sức khỏe": {
        "urls": [
            "https://tiki.vn/mat-na-cac-loai/c1601",
            "https://tiki.vn/xit-khoang/c5872",
            "https://tiki.vn/cham-soc-vung-da-mat/c3424",
            "https://tiki.vn/nuoc-hoa-hong-toner/c2347",
            "https://tiki.vn/kem-chong-nang/c3422",
            "https://tiki.vn/san-pham-tri-mun-va-seo/c3426",
            "https://tiki.vn/trang-diem-mat/c1585",
            "https://tiki.vn/son-trang-diem-moi/c1587",
            "https://tiki.vn/lan-xit-khu-mui/c17162",
            "https://tiki.vn/cham-soc-mat/c2162",
            "https://tiki.vn/duong-the/c1610",
            "https://tiki.vn/tay-te-bao-chet-co-the/c8220",
            "https://tiki.vn/nuoc-hoa-nu/c1636",
            "https://tiki.vn/nuoc-hoa-nam/c1637",
            "https://tiki.vn/nuoc-suc-mieng/c1627",
            "https://tiki.vn/kem-danh-rang/c11835",
            "https://tiki.vn/tay-trang-rang/c11837",
            "https://tiki.vn/sua-rua-mat/c1583",
            "https://tiki.vn/chong-lao-hoa-da/c5893",
            "https://tiki.vn/serum/c53350",
        ],
        "icon": "💄",
    },
}

ALL_CATEGORIES = list(CATEGORY_CONFIG.keys())


def catid_from_url(url: str) -> int:
    """Trích category ID từ URL Tiki (dạng /c123)."""
    m = re.search(r'/c(\d+)(?:[/?#]|$)', url.strip())
    if m:
        return int(m.group(1))
    raise ValueError(
        f"Không tìm thấy catid trong URL: {url!r}\n"
        f"URL phải có dạng https://tiki.vn/ten-danh-muc/cSỐ"
    )


# ══════════════════════════════════════════════════════════════════════
# 🎨 CONSOLE COLORS & LOGGING
# ══════════════════════════════════════════════════════════════════════
C_RESET = "\033[0m"
C_GREEN = "\033[92m"
C_RED   = "\033[91m"
C_CYAN  = "\033[96m"
C_BOLD  = "\033[1m"
C_DIM   = "\033[2m"


def log(msg: str, color: str = "", indent: int = 0) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"{C_DIM}[{ts}]{C_RESET} {'  ' * indent}{color}{msg}{C_RESET}")