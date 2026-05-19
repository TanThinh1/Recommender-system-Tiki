"""
processing/product.py — Parse và làm giàu dữ liệu sản phẩm từ Tiki API.
"""

import math
import re
from datetime import datetime

_SENSITIVE_WORDS = frozenset([
    "quần lót", "sịp", "quần sịp", "áo ngực", "bra", "bao cao su",
    "condom", "bcs", "tình dục", "sexy", "đồ lót", "nội y", "lingerie",
    "gợi cảm", "kích dục", "gel bôi trơn",
])

def is_sensitive(name: str) -> bool:
    nl = name.lower()
    return any(w in nl for w in _SENSITIVE_WORDS)

def _price_range(price: int) -> str:
    if price <= 200_000:
        return "budget"
    elif price <= 1_000_000:
        return "mid"
    return "premium"

def _build_tags(name: str, brand: str | None, category: str) -> list[str]:
    words = [re.sub(r"[^\w]", "", w) for w in re.split(r"\s+|[-/,]", name.lower()) if len(w) > 1]
    tags = dict.fromkeys(w for w in words if w)
    if brand:
        tags[brand.lower()] = None
    tags[category.lower()] = None
    return list(tags)[:12]

def _extract_attrs(name: str, category: str) -> dict:
    attrs = {}
    nl = name.lower()
    if category == "Thời trang":
        for c in ["đen","trắng","xanh","đỏ","vàng","xám","be","nâu","hồng"]:
            if c in nl:
                attrs["color"] = c; break
        for m in ["cotton","polyester","linen","denim","len","da","lụa"]:
            if m in nl:
                attrs["material"] = m; break
    elif category == "Thiết bị điện tử":
        if "bluetooth" in nl:
            attrs["connectivity"] = "bluetooth"
        m = re.search(r"(\d{4,6})\s*mah", nl)
        if m:
            attrs["capacity_mah"] = int(m.group(1))
        m = re.search(r"(\d+)\s*gb\b", nl)
        if m:
            attrs["storage_gb"] = int(m.group(1))
    elif category == "Làm đẹp - Sức khỏe":
        for s in ["da dầu","da khô","da hỗn hợp","da nhạy cảm"]:
            if s in nl:
                attrs["skin_type"] = s; break
        m = re.search(r"(\d+)\s*ml\b", nl)
        if m:
            attrs["volume_ml"] = int(m.group(1))
    return attrs


def _clean_html(html: str) -> str:
    """Strip HTML tags, decode entities cơ bản, chuẩn hoá whitespace."""
    if not html:
        return ""
    text = re.sub(r"<[^>]+>", " ", html)
    text = (text
            .replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            .replace("&nbsp;", " ").replace("&quot;", '"').replace("&#39;", "'")
            .replace("&ldquo;", '"').replace("&rdquo;", '"').replace("&hellip;", "…"))
    return re.sub(r"\s+", " ", text).strip()


def _parse_specifications(specs_raw: list) -> dict:
    """
    Chuyển list thông số kỹ thuật từ Tiki API sang dict phẳng.

    Input (từ API):
      [
        {"name": "Thông số kỹ thuật", "attributes": [
          {"code": "weight", "name": "Khối lượng", "value": "250 g"},
          {"code": "origin",  "name": "Xuất xứ",   "value": "Việt Nam"},
        ]}
      ]
    Output:
      {"Khối lượng": "250 g", "Xuất xứ": "Việt Nam"}
    """
    if not specs_raw or not isinstance(specs_raw, list):
        return {}
    result = {}
    for group in specs_raw:
        for attr in group.get("attributes", []):
            name  = (attr.get("name") or "").strip()
            value = (attr.get("value") or "").strip()
            if name and value:
                result[name] = value
    return result


def parse_tiki_product(raw: dict, category: str, index: int) -> dict | None:
    """
    Parse một sản phẩm thô từ Tiki category API.

    `raw` có thể chứa thêm các field từ product detail API nếu đã fetch:
      - raw["description"]    : mô tả đầy đủ (str, có thể HTML)
      - raw["specifications"] : thông số kỹ thuật (list, format Tiki)
    Nếu chưa fetch detail → dùng short_description và specifications rỗng.
    """
    name = raw.get("name", "").strip()
    if not name or len(name) < 5 or is_sensitive(name):
        return None

    price          = raw.get("price", 0) or 0
    original_price = raw.get("original_price", price) or price
    discount       = 0
    if original_price > price > 0:
        discount = round((original_price - price) / original_price * 100)

    rating_avg_api  = float(raw.get("rating_average", 0) or 0) or None
    rating_count_api = int(raw.get("review_count", 0) or 0)
    sold_count      = int((raw.get("quantity_sold") or {}).get("value", 0) or 0)
    popularity_score = round(math.log1p(sold_count) * (rating_avg_api or 3.0) / 5.0, 4)

    brand_data = raw.get("brand") or {}
    brand      = brand_data.get("name") or None

    thumbnail = raw.get("thumbnail_url", "") or ""
    if thumbnail and not thumbnail.startswith("http"):
        thumbnail = f"https:{thumbnail}"

    tiki_product_id = str(raw.get("id", ""))
    url_key         = raw.get("url_key", "")
    product_url     = f"https://tiki.vn/{url_key}/p{tiki_product_id}" if url_key else ""

    # ── Mô tả sản phẩm ───────────────────────────────────────────────
    # Ưu tiên: description đầy đủ (từ detail API) > short_description
    raw_desc    = raw.get("description") or raw.get("short_description") or ""
    description = _clean_html(raw_desc)[:3000]   # strip HTML, giới hạn 3000 ký tự

    # ── Thông số kỹ thuật ────────────────────────────────────────────
    # Ưu tiên: specifications từ detail API (list) > dict rỗng
    specs_raw      = raw.get("specifications") or []
    specifications = _parse_specifications(specs_raw)

    return {
        "product_id"      : f"SP{index:05d}",
        "name"            : name,
        "category"        : category,
        "brand"           : brand,
        "price"           : price,
        "original_price"  : original_price,
        "discount_pct"    : discount,
        "price_range"     : _price_range(price),
        "tags"            : _build_tags(name, brand, category),
        "attributes"      : _extract_attrs(name, category),
        "rating_avg"      : rating_avg_api,
        "rating_count"    : rating_count_api,
        "sold_count"      : sold_count,
        "popularity_score": popularity_score,
        "stock"           : None,
        "image_url"       : thumbnail,
        "local_image_path": "",
        "item_embedding"  : [],
        "tiki_product_id" : tiki_product_id,
        "product_url"     : product_url,
        "description"     : description,       # ✅ mô tả đầy đủ, plain text
        "specifications"  : specifications,    # ✅ thông số kỹ thuật dạng dict phẳng
        "crawled_at"      : datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from seed_refactor.config import TARGET_TOTAL
    from seed_refactor.db.connection import get_db
    from seed_refactor.db.storage import bulk_upsert

    db = get_db()

    raw_products = list(db.raw_products.find({}, {"_id": 0}))
    if not raw_products:
        print("❌ Không có raw_products trong MongoDB. Chạy tiki_crawler.py trước.")
        exit(1)

    if "category" not in raw_products[0]:
        print("❌ raw_products thiếu trường 'category'. Kiểm tra lại tiki_crawler.py.")
        exit(1)

    products = []
    idx = 1
    for raw in raw_products:
        cat = raw.get("category")
        if not cat:
            continue
        prod = parse_tiki_product(raw, cat, idx)
        if prod:
            products.append(prod)
            idx += 1
            if idx > TARGET_TOTAL:
                break

    bulk_upsert(db.products, products, key_field="product_id")
    db.products.create_index("product_id",            unique=True)
    db.products.create_index("category")
    db.products.create_index("brand")
    db.products.create_index([("popularity_score", -1)])

    has_desc = sum(1 for p in products if p.get("description"))
    print(f"✅ Đã lưu {len(products):,} sản phẩm vào MongoDB (products)")
    print(f"   Có mô tả: {has_desc:,} / {len(products):,}")
    print(f"   💡 Chạy crawl_reviews.py để lấy mô tả đầy đủ + reviews")
    "quần lót", "sịp", "quần sịp", "áo ngực", "bra", "bao cao su",
    "condom", "bcs", "tình dục", "sexy", "đồ lót", "nội y", "lingerie",
    "gợi cảm", "kích dục", "gel bôi trơn",


def is_sensitive(name: str) -> bool:
    nl = name.lower()
    return any(w in nl for w in _SENSITIVE_WORDS)

def _price_range(price: int) -> str:
    if price <= 200_000:
        return "budget"
    elif price <= 1_000_000:
        return "mid"
    return "premium"

def _build_tags(name: str, brand: str | None, category: str) -> list[str]:
    words = [re.sub(r"[^\w]", "", w) for w in re.split(r"\s+|[-/,]", name.lower()) if len(w) > 1]
    tags = dict.fromkeys(w for w in words if w)
    if brand:
        tags[brand.lower()] = None
    tags[category.lower()] = None
    return list(tags)[:12]

def _extract_attrs(name: str, category: str) -> dict:
    attrs = {}
    nl = name.lower()
    if category == "Thời trang":
        for c in ["đen","trắng","xanh","đỏ","vàng","xám","be","nâu","hồng"]:
            if c in nl:
                attrs["color"] = c; break
        for m in ["cotton","polyester","linen","denim","len","da","lụa"]:
            if m in nl:
                attrs["material"] = m; break
    elif category == "Thiết bị điện tử":
        if "bluetooth" in nl:
            attrs["connectivity"] = "bluetooth"
        m = re.search(r"(\d{4,6})\s*mah", nl)
        if m:
            attrs["capacity_mah"] = int(m.group(1))
        m = re.search(r"(\d+)\s*gb\b", nl)
        if m:
            attrs["storage_gb"] = int(m.group(1))
    elif category == "Làm đẹp - Sức khỏe":
        for s in ["da dầu","da khô","da hỗn hợp","da nhạy cảm"]:
            if s in nl:
                attrs["skin_type"] = s; break
        m = re.search(r"(\d+)\s*ml\b", nl)
        if m:
            attrs["volume_ml"] = int(m.group(1))
    return attrs

def parse_tiki_product(raw: dict, category: str, index: int) -> dict | None:
    name = raw.get("name", "").strip()
    if not name or len(name) < 5 or is_sensitive(name):
        return None
    price = raw.get("price", 0) or 0
    original_price = raw.get("original_price", price) or price
    discount = 0
    if original_price > price > 0:
        discount = round((original_price - price) / original_price * 100)
    rating_avg_api = float(raw.get("rating_average", 0) or 0) or None
    rating_count_api = int(raw.get("review_count", 0) or 0)
    sold_count = int((raw.get("quantity_sold") or {}).get("value", 0) or 0)
    popularity_score = round(math.log1p(sold_count) * (rating_avg_api or 3.0) / 5.0, 4)
    brand_data = raw.get("brand") or {}
    brand = brand_data.get("name") or None
    thumbnail = raw.get("thumbnail_url", "") or ""
    if thumbnail and not thumbnail.startswith("http"):
        thumbnail = f"https:{thumbnail}"
    tiki_product_id = str(raw.get("id", ""))
    url_key = raw.get("url_key", "")
    product_url = f"https://tiki.vn/{url_key}/p{tiki_product_id}" if url_key else ""
    return {
        "product_id": f"SP{index:05d}",
        "name": name,
        "category": category,
        "brand": brand,
        "price": price,
        "original_price": original_price,
        "discount_pct": discount,
        "price_range": _price_range(price),
        "tags": _build_tags(name, brand, category),
        "attributes": _extract_attrs(name, category),
        "rating_avg": rating_avg_api,
        "rating_count": rating_count_api,
        "sold_count": sold_count,
        "popularity_score": popularity_score,
        "stock": None,
        "image_url": thumbnail,
        "local_image_path": "",
        "item_embedding": [],
        "tiki_product_id": tiki_product_id,
        "product_url": product_url,
        "description": (raw.get("short_description") or "")[:1000],
        "specifications": {},
        "crawled_at": datetime.now().isoformat(),
    }