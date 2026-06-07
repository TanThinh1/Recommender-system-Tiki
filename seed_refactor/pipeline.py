from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Optional, TypedDict

log = logging.getLogger("pipeline")


# ══════════════════════════════════════════════════════════════════════
#  INPUT SCHEMAS — bắt lỗi field name sai tại parse time
# ══════════════════════════════════════════════════════════════════════
class ProductMeta(TypedDict, total=False):
    """Schema cho metadata product lấy từ MongoDB.

    Khai báo tường minh để IDE/mypy báo lỗi ngay nếu field name sai.
    Phòng ngừa Bug 2 (sai _id vs product_id) và Bug 5 (rating vs rating_avg).
    """
    product_id : str
    price      : float
    rating     : float   # review rating theo interaction (nếu có)
    rating_avg : float   # trung bình rating SP — field chính dùng để rescore
    stock      : int
    category   : str


class CandidateItem(TypedDict, total=False):
    """Schema cho một candidate đi qua pipeline."""
    product_id : str
    score      : float       # ALS / FAISS score (đã remap về [0,1])
    similarity : float       # alias của score từ FAISS output cũ
    source     : str         # "als" | "similar" | "hybrid" | "popular"
    meta       : ProductMeta
    final_score: float

# ── (4) Load LTR model nếu có — fallback về linear combination nếu không ──
_ltr       = None
_LTR_PATH  = os.path.join(os.path.dirname(__file__), "ltr_model.pkl")
if os.path.exists(_LTR_PATH):
    try:
        _ltr = pickle.load(open(_LTR_PATH, "rb"))
        log.info(f"LTR model loaded ✓  features={_ltr['features']}")
    except Exception as _e:
        log.warning(f"Không load được LTR model, fallback về linear: {_e}")

# Mặc định trọng số re-scoring — ghi đè qua run_pipeline() params
DEFAULT_SCORE_WEIGHT  = 0.2   # trọng số cho ALS/FAISS score
DEFAULT_RATING_WEIGHT = 0.8   # trọng số cho product rating (0–5)


# ══════════════════════════════════════════════════════════════════════
#  OUTPUT SCHEMA
# ══════════════════════════════════════════════════════════════════════
@dataclass
class PipelineItem:
    product_id : str
    score      : float
    source     : str = "pipeline"
    meta       : dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    items            : list[dict]   # serializable — dùng trực tiếp trong JSON response
    total_candidates : int          # số candidates ban đầu trước khi lọc
    after_filter     : int          # số candidates còn lại sau khi lọc
    debug_info       : Optional[dict] = None


# ══════════════════════════════════════════════════════════════════════
#  INTERNAL STEPS
# ══════════════════════════════════════════════════════════════════════
def _filter(
    candidates : list[dict],
    owned_ids  : set[str],
    exclude_ids: set[str],
) -> list[dict]:
    """Bỏ sản phẩm user đã sở hữu và các id cần loại trừ."""
    blocked = owned_ids | exclude_ids
    if not blocked:
        return candidates
    return [c for c in candidates if c["product_id"] not in blocked]


def _enrich_from_db(candidates: list[CandidateItem], db: Any) -> list[CandidateItem]:
    """
    (Tuỳ chọn) Kéo thêm metadata từ MongoDB để phục vụ re-ranking.
    Trả về candidates với field 'meta' bổ sung (typed ProductMeta).
    Nếu DB không khả dụng hoặc lỗi, giữ nguyên candidates gốc.

    FIX Bug 2: query đúng bằng field product_id (string) thay vì _id (ObjectId).
    FIX Bug 5: include cả rating và rating_avg trong projection.
    """
    if db is None:
        return candidates

    try:
        ids = [c["product_id"] for c in candidates]
        #  query theo product_id (string), không phải _id (ObjectId)
        #  key dict dùng doc["product_id"] để khớp với candidates
        docs = {
            doc["product_id"]: doc
            for doc in db.products.find(
                {"product_id": {"$in": ids}},
                #  FIX: thêm rating_avg vào projection
                # ltr_model.py và server.js dùng "rating_avg" làm field rating SP
                # nếu chỉ query "rating" mà DB lưu "rating_avg" → rating luôn = 0
                {"product_id": 1, "price": 1, "category": 1,
                 "rating": 1, "rating_avg": 1, "stock": 1},
            )
        }
        enriched_count = 0
        for c in candidates:
            raw_meta = docs.get(c["product_id"], {})
            raw_meta.pop("_id", None)
            raw_meta.pop("product_id", None)
            # Cast sang ProductMeta để downstream code dùng typed access
            meta: ProductMeta = {
                "price"      : float(raw_meta.get("price") or 0),
                "rating"     : float(raw_meta.get("rating") or 0),
                "rating_avg" : float(raw_meta.get("rating_avg") or 0),
                "stock"      : int(raw_meta.get("stock") or 0),
                "category"   : str(raw_meta.get("category") or ""),
            }
            c["meta"] = meta
            if any(v for v in meta.values()):
                enriched_count += 1

        log.debug(f"enrich_from_db: {enriched_count}/{len(candidates)} candidates enriched")
    except Exception as e:
        log.warning(f"enrich_from_db failed (bỏ qua): {e}")

    return candidates


def _rescore(
    candidates    : list[CandidateItem],
    score_weight  : float = DEFAULT_SCORE_WEIGHT,
    rating_weight : float = DEFAULT_RATING_WEIGHT,
) -> list[CandidateItem]:
    if not candidates:
        return candidates

    # ── Đẩy stock=0 xuống cuối (chung cho cả 2 path) ────────────────────
    def _is_out_of_stock(c: dict) -> bool:
        stock = c.get("meta", {}).get("stock", None)
        return stock is not None and stock <= 0

    # ── (4) LTR path ─────────────────────────────────────────────────────
    if _ltr is not None:
        model    = _ltr["model"]
        features = _ltr["features"]

        rows = []
        for c in candidates:
            meta: ProductMeta = c.get("meta", {})
            rows.append({
                "price_norm"              : min(float(meta.get("price", 0) or 0), 1e7) / 1e7,
                # FIX Bug 5: ưu tiên rating_avg — field thực tế trong products collection
                "rating_norm"             : float(meta.get("rating_avg") or meta.get("rating") or 0) / 5.0,
                "stock_flag"              : int(not _is_out_of_stock(c)),
                # cat_purchase_count_norm không có user context ở đây → 0.0
                # sẽ cải thiện khi online monitoring cung cấp thêm user context
                "cat_purchase_count_norm" : 0.0,
            })

        X     = [[r[f] for f in features] for r in rows]
        probs = model.predict_proba(X)[:, 1]   # P(purchase)

        for c, prob in zip(candidates, probs):
            c["final_score"] = 0.0 if _is_out_of_stock(c) else round(float(prob), 4)

        return sorted(candidates, key=lambda c: c["final_score"], reverse=True)

    # ── Linear fallback (giữ nguyên logic cũ) ────────────────────────────
    # Normalize weights về tổng = 1.0
    total_w = score_weight + rating_weight
    if total_w <= 0:
        score_weight, rating_weight = DEFAULT_SCORE_WEIGHT, DEFAULT_RATING_WEIGHT
        total_w = 1.0
    score_w  = score_weight  / total_w
    rating_w = rating_weight / total_w

    #  Dùng absolute clamp thay cho min-max relative normalization.
    # Root cause 0%: k-NN FAISS trả về cosine scores rất gần nhau
    # (ví dụ 0.96, 0.95, 0.94) → rng = max-min ≈ 0.02 → cả nhóm norm_score ≈ 0
    # → final_score ≈ 0 → UI hiển thị 0%.
    # Scores đã được remap về [0,1] tại _faiss_similar → clamp là đủ an toàn.
    scores = [c.get("score", c.get("similarity", 0.0)) for c in candidates]

    for c, raw_score in zip(candidates, scores):
        # Clamp về [0, 1]
        norm_score = min(max(float(raw_score), 0.0), 1.0)

        meta: ProductMeta = c.get("meta", {})
        #  dùng rating_avg (field thực tế trong DB) với fallback sang rating
        rating      = float(meta.get("rating_avg") or meta.get("rating") or 0)
        norm_rating = min(rating / 5.0, 1.0)

        if _is_out_of_stock(c):
            c["final_score"] = 0.0
        else:
            c["final_score"] = round(score_w * norm_score + rating_w * norm_rating, 4)

    return sorted(candidates, key=lambda c: c["final_score"], reverse=True)


# ══════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════
def run_pipeline(
    raw_candidates : list[CandidateItem],
    user_id        : str,
    db             : Any,
    n              : int,
    owned_ids      : set[str] | None = None,
    exclude_ids    : set[str] | None = None,
    score_weight   : float = DEFAULT_SCORE_WEIGHT,
    rating_weight  : float = DEFAULT_RATING_WEIGHT,
    debug          : bool = False,
) -> PipelineResult:
    owned_ids   = owned_ids   or set()
    exclude_ids = exclude_ids or set()

    total = len(raw_candidates)

    # Chuẩn hoá key "similarity" → "score" cho FAISS candidates
    for c in raw_candidates:
        if "score" not in c and "similarity" in c:
            c["score"] = c["similarity"]

    # Step 1 — filter
    filtered = _filter(raw_candidates, owned_ids, exclude_ids)
    after_filter = len(filtered)

    # Step 2 — enrich (metadata từ DB)
    enriched = _enrich_from_db(filtered, db)

    # Step 3 — re-score & sort (với weights có thể tuỳ chỉnh)
    rescored = _rescore(enriched, score_weight=score_weight, rating_weight=rating_weight)

    # Step 4 — top-N
    top = rescored[:n]

    # Serialize thành plain dict (an toàn cho JSON response)
    items = [
        {
            "product_id": c["product_id"],
            "score"     : c.get("final_score", c.get("score", 0.0)),
            "source"    : c.get("source", "pipeline"),
        }
        for c in top
    ]

    debug_info = None
    if debug:
        # Normalize weights để hiển thị giá trị thực dùng
        total_w = (score_weight + rating_weight) or 1.0
        debug_info = {
            "user_id"          : user_id,
            "total"            : total,
            "after_filter"     : after_filter,
            "owned_count"      : len(owned_ids),
            "excluded"         : list(exclude_ids),
            "returned"         : len(items),
            "score_weight_used": round(score_weight / total_w, 4),
            "rating_weight_used": round(rating_weight / total_w, 4),
            "enriched_count"   : sum(1 for c in enriched if c.get("meta")),
        }
        log.debug(f"pipeline debug: {debug_info}")

    log.info(
        f"pipeline  user={user_id[:16]}  "
        f"candidates={total}→{after_filter}→{len(items)}  "
        f"weights=({score_weight:.2f}/{rating_weight:.2f})"
    )

    return PipelineResult(
        items            = items,
        total_candidates = total,
        after_filter     = after_filter,
        debug_info       = debug_info,
    )