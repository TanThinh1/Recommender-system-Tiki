import json
import logging
import os
import pickle
import sys
import uuid
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline import run_pipeline

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
MODEL_PATH   = BASE_DIR / "als_model.pkl"
META_PATH    = BASE_DIR / "als_meta.json"
FAISS_PATH   = BASE_DIR / "faiss_index.bin"
ID_MAP_PATH  = BASE_DIR / "faiss_id_map.json"

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("recommend_api")

# ══════════════════════════════════════════════════════════════════════
#  GLOBAL STATE — load một lần khi khởi động
# ══════════════════════════════════════════════════════════════════════
class ModelStore:
    """Giữ toàn bộ model artifacts trong memory."""

    als_model    = None
    user2idx     : dict = {}
    product2idx  : dict = {}
    idx2user     : dict = {}
    idx2product  : dict = {}
    user_item    = None        # scipy sparse (n_users × n_products)
    popular_items: list = []

    faiss_index  = None
    faiss_id_map : list = []   # faiss_idx → product_id

    meta         : dict = {}
    loaded_at    : Optional[str] = None

store = ModelStore()


def _load_als():
    """Load ALS model từ pickle."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy {MODEL_PATH}. Hãy chạy train_model.py trước.")

    log.info(f"Load ALS model  ← {MODEL_PATH.name}")
    t0 = time.time()
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)

    store.als_model   = data["model"]
    store.user2idx    = data["user2idx"]
    store.product2idx = data["product2idx"]
    store.idx2user    = data["idx2user"]
    store.idx2product = data["idx2product"]
    store.user_item   = data["user_item"]
    store.popular_items = data.get("popular_items", [])
    log.info(f"ALS loaded  {len(store.user2idx):,} users · {len(store.product2idx):,} products  ({time.time()-t0:.1f}s)")


def _load_faiss():
    """Load FAISS index + id map."""
    import faiss  # lazy import — chỉ cần khi dùng /similar

    if not FAISS_PATH.exists():
        log.warning(f"FAISS index không tồn tại ({FAISS_PATH}). /similar sẽ bị tắt.")
        return

    log.info(f"Load FAISS index  ← {FAISS_PATH.name}")
    t0 = time.time()
    store.faiss_index  = faiss.read_index(str(FAISS_PATH))
    store.faiss_id_map = json.loads(ID_MAP_PATH.read_text(encoding="utf-8"))
    log.info(f"FAISS loaded  {store.faiss_index.ntotal:,} vectors  ({time.time()-t0:.2f}s)")


def _load_meta():
    if META_PATH.exists():
        store.meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        log.info(f"Meta: trained_at={store.meta.get('trained_at','?')}  "
                 f"precision@5={store.meta.get('metrics',{}).get('precision@5','?')}")


# ══════════════════════════════════════════════════════════════════════
#  DB (optional — dùng cho /feedback)
# ═════════════════════════════════════════════════════════════════════
try:
    sys.path.insert(0, str(BASE_DIR.parent))
    from db.connection import get_db
    db = get_db()
    log.info("MongoDB connected")
except Exception as e:
    db = None
    log.warning(f"MongoDB không kết nối được ({e}). /feedback sẽ bị tắt.")


def _smoke_test_similar() -> None:
    """
    Chạy 1 lần khi API khởi động — phát hiện 0% score trước khi serve traffic.

    Lớp bảo vệ cuối: kiểm tra toàn bộ chuỗi FAISS → pipeline → score > 0.
    Nếu thất bại, log cảnh báo rõ ràng thay vì âm thầm trả về 0% cho user.
    """
    if store.faiss_index is None or db is None:
        log.info("Smoke test skipped (FAISS hoặc DB chưa sẵn sàng)")
        return

    try:
        sample_id   = store.faiss_id_map[0]
        candidates  = _faiss_similar(sample_id, n=5)
        result      = run_pipeline(
            raw_candidates = candidates,
            user_id        = "__smoke__",
            db             = db,
            n              = 5,
            score_weight   = 0.8,
            rating_weight  = 0.2,
        )
        scores = [item["score"] for item in result.items]
        if not scores:
            log.warning(" Smoke test: pipeline trả về 0 items — kiểm tra FAISS index")
            return
        if all(s == 0.0 for s in scores):
            log.error(
                " Smoke test FAILED — toàn bộ scores = 0.0\n"
                "   Nguyên nhân có thể:\n"
                "   • DB enrichment lỗi (kiểm tra product_id field trong MongoDB)\n"
                "   • FAISS score chưa được remap về [0,1]\n"
                "   • rating_avg missing trong products collection\n"
                f"   sample_id={sample_id}  candidates={len(candidates)}"
            )
        else:
            log.info(f"✅ Smoke test passed — sample scores: {scores}")
    except Exception as e:
        log.error(f" Smoke test exception: {e}")


# ══════════════════════════════════════════════════════════════════════
#  LIFESPAN — load tất cả khi app khởi động
# ══════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("═" * 55)
    log.info("  RECOMMEND API  — khởi động")
    log.info("═" * 55)
    _load_meta()
    _load_als()
    _load_faiss()
    store.loaded_at = datetime.now().isoformat()
    _smoke_test_similar()   # kiểm tra score > 0 trước khi serve traffic
    log.info(" Sẵn sàng phục vụ")
    yield
    log.info("Shutting down…")


# ══════════════════════════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="Product Recommendation API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # thu hẹp lại khi deploy production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════
#  SCHEMAS
# ══════════════════════════════════════════════════════════════════════
class RecommendItem(BaseModel):
    product_id : str
    score      : float
    source     : str   # "als" | "popular" | "similar"

class RecommendResponse(BaseModel):
    user_id           : str
    items             : list[RecommendItem]
    is_cold           : bool              # True = cold-start fallback
    latency_ms        : float
    recommendation_id : str = Field(default_factory=lambda: str(uuid.uuid4()))  # (3) online monitoring

class SimilarResponse(BaseModel):
    product_id  : str
    similar     : list[dict]      # [{product_id, similarity}]
    latency_ms  : float

class FeedbackRequest(BaseModel):
    user_id           : str
    product_id        : str
    action            : str   = "click"         # click | purchase | view | add_to_cart | impression
    weight            : float = 0.3
    source            : str   = "api_feedback"
    recommendation_id : Optional[str] = None    # (3) gửi kèm khi click từ gợi ý để tracking CTR


# ══════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════
def _popular_fallback(n: int) -> list[RecommendItem]:
    """Trả về top-N popular items (cold-start)."""
    return [
        RecommendItem(product_id=pid, score=1.0 - i * 0.01, source="popular")
        for i, pid in enumerate(store.popular_items[:n])
    ]


def _als_recommend(user_id: str, n: int, filter_owned: bool) -> tuple[list[RecommendItem], bool]:
    """
    Gợi ý bằng ALS.
    Trả về (items, is_cold).
    is_cold=True khi user không có trong model → dùng popularity fallback.
    """
    u_idx = store.user2idx.get(user_id)

    if u_idx is None:
        # Cold-start: user mới / quá ít tương tác
        return _popular_fallback(n), True

    user_row = store.user_item[u_idx]   # 1 row sparse
    item_ids, scores = store.als_model.recommend(
        u_idx,
        user_row,
        N=n,
        filter_already_liked_items=filter_owned,
    )

    items = [
        RecommendItem(
            product_id=store.idx2product[int(iid)],
            score=round(float(s), 4),
            source="als",
        )
        for iid, s in zip(item_ids, scores)
    ]
    return items, False


def _faiss_similar(product_id: str, n: int) -> list[dict]:
    """
    Tìm n sản phẩm tương tự bằng FAISS cosine similarity.
    """
    if store.faiss_index is None:
        raise HTTPException(503, "FAISS index chưa được load.")

    id_to_idx = {pid: i for i, pid in enumerate(store.faiss_id_map)}
    q_idx = id_to_idx.get(product_id)
    if q_idx is None:
        raise HTTPException(404, f"product_id '{product_id}' không có trong FAISS index.")

    query_vec = store.faiss_index.reconstruct(q_idx).reshape(1, -1)
    scores, indices = store.faiss_index.search(query_vec, n + 1)  # +1 vì có chính nó

    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0 or idx >= len(store.faiss_id_map):
            continue
        pid = store.faiss_id_map[idx]
        if pid == product_id:
            continue
        # FIX Bug 4: remap cosine similarity [-1, 1] → [0, 1]
        # IndexFlatIP + normalized vectors → inner product ∈ [-1, 1]
        # Nếu giữ raw value: score âm bị clamp về 0 → pipeline nhận 0 → hiển thị 0%
        sim_score = (float(score) + 1.0) / 2.0
        results.append({"product_id": pid, "score": round(sim_score, 4), "source": "similar"})
        if len(results) >= n:
            break
    return results


# ══════════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════════════

# ── 1. Health check ──────────────────────────────────────────────────
@app.get("/health")
def health():
    """Trạng thái model + thông số đã train."""
    return {
        "status"       : "ok",
        "loaded_at"    : store.loaded_at,
        "als_users"    : len(store.user2idx),
        "als_products" : len(store.product2idx),
        "faiss_vectors": store.faiss_index.ntotal if store.faiss_index else 0,
        "popular_count": len(store.popular_items),
        "meta"         : store.meta,
    }


# ── 2. ALS Recommendation ────────────────────────────────────────────
@app.get("/recommend/{user_id}")
def recommend(
    user_id      : str,
    n            : int  = Query(default=10, ge=1, le=50),
    filter_owned : bool = Query(default=True),
    debug        : bool = Query(default=False),
):
    t0 = time.perf_counter()
    if store.als_model is None:
        raise HTTPException(503, "Model chưa được load.")
    # Lấy candidates từ ALS (nhiều hơn n để pipeline có đủ để lọc)
    pool = min(n * 5, 100)
    raw_items, is_cold = _als_recommend(user_id, pool, filter_owned=False)
    # Lấy owned_ids nếu cần filter
    owned = set()
    if filter_owned and db is not None:
        u_idx = store.user2idx.get(user_id)
        if u_idx is not None:
            owned = {
                store.idx2product[i]
                for i in store.user_item[u_idx].indices
            }
    # Chạy pipeline
    raw_candidates = [{"product_id": r.product_id, "score": r.score}
                      for r in raw_items]
    result = run_pipeline(
        raw_candidates = raw_candidates,
        user_id        = user_id,
        db             = db,
        n              = n,
        owned_ids      = owned,
        debug          = debug,
    )
    latency = round((time.perf_counter() - t0) * 1000, 2)
    log.info(f"recommend  user={user_id[:12]}  n={n}  cold={is_cold}  {latency}ms")
    return {
        "user_id"         : user_id,
        "is_cold"         : is_cold,
        "items"           : result.items,
        "total_candidates": result.total_candidates,
        "after_filter"    : result.after_filter,
        "latency_ms"      : latency,
    }


# ── 3. Similar Products (FAISS) ──────────────────────────────────────
@app.get("/similar/{product_id}")
def similar(
    product_id : str,
    n          : int  = Query(default=10, ge=1, le=50),
    debug      : bool = Query(default=False),
):
    t0 = time.perf_counter()

    # Guard: FAISS phải được load trước
    if store.faiss_index is None:
        raise HTTPException(503, "FAISS index chưa được load. Hãy chạy generate_embeddings.py trước.")

    try:
        raw_candidates = _faiss_similar(product_id, n * 5)  # pool lớn hơn
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"similar  FAISS search error: {e}")
        raise HTTPException(500, f"Lỗi FAISS search: {e}")

    result = run_pipeline(
        raw_candidates = raw_candidates,
        user_id        = f"similar:{product_id}",
        db             = db,
        n              = n,
        owned_ids      = {product_id},   # loại chính nó khỏi kết quả
        # FIX Bug 1: score_weight=0.8 cho /similar — FAISS similarity là tín hiệu chính.
        # Default score_weight=0.2 / rating_weight=0.8 đúng cho ALS (rating quan trọng hơn),
        # nhưng với content-based similarity thì ngược lại → score → ~0 → 0%.
        score_weight   = 0.8,
        rating_weight  = 0.2,
        debug          = debug,
    )
    latency = round((time.perf_counter() - t0) * 1000, 2)
    log.info(f"similar  product={product_id}  n={n}  {latency}ms")
    return {
        "product_id": product_id,
        "items"     : result.items,
        "latency_ms": latency,
    }


# ── 4. Hybrid (ALS + FAISS rerank) ───────────────────────────────────
@app.get("/hybrid/{user_id}", response_model=RecommendResponse)
def hybrid(
    user_id      : str,
    n            : int   = Query(default=10, ge=1, le=50),
    als_weight   : float = Query(default=0.7, ge=0.0, le=1.0, description="Trọng số ALS (0–1)"),
    filter_owned : bool  = Query(default=True),
    debug        : bool  = Query(default=False),
):
    """
    Kết hợp ALS score + FAISS similarity rồi đi qua run_pipeline():
      hybrid_score = als_weight × als_norm + (1−als_weight) × faiss_norm
      → pipeline: filter owned / stock=0, enrich DB, re-rank với rating

    Hữu ích khi muốn cân bằng giữa "phù hợp user" và "tương tự SP đang xem".
    """
    t0 = time.perf_counter()

    if store.als_model is None:
        raise HTTPException(503, "Model chưa được load.")

    # Pool lớn hơn n để pipeline còn đủ candidates sau khi lọc
    pool_size = min(n * 5, 100)
    als_items, is_cold = _als_recommend(user_id, pool_size, filter_owned=False)

    # ── Lấy owned_ids để truyền vào pipeline ────────────────────────
    owned: set[str] = set()
    if filter_owned:
        u_idx = store.user2idx.get(user_id)
        if u_idx is not None:
            owned = {
                store.idx2product[i]
                for i in store.user_item[u_idx].indices
            }

    # ── Cold-start hoặc không có FAISS → dùng ALS/popular qua pipeline
    if is_cold or store.faiss_index is None:
        raw_candidates = [
            {"product_id": item.product_id, "score": item.score, "source": item.source}
            for item in als_items
        ]
        result = run_pipeline(
            raw_candidates = raw_candidates,
            user_id        = user_id,
            db             = db,
            n              = n,
            owned_ids      = owned,
            debug          = debug,
        )
        latency = round((time.perf_counter() - t0) * 1000, 2)
        items = [
            RecommendItem(product_id=it["product_id"], score=it["score"], source=it["source"])
            for it in result.items
        ]
        return RecommendResponse(user_id=user_id, items=items, is_cold=is_cold, latency_ms=latency)

    # ── Normalize ALS scores → [0, 1] ───────────────────────────────
    als_scores = {item.product_id: item.score for item in als_items}
    max_s = max(als_scores.values()) or 1.0
    als_norm = {pid: s / max_s for pid, s in als_scores.items()}

    # ── FAISS: user vector = trung bình embedding SP đã mua ──────────
    u_idx    = store.user2idx.get(user_id)
    user_row = store.user_item[u_idx]
    bought_indices = user_row.indices.tolist()
    faiss_norm: dict[str, float] = {}

    if bought_indices:
        try:
            id_to_faiss = {pid: i for i, pid in enumerate(store.faiss_id_map)}
            faiss_indices = [
                id_to_faiss[store.idx2product[bi]]
                for bi in bought_indices
                if store.idx2product[bi] in id_to_faiss
            ]
            if faiss_indices:
                # reconstruct() chỉ hoạt động với IndexFlat; nếu index type khác
                # (IVF, HNSW…) thì bỏ qua phần FAISS, dùng ALS-only.
                vecs     = np.array([store.faiss_index.reconstruct(fi) for fi in faiss_indices], dtype="float32")
                user_vec = vecs.mean(axis=0, keepdims=True)
                norm     = np.linalg.norm(user_vec)
                if norm > 0:
                    user_vec /= norm
                faiss_scores_arr, faiss_ids_arr = store.faiss_index.search(user_vec, pool_size + 10)
                for fi, fs in zip(faiss_ids_arr[0], faiss_scores_arr[0]):
                    if fi >= 0:
                        pid = store.faiss_id_map[fi]
                        # Normalized vectors → inner product ∈ [-1, 1]; map → [0, 1]
                        faiss_norm[pid] = (float(fs) + 1) / 2
        except Exception as e:
            log.warning(f"hybrid  FAISS reconstruct failed ({e}); fallback ALS-only")
            faiss_norm = {}  # tiếp tục với ALS score thuần

    # ── Merge scores ─────────────────────────────────────────────────
    all_pids = set(als_norm) | set(faiss_norm)
    merged: dict[str, float] = {}
    for pid in all_pids:
        a = als_norm.get(pid, 0.0)
        f = faiss_norm.get(pid, 0.0)   # đã map về [0,1] ở trên
        merged[pid] = als_weight * a + (1 - als_weight) * f

    # ── Đưa vào run_pipeline() để: filter owned/stock, enrich, re-rank
    raw_candidates = [
        {"product_id": pid, "score": score, "source": "hybrid"}
        for pid, score in merged.items()
    ]
    result = run_pipeline(
        raw_candidates = raw_candidates,
        user_id        = user_id,
        db             = db,
        n              = n,
        owned_ids      = owned,
        debug          = debug,
    )

    latency = round((time.perf_counter() - t0) * 1000, 2)
    log.info(f"hybrid  user={user_id[:12]}  n={n}  als_w={als_weight}  "
             f"pool={len(raw_candidates)}→{len(result.items)}  {latency}ms")

    items = [
        RecommendItem(product_id=it["product_id"], score=it["score"], source=it["source"])
        for it in result.items
    ]
    return RecommendResponse(user_id=user_id, items=items, is_cold=False, latency_ms=latency)


# ── 5. Feedback / Logging ────────────────────────────────────────────
@app.post("/feedback", status_code=202)
def feedback(req: FeedbackRequest):
    """
    Ghi nhận tương tác người dùng (click, view, purchase, impression).
    Nếu có recommendation_id → hành động này được liên kết với một lần gợi ý
    cụ thể, dùng để tính CTR / Conversion Rate trong /metrics/online.

    Trả về 202 Accepted — ghi async vào MongoDB.
    """
    if db is None:
        raise HTTPException(503, "MongoDB không kết nối. Không thể lưu feedback.")

    doc = {
        "user_id"    : req.user_id,
        "product_id" : req.product_id,
        "action"     : req.action,
        "weight"     : req.weight,
        "source"     : req.source,
        "timestamp"  : datetime.utcnow(),
    }
    # (3) Lưu recommendation_id nếu có → phục vụ online monitoring
    if req.recommendation_id:
        doc["recommendation_id"] = req.recommendation_id

    db.interactions.insert_one(doc)
    log.info(
        f"feedback  user={req.user_id[:12]}  product={req.product_id}"
        f"  action={req.action}"
        + (f"  rec_id={req.recommendation_id[:8]}…" if req.recommendation_id else "")
    )
    return {"status": "accepted"}


# ── 6. Online Metrics ─────────────────────────────────────────────────
@app.get("/metrics/online")
def online_metrics(days: int = Query(default=7, ge=1, le=90)):
    """
    (3) Online monitoring — CTR, Conversion Rate, Coverage trong N ngày gần nhất.

    Chỉ tính các interaction có recommendation_id (tức là đến từ gợi ý của hệ thống).

    - CTR             = số click / số impression
    - Conversion Rate = số purchase / số impression
    - Coverage        = số SP khác nhau được gợi ý / tổng SP trong catalogue
    """
    if db is None:
        raise HTTPException(503, "MongoDB không kết nối.")

    cutoff = datetime.utcnow() - timedelta(days=days)
    base_filter = {
        "recommendation_id": {"$exists": True},
        "timestamp"        : {"$gte": cutoff},
    }

    impressions = db.interactions.count_documents({**base_filter, "action": "impression"})
    clicks      = db.interactions.count_documents({**base_filter, "action": {"$in": ["click", "view"]}})
    purchases   = db.interactions.count_documents({**base_filter, "action": "purchase"})
    add_to_cart = db.interactions.count_documents({**base_filter, "action": "add_to_cart"})

    distinct_products = db.interactions.distinct(
        "product_id",
        {**base_filter, "action": {"$in": ["click", "view", "purchase", "add_to_cart"]}},
    )
    total_products = db.products.count_documents({})

    return {
        "period_days"    : days,
        "impressions"    : impressions,
        "clicks"         : clicks,
        "purchases"      : purchases,
        "add_to_cart"    : add_to_cart,
        "ctr"            : round(clicks / impressions, 4) if impressions else 0,
        "conversion_rate": round(purchases / impressions, 4) if impressions else 0,
        "atc_rate"       : round(add_to_cart / impressions, 4) if impressions else 0,
        "coverage"       : round(len(distinct_products) / total_products, 4) if total_products else 0,
        "distinct_products_shown": len(distinct_products),
        "total_products" : total_products,
    }


# ══════════════════════════════════════════════════════════════════════
#  RELOAD MODEL (dùng sau khi retrain)
# ══════════════════════════════════════════════════════════════════════
@app.post("/admin/reload", include_in_schema=False)
def reload_model(secret: str = Query(...)):
    """
    Reload model + FAISS không cần restart server.
    Gọi sau khi train_model.py / generate_embeddings.py chạy xong.

    Bảo vệ bằng secret query param (set RELOAD_SECRET trong env).
    """
    expected = os.getenv("RELOAD_SECRET", "change-me-in-production")
    if secret != expected:
        raise HTTPException(403, "Invalid secret.")

    log.info("Hot-reload model…")
    _load_als()
    _load_faiss()
    _load_meta()
    store.loaded_at = datetime.now().isoformat()
    return {"status": "reloaded", "loaded_at": store.loaded_at}


# ══════════════════════════════════════════════════════════════════════
# ▶  CHẠY TRỰC TIẾP
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "recommend_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,        # tắt khi deploy production
        log_level="info",
    )