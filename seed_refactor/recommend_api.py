import json
import hashlib
import logging
import os
import pickle
import sys
import threading
import uuid
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

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
#  PICKLE INTEGRITY CHECK
#  Mỗi file .pkl phải có file .pkl.sha256 đi kèm chứa SHA-256 hex digest.
#  Tạo file hash sau khi train xong:
#    python -c "
#      import hashlib, pathlib
#      p = pathlib.Path('als_model.pkl')
#      p.with_suffix('.pkl.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest())
#    "
#  Nếu file .sha256 không tồn tại → log warning nhưng vẫn load (backward compat).
#  Nếu file .sha256 tồn tại mà hash sai → raise RuntimeError, từ chối load.
# ══════════════════════════════════════════════════════════════════════
def _verify_and_load_pickle(path: Path) -> object:
    """Load pickle file với integrity check từ file .sha256 đi kèm."""
    hash_path = path.with_suffix(path.suffix + ".sha256")
    raw = path.read_bytes()

    if hash_path.exists():
        expected_hex = hash_path.read_text().strip()
        actual_hex   = hashlib.sha256(raw).hexdigest()
        if actual_hex != expected_hex:
            raise RuntimeError(
                f"Pickle integrity check FAILED: {path.name}\n"
                f"  expected: {expected_hex}\n"
                f"  actual  : {actual_hex}\n"
                "File có thể đã bị sửa đổi hoặc hỏng. Retrain và tạo lại .sha256."
            )
        log.info(f"Pickle integrity OK ✓  {path.name}")
    else:
        log.warning(
            f"Không tìm thấy {hash_path.name} — bỏ qua integrity check. "
            "Tạo file .sha256 sau khi train để bảo vệ model."
        )

    return pickle.loads(raw)  # noqa: S301

# ══════════════════════════════════════════════════════════════════════
#  RATE LIMITING
#  Giới hạn theo IP. Cấu hình qua env vars:
#    RATE_LIMIT_RECOMMEND  (default: "60/minute")  — /recommend, /hybrid
#    RATE_LIMIT_SIMILAR    (default: "120/minute") — /similar (stateless hơn)
#    RATE_LIMIT_FEEDBACK   (default: "200/minute") — /feedback (write-heavy)
#    RATE_LIMIT_STORAGE    (default: "memory://") — dùng "redis://localhost" ở production
#
#  Khi vượt giới hạn → trả về 429 Too Many Requests với header Retry-After.
# ══════════════════════════════════════════════════════════════════════
_STORAGE_URI        = os.getenv("RATE_LIMIT_STORAGE", "memory://")
_LIMIT_RECOMMEND    = os.getenv("RATE_LIMIT_RECOMMEND", "60/minute")
_LIMIT_SIMILAR      = os.getenv("RATE_LIMIT_SIMILAR",  "120/minute")
_LIMIT_FEEDBACK     = os.getenv("RATE_LIMIT_FEEDBACK", "200/minute")

limiter = Limiter(key_func=get_remote_address, storage_uri=_STORAGE_URI)

# ══════════════════════════════════════════════════════════════════════
#  API KEY AUTHENTICATION
#  Bắt buộc đặt API_KEY trong env. Nếu thiếu → server không khởi động.
#  Client gửi header:  X-API-Key: <key>
#  Các endpoint public (health) được exempt qua deps=[] hoặc không dùng Depends.
#
#  Tạo key mạnh:  python -c "import secrets; print(secrets.token_hex(32))"
# ══════════════════════════════════════════════════════════════════════
_API_KEY = os.getenv("API_KEY", "")
if not _API_KEY:
    raise RuntimeError(
        "Biến môi trường API_KEY chưa được đặt.\n"
        "  Tạo key: python -c \"import secrets; print(secrets.token_hex(32))\"\n"
        "  Sau đó: export API_KEY=<key>  (hoặc thêm vào .env)"
    )

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def require_api_key(key: str | None = Security(_api_key_header)) -> None:
    """Dependency — gắn vào mọi endpoint cần bảo vệ."""
    if not key or key != _API_KEY:
        raise HTTPException(status_code=401, detail="API key không hợp lệ hoặc thiếu.")

# ══════════════════════════════════════════════════════════════════════
#  GLOBAL STATE — load một lần khi khởi động
# ══════════════════════════════════════════════════════════════════════
class ModelStore:
    """Giữ toàn bộ model artifacts trong memory.

    Hot-reload dùng _swap() để thay thế store atomically dưới _store_lock,
    đảm bảo request đang chạy không thấy trạng thái nửa-chừng.
    """

    def __init__(self):
        self.als_model    = None
        self.user2idx     : dict = {}
        self.product2idx  : dict = {}
        self.idx2user     : dict = {}
        self.idx2product  : dict = {}
        self.user_item    = None        # scipy sparse (n_users × n_products)
        self.popular_items: list = []

        self.faiss_index   = None
        self.faiss_id_map  : list = []   # faiss_idx → product_id
        self.faiss_id_to_idx: dict = {}  # FIX P1-7: cache reverse map, build once khi load FAISS

        self.meta         : dict = {}
        self.loaded_at    : Optional[str] = None

# Lock bảo vệ swap store — chỉ dùng khi hot-reload, không dùng khi đọc
_store_lock = threading.Lock()
store = ModelStore()


def _swap_store(new_store: ModelStore) -> None:
    """Thay thế global store atomically.

    Các request đang chạy giữ reference đến store cũ (đã được đọc trước khi swap)
    và hoàn thành bình thường. Request mới sau swap thấy store mới.
    """
    global store
    with _store_lock:
        store = new_store
    log.info("Store swapped atomically ✓")


def _load_als(target: ModelStore) -> None:
    """Load ALS model từ pickle vào target store."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy {MODEL_PATH}. Hãy chạy train_model.py trước.")

    log.info(f"Load ALS model  ← {MODEL_PATH.name}")
    t0 = time.time()
    data = _verify_and_load_pickle(MODEL_PATH)

    target.als_model    = data["model"]
    target.user2idx     = data["user2idx"]
    target.product2idx  = data["product2idx"]
    target.idx2user     = data["idx2user"]
    target.idx2product  = data["idx2product"]
    target.user_item    = data["user_item"]
    target.popular_items = data.get("popular_items", [])
    log.info(f"ALS loaded  {len(target.user2idx):,} users · {len(target.product2idx):,} products  ({time.time()-t0:.1f}s)")


def _load_faiss(target: ModelStore) -> None:
    """Load FAISS index + id map vào target store.

    FIX P1-7: build faiss_id_to_idx dict một lần tại đây thay vì
    rebuild O(n) mỗi request trong _faiss_similar().
    """
    import faiss  # lazy import — chỉ cần khi dùng /similar

    if not FAISS_PATH.exists():
        log.warning(f"FAISS index không tồn tại ({FAISS_PATH}). /similar sẽ bị tắt.")
        return

    log.info(f"Load FAISS index  ← {FAISS_PATH.name}")
    t0 = time.time()
    target.faiss_index   = faiss.read_index(str(FAISS_PATH))
    target.faiss_id_map  = json.loads(ID_MAP_PATH.read_text(encoding="utf-8"))
    # Cache reverse map một lần — tránh rebuild O(n) mỗi request /similar
    target.faiss_id_to_idx = {pid: i for i, pid in enumerate(target.faiss_id_map)}
    log.info(f"FAISS loaded  {target.faiss_index.ntotal:,} vectors  ({time.time()-t0:.2f}s)")


def _load_meta(target: ModelStore) -> None:
    if META_PATH.exists():
        target.meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        log.info(f"Meta: trained_at={target.meta.get('trained_at','?')}  "
                 f"precision@5={target.meta.get('metrics',{}).get('precision@5','?')}")


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
    s = ModelStore()
    _load_meta(s)
    _load_als(s)
    _load_faiss(s)
    s.loaded_at = datetime.now().isoformat()
    _swap_store(s)
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

# Gắn limiter vào app state để slowapi middleware hoạt động
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    # Đặt ALLOWED_ORIGINS trong env, phân cách bằng dấu phẩy.
    # Ví dụ: ALLOWED_ORIGINS=https://tiki.vn,https://admin.tiki.vn
    # Để trống (hoặc không set) → không cho phép cross-origin request nào.
    allow_origins=[ o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip() ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
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
    """Trả về top-N popular items (cold-start).

    FIX P2-14: dùng normalized popularity_score thực từ DB thay vì
    score giả tạo (1.0 - i*0.01) không phản ánh giá trị thực.
    store.popular_items là list of (product_id, popularity_score) tuples
    đã được sort desc khi train. Nếu format cũ (list of str), fallback
    về linear decay để backward compatible.
    """
    items = store.popular_items[:n]
    if not items:
        return []

    # Hỗ trợ cả format mới [(pid, score), ...] và format cũ [pid, ...]
    if isinstance(items[0], (list, tuple)) and len(items[0]) == 2:
        max_score = float(items[0][1]) or 1.0   # items đã sort desc, phần tử 0 là max
        return [
            RecommendItem(
                product_id = str(pid),
                score      = round(float(score) / max_score, 4),   # normalize về [0,1]
                source     = "popular",
            )
            for pid, score in items
        ]
    else:
        # Format cũ — backward compat
        return [
            RecommendItem(product_id=str(pid), score=round(1.0 - i * 0.01, 4), source="popular")
            for i, pid in enumerate(items)
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
    Dùng store.faiss_id_to_idx đã được build sẵn khi load — O(1) lookup.
    """
    if store.faiss_index is None:
        raise HTTPException(503, "FAISS index chưa được load.")

    # FIX P1-7: không rebuild dict O(n) ở đây nữa — dùng cache từ _load_faiss()
    q_idx = store.faiss_id_to_idx.get(product_id)
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
        # remap cosine similarity [-1, 1] → [0, 1]
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
@limiter.limit(_LIMIT_RECOMMEND)
def recommend(
    request      : Request,
    user_id      : str,
    n            : int  = Query(default=10, ge=1, le=50),
    filter_owned : bool = Query(default=True),
    debug        : bool = Query(default=False),
    _auth        : None = Depends(require_api_key),
):
    t0 = time.perf_counter()
    if store.als_model is None:
        raise HTTPException(503, "Model chưa được load.")

    # Pool lớn hơn n để pipeline còn đủ candidates sau khi lọc stock=0
    pool = min(n * 5, 100)

    # FIX P1-9: dùng filter_already_liked_items=True trực tiếp trong ALS
    # thay vì filter_owned=False rồi lấy owned_ids riêng để truyền vào pipeline.
    # Cách cũ khiến ALS vẫn đưa SP đã mua vào pool, làm giảm số candidates
    # hữu ích khi user có lịch sử mua nhiều.
    raw_items, is_cold = _als_recommend(user_id, pool, filter_owned=filter_owned)

    raw_candidates = [{"product_id": r.product_id, "score": r.score}
                      for r in raw_items]
    result = run_pipeline(
        raw_candidates = raw_candidates,
        user_id        = user_id,
        db             = db,
        n              = n,
        # owned_ids không cần thiết nữa vì ALS đã lọc — truyền empty set
        owned_ids      = set(),
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
@limiter.limit(_LIMIT_SIMILAR)
def similar(
    request    : Request,
    product_id : str,
    n          : int  = Query(default=10, ge=1, le=50),
    debug      : bool = Query(default=False),
    _auth      : None = Depends(require_api_key),
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
@limiter.limit(_LIMIT_RECOMMEND)
def hybrid(
    request      : Request,
    user_id      : str,
    n            : int   = Query(default=10, ge=1, le=50),
    als_weight   : float = Query(default=0.7, ge=0.0, le=1.0, description="Trọng số ALS (0–1)"),
    filter_owned : bool  = Query(default=True),
    debug        : bool  = Query(default=False),
    _auth        : None  = Depends(require_api_key),
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
@limiter.limit(_LIMIT_FEEDBACK)
def feedback(
    request    : Request,
    req        : FeedbackRequest,
    background : BackgroundTasks,
    _auth      : None = Depends(require_api_key),
):
    """
    Ghi nhận tương tác người dùng (click, view, purchase, impression).
    Nếu có recommendation_id → hành động này được liên kết với một lần gợi ý
    cụ thể, dùng để tính CTR / Conversion Rate trong /metrics/online.

    Trả về 202 Accepted ngay lập tức — ghi DB chạy trong background,
    không block worker thread (FIX P1-6).
    """
    if db is None:
        raise HTTPException(503, "MongoDB không kết nối. Không thể lưu feedback.")

    doc = {
        "user_id"    : req.user_id,
        "product_id" : req.product_id,
        "action"     : req.action,
        "weight"     : req.weight,
        "source"     : req.source,
        "timestamp"  : datetime.now(timezone.utc),
    }
    if req.recommendation_id:
        doc["recommendation_id"] = req.recommendation_id

    # FIX P1-6: insert_one chạy non-blocking trong background
    background.add_task(db.interactions.insert_one, doc)

    log.info(
        f"feedback  user={req.user_id[:12]}  product={req.product_id}"
        f"  action={req.action}"
        + (f"  rec_id={req.recommendation_id[:8]}…" if req.recommendation_id else "")
    )
    return {"status": "accepted"}


# ── 6. Online Metrics ─────────────────────────────────────────────────
@app.get("/metrics/online")
def online_metrics(days: int = Query(default=7, ge=1, le=90), _auth: None = Depends(require_api_key)):
    """
    (3) Online monitoring — CTR, Conversion Rate, Coverage trong N ngày gần nhất.

    Chỉ tính các interaction có recommendation_id (tức là đến từ gợi ý của hệ thống).

    - CTR             = số click / số impression
    - Conversion Rate = số purchase / số impression
    - Coverage        = số SP khác nhau được gợi ý / tổng SP trong catalogue
    """
    if db is None:
        raise HTTPException(503, "MongoDB không kết nối.")

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
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
def reload_model(secret: str = Query(...), _auth: None = Depends(require_api_key)):
    """
    Reload model + FAISS không cần restart server.
    Gọi sau khi train_model.py / generate_embeddings.py chạy xong.

    Bảo vệ bằng hai lớp:
      1. X-API-Key header (require_api_key dependency)
      2. RELOAD_SECRET query param — phải set trong env, không có default
    """
    _expected = os.getenv("RELOAD_SECRET", "")
    if not _expected:
        raise HTTPException(
            500,
            "RELOAD_SECRET chưa được đặt trong env. "
            "Set biến môi trường trước khi dùng endpoint này."
        )
    if secret != _expected:
        raise HTTPException(403, "Reload secret không hợp lệ.")

    log.info("Hot-reload model — building new store…")
    s = ModelStore()
    _load_als(s)
    _load_faiss(s)
    _load_meta(s)
    s.loaded_at = datetime.now().isoformat()
    _swap_store(s)   # atomic — request đang chạy không bị ảnh hưởng
    return {"status": "reloaded", "loaded_at": s.loaded_at}


# ══════════════════════════════════════════════════════════════════════
#   CHẠY TRỰC TIẾP
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