from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import faiss

_HERE   = Path(__file__).parent.resolve()
_PARENT = _HERE.parent
for _p in (_HERE, _PARENT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# CẤU HÌNH
MODEL_NAME   = "paraphrase-multilingual-MiniLM-L12-v2"  # embedding 384 chiều, hỗ trợ tiếng Việt tốt, cân bằng giữa chất lượng và tốc độ
BATCH_SIZE   = 64
FAISS_PATH   = Path(__file__).parent / "faiss_index.bin"
ID_MAP_PATH  = Path(__file__).parent / "faiss_id_map.json" 
FAISS_META_PATH = Path(__file__).parent / "faiss_meta.json"   # lưu metadata về index (type, nlist, nprobe) để load_faiss() dùng lại
EMBED_DIM    = 384
NORMALIZE    = True    # normalize embedding trước khi add vào FAISS → cosine similarity = inner product

IVFFLAT_THRESHOLD = 50_000 # Nếu >50K SP → dùng IVFFlat, nếu ≤50K SP → dùng FlatIP (exact search)

IVFFLAT_NPROBE_RATIO = 0.10   # nprobe = nlist * 0.1, clamp về [16, 256]

# Console colors
from utils.console import log, C_RESET, C_GREEN, C_CYAN, C_RED, C_BOLD, C_DIM, C_YEL

# XÂY DỰNG TEXT ĐỂ ENCODE
def build_text(product: dict) -> str:
    """
    Ghép các trường quan trọng thành 1 chuỗi để encode.
    Thứ tự ưu tiên: name (quan trọng nhất) → brand → category → tags
                    → specifications → description
    Lặp name 2 lần để tăng trọng số → model tập trung vào tên SP hơn.
    Description dùng tối đa 300 ký tự — đủ để capture nội dung chính
    mà không làm loãng signal từ name/category.
    """
    parts = []

    name = (product.get("name") or "").strip()
    if name:
        parts.append(name)
        parts.append(name)  # lặp name để tăng trọng số

    brand = (product.get("brand") or "").strip()
    if brand and brand != "No Brand":
        parts.append(brand)

    category = (product.get("category") or "").strip()
    if category:
        parts.append(category)

    tags = product.get("tags") or []
    if tags:
        parts.append(" ".join(tags[:8]))

    # Thông số kỹ thuật — trích key-value quan trọng (tối đa 5 cặp)
    specs = product.get("specifications") or {}
    if isinstance(specs, dict) and specs:
        spec_parts = [f"{k}: {v}" for k, v in list(specs.items())[:5]]
        parts.append(" | ".join(spec_parts))

    # Mô tả đầy đủ — tăng từ 100 lên 300 ký tự
    desc = (product.get("description") or "").strip()[:300]
    if desc:
        parts.append(desc)

    return " | ".join(parts)

# ENCODE BATCH
def encode_products(
    model,
    products: list[dict],
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """
    Encode toàn bộ sản phẩm theo batch.
    Trả về numpy array shape (N, 384), float32, normalized nếu NORMALIZE=True.
    """
    texts = [build_text(p) for p in products]
    total = len(texts)
    all_embeddings = []
    start = time.time()

    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        embs  = model.encode(
            batch,
            normalize_embeddings=NORMALIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        all_embeddings.append(embs)

        done    = min(i + batch_size, total)
        elapsed = time.time() - start
        speed   = done / elapsed if elapsed > 0 else 0
        eta     = (total - done) / speed if speed > 0 else 0
        log(
            f"Encoded {done:,}/{total:,}  "
            f"({done/total*100:.1f}%)  "
            f"speed {speed:.0f} SP/s  "
            f"ETA {eta:.0f}s",
            C_DIM, 1
        )

    return np.vstack(all_embeddings).astype("float32")

# BUILD FAISS INDEX  [v2: auto IndexFlatIP / IndexIVFFlat]
def _calc_nlist(n: int) -> int:
    """
    Tính nlist theo heuristic:
      nlist ≈ sqrt(N), tối thiểu 64.
    Constraint của FAISS IVF: cần ít nhất 39 * nlist training points.
    → Giới hạn trên: n // 39 để tránh lỗi khi N nhỏ.
    """
    nlist = max(64, int(math.sqrt(n)))
    nlist = min(nlist, n // 39)   # FAISS yêu cầu n_train >= 39 * nlist
    return max(nlist, 1)

def _calc_nprobe(nlist: int) -> int:
    """nprobe = nlist * ratio, clamp về [16, 256]."""
    return max(16, min(256, round(nlist * IVFFLAT_NPROBE_RATIO)))

def build_faiss_index(embeddings: np.ndarray) -> tuple[faiss.Index, dict]:
    """
    Tự động chọn index type phù hợp với kích thước catalog.

    Returns
    -------
    index     : faiss.Index đã add embeddings và sẵn sàng search
    meta      : dict mô tả index (type, nlist, nprobe, n_vectors, dim)
                → lưu vào faiss_meta.json để load_faiss() dùng lại
    """
    n, dim = embeddings.shape

    if n <= IVFFLAT_THRESHOLD:
        # ── Flat index: exact search, phù hợp ≤50K SP ─────────────────
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        meta = {
            "index_type" : "IndexFlatIP",
            "n_vectors"  : n,
            "dim"        : dim,
            "nlist"      : None,
            "nprobe"     : None,
            "normalize"  : NORMALIZE,
        }
        log(
            f"FAISS {C_GREEN}IndexFlatIP{C_RESET}: "
            f"{n:,} vectors, dim={dim}  (exact search)",
            C_GREEN, 1
        )
    else:
        # ── IVFFlat: approximate search, phù hợp >50K SP ───────────────
        nlist  = _calc_nlist(n)
        nprobe = _calc_nprobe(nlist)

        quantizer = faiss.IndexFlatIP(dim)
        index     = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

        log(f"Training IVFFlat  nlist={nlist}  (n={n:,}) ...", C_DIM, 1)
        index.train(embeddings)    # cần train trước khi add
        index.add(embeddings)

        # make_direct_map() → cho phép gọi index.reconstruct(i)
        # Cần thiết cho get_similar_products() và get_candidates_for_user()
        index.make_direct_map()

        index.nprobe = nprobe      # số cluster duyệt khi query

        meta = {
            "index_type" : "IndexIVFFlat",
            "n_vectors"  : n,
            "dim"        : dim,
            "nlist"      : nlist,
            "nprobe"     : nprobe,
            "normalize"  : NORMALIZE,
        }
        log(
            f"FAISS {C_YEL}IndexIVFFlat{C_RESET}: "
            f"{n:,} vectors, dim={dim}  nlist={nlist}  nprobe={nprobe}",
            C_YEL, 1
        )

    return index, meta

# ══════════════════════════════════════════════════════════════════════
# KIỂM TRA CHẤT LƯỢNG
# ══════════════════════════════════════════════════════════════════════
def quality_check(
    model,
    index,
    products: list[dict],
    id_map: list[str],
    n_samples: int = 5,
):
    """
    Query K nearest neighbors cho N sản phẩm ngẫu nhiên.
    In kết quả để kiểm tra xem gợi ý có hợp lý không.
    """
    import random

    prod_map = {p["product_id"]: p for p in products}

    print(f"\n{C_BOLD}{C_CYAN}  🔍 KIỂM TRA CHẤT LƯỢNG — {n_samples} sản phẩm mẫu{C_RESET}")
    print(f"  {'─'*56}")

    samples = random.sample(products, min(n_samples, len(products)))

    for prod in samples:
        query_text = build_text(prod)
        query_vec  = model.encode(
            [query_text],
            normalize_embeddings=NORMALIZE,
            convert_to_numpy=True,
        ).astype("float32")

        scores, indices = index.search(query_vec, 6)

        print(f"\n  Query: {C_BOLD}{prod['name'][:50]}{C_RESET}")
        print(f"  [{prod['category']}]  {prod['brand']}  {prod.get('price_range','')}")
        print(f"  Top 5 similar:")

        for rank, (idx, score) in enumerate(
            zip(indices[0][1:], scores[0][1:]), 1
        ):
            if idx < 0 or idx >= len(id_map):
                continue
            pid  = id_map[idx]
            near = prod_map.get(pid, {})
            name = near.get("name", "?")[:45]
            cat  = near.get("category", "?")
            sim  = f"{score:.3f}"
            mark = "ok" if cat == prod["category"] else " not ok"
            print(f"    {rank}. {mark} [{sim}] {name}  [{cat}]")

    print(f"\n  {'─'*56}")
    print(f"  ok = cùng danh mục (tốt)  not ok = khác danh mục (kiểm tra lại)")

# ══════════════════════════════════════════════════════════════════════
# LƯU / ĐỌC FAISS
# ══════════════════════════════════════════════════════════════════════
def save_faiss(index: faiss.Index, id_map: list[str], meta: dict):
    """Lưu FAISS index, ID map, và metadata."""
    faiss.write_index(index, str(FAISS_PATH))
    ID_MAP_PATH.write_text(json.dumps(id_map, ensure_ascii=False), encoding="utf-8")
    FAISS_META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Đã lưu FAISS index → {FAISS_PATH}", C_GREEN, 1)
    log(f"Đã lưu ID map      → {ID_MAP_PATH}", C_GREEN, 1)
    log(f"Đã lưu FAISS meta  → {FAISS_META_PATH}", C_GREEN, 1)


def load_faiss() -> tuple[faiss.Index, list[str]]:
    """
    Load FAISS index và ID map.
    Đọc faiss_meta.json để restore nprobe cho IVFFlat,
         tránh silent fallback về nprobe=1 (mặc định sau khi load).
    """
    index  = faiss.read_index(str(FAISS_PATH))
    id_map = json.loads(ID_MAP_PATH.read_text(encoding="utf-8"))

    # Restore nprobe nếu là IVFFlat
    if FAISS_META_PATH.exists():
        meta = json.loads(FAISS_META_PATH.read_text(encoding="utf-8"))
        if meta.get("index_type") == "IndexIVFFlat":
            nprobe = meta.get("nprobe") or _calc_nprobe(meta.get("nlist") or 64)
            index.nprobe = nprobe
            log(f"[load_faiss] IndexIVFFlat: nprobe={nprobe} restored", C_DIM)
    else:
        # Không có meta → heuristic fallback
        if hasattr(index, "nprobe"):
            index.nprobe = 16
            log("[load_faiss] Không có faiss_meta.json → nprobe=16 (fallback)", C_YEL)

    return index, id_map

# ══════════════════════════════════════════════════════════════════════
# BULK UPDATE MONGODB
# ══════════════════════════════════════════════════════════════════════
def save_embeddings_to_mongo(db, products: list[dict], embeddings: np.ndarray):
    """
    Cập nhật field item_embedding cho từng sản phẩm trong MongoDB.
    Dùng bulk_write để nhanh hơn update từng doc.
    """
    from pymongo import UpdateOne

    CHUNK = 500
    total = len(products)
    ops   = []

    for i, (prod, vec) in enumerate(zip(products, embeddings)):
        ops.append(UpdateOne(
            {"product_id": prod["product_id"]},
            {"$set": {"item_embedding": vec.tolist()}},
        ))

        if len(ops) >= CHUNK:
            db.products.bulk_write(ops, ordered=False)
            print(f"  ↳ {min(i+1, total):,}/{total:,}")
            ops = []

    if ops:
        db.products.bulk_write(ops, ordered=False)
        print(f"  ↳ {total:,}/{total:,}")

    log(f"Đã cập nhật item_embedding cho {total:,} sản phẩm", C_GREEN, 1)

# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print(f"\n{C_BOLD}{C_CYAN}{'═'*60}{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}    GENERATE EMBEDDINGS + FAISS INDEX  v2{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}{'═'*60}{C_RESET}\n")
    print(f"  IndexFlatIP  : N ≤ {IVFFLAT_THRESHOLD:,} SP  (exact search)")
    print(f"  IndexIVFFlat : N >  {IVFFLAT_THRESHOLD:,} SP  (ANN, nprobe={round(64*IVFFLAT_NPROBE_RATIO)}..)\n")

    # ── 0. Kiểm tra thư viện ─────────────────────────────────────────
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        log(" sentence-transformers và faiss-cpu đã cài", C_GREEN)
    except ImportError as e:
        print(f"{C_RED} Thiếu thư viện: {e}{C_RESET}")
        print(f"{C_RED} Chạy: pip install sentence-transformers faiss-cpu numpy{C_RESET}")
        sys.exit(1)

    # ── 1. Kết nối MongoDB ───────────────────────────────────────────
    try:
        from db.connection import get_db
        db = get_db()
        log(" Kết nối MongoDB thành công", C_GREEN)
    except Exception as e:
        print(f"{C_RED} MongoDB error: {e}{C_RESET}")
        sys.exit(1)

    # ── 2. Load sản phẩm ─────────────────────────────────────────────
    log("Đang load sản phẩm từ MongoDB...", C_DIM)
    products = list(db.products.find(
        {},
        {
            "product_id": 1, "name": 1, "brand": 1,
            "category": 1, "tags": 1, "description": 1, "specifications": 1,
            "price_range": 1, "item_embedding": 1,
            "_id": 0,
        }
    ))

    if not products:
        print(f"{C_RED} Không có sản phẩm trong DB. Chạy db/storage.py trước.{C_RESET}")
        sys.exit(1)

    n = len(products)
    log(f"Loaded {n:,} sản phẩm", C_GREEN)

    # Thông báo index type sẽ dùng
    if n <= IVFFLAT_THRESHOLD:
        log(f"→ Sẽ dùng IndexFlatIP  (N={n:,} ≤ {IVFFLAT_THRESHOLD:,})", C_CYAN, 1)
    else:
        nlist  = _calc_nlist(n)
        nprobe = _calc_nprobe(nlist)
        log(f"→ Sẽ dùng IndexIVFFlat  (N={n:,}, nlist={nlist}, nprobe={nprobe})", C_YEL, 1)

    # ── Kiểm tra đã có embedding chưa ────────────────────────────────
    already_done = sum(1 for p in products if p.get("item_embedding"))
    if already_done == len(products) and FAISS_PATH.exists():
        log(f" Tất cả {already_done:,} SP đã có embedding và FAISS index tồn tại.", C_GREEN)
        ans = input("  Re-generate lại không? [y/N]: ").strip().lower()
        if ans != "y":
            log("Bỏ qua. Load FAISS để kiểm tra chất lượng...", C_DIM)
            sentence_model = SentenceTransformer(MODEL_NAME)
            index, id_map  = load_faiss()
            quality_check(sentence_model, index, products, id_map)
            return

    # ── 3. Load sentence model ───────────────────────────────────────
    print(f"\n{C_BOLD}Model: {MODEL_NAME}{C_RESET}")
    log("Đang load model (lần đầu sẽ download ~120MB)...", C_DIM)
    t_total = time.time()
    t0      = time.time()
    sentence_model = SentenceTransformer(MODEL_NAME)
    log(f"Model loaded trong {time.time()-t0:.1f}s", C_GREEN)
    log(f"Embedding dim: {sentence_model.get_sentence_embedding_dimension()}", C_DIM)

    # ── 4. Encode ────────────────────────────────────────────────────
    print(f"\n{C_BOLD}Encoding {n:,} sản phẩm (batch_size={BATCH_SIZE})...{C_RESET}")
    t0         = time.time()
    embeddings = encode_products(sentence_model, products, BATCH_SIZE)
    elapsed    = time.time() - t0
    log(
        f"Encoding hoàn tất: {elapsed:.1f}s  "
        f"({n/elapsed:.0f} SP/s)  "
        f"shape={embeddings.shape}",
        C_GREEN
    )

    # ── 5. Build FAISS index  [v2: auto type] ────────────────────────
    print(f"\n{C_BOLD}Build FAISS index...{C_RESET}")
    id_map        = [p["product_id"] for p in products]
    index, f_meta = build_faiss_index(embeddings)   # ← trả về (index, meta)
    save_faiss(index, id_map, f_meta)

    # ── 6. Lưu embedding vào MongoDB ─────────────────────────────────
    print(f"\n{C_BOLD}Lưu item_embedding vào MongoDB...{C_RESET}")
    save_embeddings_to_mongo(db, products, embeddings)

    # ── 7. Kiểm tra chất lượng ───────────────────────────────────────
    quality_check(sentence_model, index, products, id_map)

    # ── 8. Thống kê cuối ─────────────────────────────────────────────
    size_mb    = FAISS_PATH.stat().st_size / 1_048_576
    total_time = time.time() - t_total
    print(f"\n{C_GREEN}{C_BOLD} HOÀN TẤT!{C_RESET}")
    print(f"   Model     : {MODEL_NAME}")
    print(f"   Sản phẩm  : {n:,}")
    print(f"   Dim       : {embeddings.shape[1]}")
    print(f"   Index     : {f_meta['index_type']}", end="")
    if f_meta["index_type"] == "IndexIVFFlat":
        print(f"  (nlist={f_meta['nlist']}, nprobe={f_meta['nprobe']})")
    else:
        print()
    print(f"   FAISS file: {FAISS_PATH.name}  ({size_mb:.1f} MB)")
    print(f"   ID map    : {ID_MAP_PATH.name}")
    print(f"   Meta      : {FAISS_META_PATH.name}")
    print(f"   Tổng thời gian: {total_time:.0f}s\n")
    print(f"  {C_DIM}Bước tiếp theo: chạy train_model.py{C_RESET}\n")


# ══════════════════════════════════════════════════════════════════════
# 🔎 HELPER: dùng trong recommend_api.py
# ══════════════════════════════════════════════════════════════════════
def get_similar_products(
    query_product_id: str,
    top_k: int = 50,
    same_category_only: bool = False,
    db=None,
) -> list[dict]:
    """
    Trả về top_k sản phẩm tương tự với query_product_id.
    Dùng FAISS để tìm nhanh, trả về list[{product_id, score}].
    [v2] load_faiss() tự động restore nprobe cho IVFFlat.
    """
    if not FAISS_PATH.exists():
        raise FileNotFoundError(
            "faiss_index.bin không tồn tại. Chạy generate_embeddings.py trước."
        )

    index, id_map = load_faiss()   # ← nprobe được restore tự động

    try:
        query_idx = id_map.index(query_product_id)
    except ValueError:
        raise ValueError(f"product_id '{query_product_id}' không có trong FAISS index.")

    query_vec = index.reconstruct(query_idx).reshape(1, -1)

    k = top_k + 1
    scores, indices = index.search(query_vec, k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0 or idx >= len(id_map):
            continue
        pid = id_map[idx]
        if pid == query_product_id:
            continue
        results.append({"product_id": pid, "similarity": round(float(score), 4)})

    return results[:top_k]


def get_candidates_for_user(
    user_id: str,
    top_k: int = 100,
    db=None,
) -> list[dict]:
    """
    Candidate Generation cho một user:
    1. Lấy danh sách SP user đã mua
    2. Lấy embedding trung bình → user vector
    3. FAISS search → top_k candidates
    4. Loại SP đã mua
    [v2] load_faiss() tự động restore nprobe.
    """
    if db is None:
        from db.connection import get_db
        db = get_db()

    purchased = list(db.interactions.find(
        {"user_id": user_id, "action": "purchase"},
        {"product_id": 1, "_id": 0}
    ))
    purchased_ids = {p["product_id"] for p in purchased}

    if not purchased_ids:
        popular = list(db.products.find(
            {"item_embedding": {"$ne": []}},
            {"product_id": 1, "_id": 0}
        ).sort("popularity_score", -1).limit(top_k))
        return [{"product_id": p["product_id"], "similarity": 0.0} for p in popular]

    index, id_map = load_faiss()   # ← nprobe restored
    id_to_idx = {pid: i for i, pid in enumerate(id_map)}

    bought_indices = [id_to_idx[pid] for pid in purchased_ids if pid in id_to_idx]
    if not bought_indices:
        return []

    vecs     = np.array([index.reconstruct(i) for i in bought_indices], dtype="float32")
    user_vec = vecs.mean(axis=0, keepdims=True)

    if NORMALIZE:
        norm = np.linalg.norm(user_vec)
        if norm > 0:
            user_vec /= norm

    k = top_k + len(purchased_ids) + 1
    scores, indices = index.search(user_vec, k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0 or idx >= len(id_map):
            continue
        pid = id_map[idx]
        if pid in purchased_ids:
            continue
        results.append({"product_id": pid, "similarity": round(float(score), 4)})
        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    main()