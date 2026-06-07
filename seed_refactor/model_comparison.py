from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# ── Path setup ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
for _p in (BASE_DIR, BASE_DIR.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ── Console colors ─────────────────────────────────────────────────────
C_RESET = "\033[0m"; C_GREEN = "\033[92m"; C_RED = "\033[91m"
C_CYAN  = "\033[96m"; C_BOLD  = "\033[1m";  C_DIM = "\033[2m"
C_YEL   = "\033[93m"; C_MAG   = "\033[95m"

def log(msg, color="", indent=0):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"{C_DIM}[{ts}]{C_RESET} {'  '*indent}{color}{msg}{C_RESET}")

# ══════════════════════════════════════════════════════════════════════
#   CẤU HÌNH — dùng chung cho cả 3 model
# ══════════════════════════════════════════════════════════════════════
INCLUDE_ACTIONS = ["purchase", "add_to_cart", "click", "view"]
INCLUDE_SOURCES = ["real_review"]
MIN_INTERACTIONS = 2
TEST_RATIO       = 0.2
TOP_K_LIST       = [5, 10, 20]

# CF hyperparams (lấy từ best_params.json nếu có)
BEST_PARAMS_PATH = BASE_DIR / "best_params.json"
if BEST_PARAMS_PATH.exists():
    with open(BEST_PARAMS_PATH) as f:
        _best = json.load(f)
    CF_FACTORS     = _best.get("factors", 96)
    CF_ALPHA       = _best.get("alpha", 80)
    CF_REG         = _best.get("reg", 0.05)
else:
    CF_FACTORS, CF_ALPHA, CF_REG = 96, 80, 0.05
CF_ITERATIONS = 100

# CF weighting
ACTION_CONFIDENCE = {"purchase": 10.0, "add_to_cart": 5.0, "click": 2.0, "view": 1.0}
DECAY_HALF_LIFE_DAYS = 90
decay_lambda = math.log(2) / DECAY_HALF_LIFE_DAYS

# Hybrid mixing weight: score = CF_WEIGHT × cf + (1-CF_WEIGHT) × cbf
CF_WEIGHT  = 0.2
CBF_WEIGHT = 0.8

# CBF: top-N candidates từ CBF trước khi rank (giới hạn để tăng tốc)
CBF_CANDIDATE_K = max(TOP_K_LIST) * 5   # lấy top-100 từ cosine sim

# ══════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════
def load_data(db):
    log("Load interactions...", indent=1)
    t0 = time.time()
    raw = list(db.interactions.find(
        {"action": {"$in": INCLUDE_ACTIONS}, "source": {"$in": INCLUDE_SOURCES}},
        {"user_id": 1, "product_id": 1, "action": 1, "weight": 1, "timestamp": 1, "_id": 0},
    ))
    log(f"{len(raw):,} interactions  ({time.time()-t0:.1f}s)", C_GREEN, indent=2)

    log("Load products...", indent=1)
    products = list(db.products.find(
        {},
        {"product_id": 1, "name": 1, "category": 1, "brand": 1,
         "tags": 1, "description": 1, "price_range": 1,
         "popularity_score": 1, "_id": 0},
    ))
    log(f"{len(products):,} products", C_GREEN, indent=2)

    # Filter active users
    user_counts = defaultdict(int)
    for r in raw:
        user_counts[r["user_id"]] += 1
    active = {u for u, c in user_counts.items() if c >= MIN_INTERACTIONS}
    filtered = [r for r in raw if r["user_id"] in active]
    log(f"Active users (≥{MIN_INTERACTIONS} interactions): {len(active):,}", indent=2)
    log(f"Interactions sau lọc: {len(filtered):,}", indent=2)

    return filtered, products

# ══════════════════════════════════════════════════════════════════════
# 2. TIME-BASED SPLIT (dùng chung)
# ══════════════════════════════════════════════════════════════════════
def time_based_split(interactions, test_ratio=TEST_RATIO):
    user_ints = defaultdict(list)
    for r in interactions:
        user_ints[r["user_id"]].append(r)

    n_has_ts = sum(1 for r in interactions if isinstance(r.get("timestamp"), datetime))
    use_time = (n_has_ts / max(len(interactions), 1)) >= 0.5
    log(f"Split: {'time-based' if use_time else 'per-user holdout'}  "
        f"({n_has_ts:,}/{len(interactions):,} có timestamp)", C_CYAN, indent=2)

    now_utc = datetime.now(tz=timezone.utc)
    def _ts_key(x):
        ts = x.get("timestamp")
        if not isinstance(ts, datetime):
            return datetime.min.replace(tzinfo=timezone.utc)
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)

    train_ints, test_ints = [], []
    for uid, ints in user_ints.items():
        ints_sorted = sorted(ints, key=_ts_key) if use_time else ints
        cut = max(1, len(ints_sorted) - max(1, round(len(ints_sorted) * test_ratio)))
        train_ints.extend(ints_sorted[:cut])
        test_ints.extend(ints_sorted[cut:])

    # Test = purchase only
    test_ints = [r for r in test_ints if r.get("action") == "purchase"]
    log(f"Train: {len(train_ints):,}  Test (purchase): {len(test_ints):,}", indent=2)
    return train_ints, test_ints

# ══════════════════════════════════════════════════════════════════════
# 3. SHARED EVALUATION FUNCTION
# ══════════════════════════════════════════════════════════════════════
def evaluate(
    get_recs_fn,       # fn(u_idx, n) → list[int] product indices ranked
    filter_train_fn,   # fn(u_idx) → set[int] indices đã biết trong train
    test_ui,           # sp.csr_matrix binary test
    idx2product: dict,
    n_users: int,
    top_k_list=TOP_K_LIST,
    label="model",
) -> dict:
    results = {k: {"precision": 0., "recall": 0., "ndcg": 0., "hit_rate": 0.}
               for k in top_k_list}
    covered = {k: set() for k in top_k_list}
    n_eval  = 0
    max_k   = max(top_k_list)
    t0      = time.time()

    for u_idx in range(n_users):
        test_items = set(test_ui[u_idx].indices)
        if not test_items:
            continue

        known = filter_train_fn(u_idx)
        rec_indices = get_recs_fn(u_idx, max_k + len(known))

        # filter known items, take top max_k
        rec_filtered = [i for i in rec_indices if i not in known][:max_k]

        for k in top_k_list:
            rec_k  = rec_filtered[:k]
            hits   = len(set(rec_k) & test_items)
            prec   = hits / k
            recall = hits / len(test_items)
            hr     = 1.0 if hits > 0 else 0.0
            dcg    = sum(1. / math.log2(r + 1)
                         for r, item in enumerate(rec_k, start=1)
                         if item in test_items)
            idcg   = sum(1. / math.log2(i + 2) for i in range(min(len(test_items), k)))
            ndcg   = dcg / idcg if idcg > 0 else 0.

            results[k]["precision"] += prec
            results[k]["recall"]    += recall
            results[k]["ndcg"]      += ndcg
            results[k]["hit_rate"]  += hr
            covered[k].update(rec_k)

        n_eval += 1

    elapsed = time.time() - t0
    log(f"[{label}] {n_eval:,} users đánh giá  ({elapsed:.1f}s)", indent=2)

    for k in top_k_list:
        if n_eval > 0:
            for m in ("precision", "recall", "ndcg", "hit_rate"):
                results[k][m] /= n_eval
        results[k]["coverage"]  = len(covered[k]) / max(len(idx2product), 1)
        results[k]["composite"] = (
            0.4 * results[k]["precision"]
            + 0.3 * results[k]["recall"]
            + 0.2 * results[k]["ndcg"]
            + 0.1 * results[k]["hit_rate"]
        )
        results[k]["n_eval"] = n_eval

    return results

# ══════════════════════════════════════════════════════════════════════
# 4. PURE CF — ALS
# ══════════════════════════════════════════════════════════════════════
def run_pure_cf(train_ints, test_ints, user2idx, product2idx, idx2product,
                n_users, n_products):
    from implicit.als import AlternatingLeastSquares

    log("", indent=0)
    log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", C_CYAN)
    log("A. PURE CF — Implicit ALS (Matrix Factorization)", C_BOLD)
    log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", C_CYAN)

    now_utc = datetime.now(tz=timezone.utc)

    def _temporal_weight(r):
        ts = r.get("timestamp")
        if not isinstance(ts, datetime):
            return 1.0
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        days = (now_utc - ts).total_seconds() / 86400.
        return math.exp(-decay_lambda * max(0., days))

    # Build train matrix (user_item)
    log("Build confidence matrix...", indent=1)
    pair_w = defaultdict(float)
    pair_n = defaultdict(int)
    for r in train_ints:
        uid = r["user_id"]; pid = r["product_id"]
        if uid not in user2idx or pid not in product2idx:
            continue
        key = (user2idx[uid], product2idx[pid])
        base_w = ACTION_CONFIDENCE.get(r.get("action", ""), 1.0)
        pair_w[key] += base_w * _temporal_weight(r)
        pair_n[key] += 1

    rows, cols, vals = [], [], []
    for (u, p), w in pair_w.items():
        rows.append(u); cols.append(p)
        freq_boost = math.log1p(pair_n[(u, p)])
        conf = 1.0 + CF_ALPHA * (w / pair_n[(u, p)]) * freq_boost
        vals.append(conf)

    train_ui = sp.csr_matrix(
        (np.array(vals, dtype=np.float32),
         (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(n_users, n_products),
    )
    log(f"Matrix: {n_users:,} × {n_products:,}  nnz={train_ui.nnz:,}", indent=1)

    # Purchase-only filter matrix (giống train_model.py)
    purchase_ints = [r for r in train_ints if r.get("action") == "purchase"]
    p_rows, p_cols, p_vals = [], [], []
    for r in purchase_ints:
        uid = r["user_id"]; pid = r["product_id"]
        if uid not in user2idx or pid not in product2idx:
            continue
        p_rows.append(user2idx[uid]); p_cols.append(product2idx[pid]); p_vals.append(1.)
    purchase_filter = sp.csr_matrix(
        (np.array(p_vals, dtype=np.float32),
         (np.array(p_rows, dtype=np.int32), np.array(p_cols, dtype=np.int32))),
        shape=(n_users, n_products),
    )

    # Test matrix
    test_rows, test_cols, test_vals = [], [], []
    for r in test_ints:
        uid = r["user_id"]; pid = r["product_id"]
        if uid not in user2idx or pid not in product2idx:
            continue
        test_rows.append(user2idx[uid]); test_cols.append(product2idx[pid]); test_vals.append(1.)
    test_ui = sp.csr_matrix(
        (np.ones(len(test_rows), dtype=np.float32),
         (np.array(test_rows, dtype=np.int32), np.array(test_cols, dtype=np.int32))),
        shape=(n_users, n_products),
    )

    # Train ALS
    log(f"Train ALS  factors={CF_FACTORS} iter={CF_ITERATIONS} reg={CF_REG} alpha={CF_ALPHA}", indent=1)
    model = AlternatingLeastSquares(
        factors=CF_FACTORS, iterations=CF_ITERATIONS,
        regularization=CF_REG, use_gpu=False,
        calculate_training_loss=True, random_state=42,
    )
    t0 = time.time()
    model.fit(train_ui, show_progress=True)
    log(f"Train done: {time.time()-t0:.1f}s", C_GREEN, indent=1)

    # Evaluate
    log("Evaluate...", indent=1)
    recs_arr = model.recommend(
        np.arange(n_users), purchase_filter, N=max(TOP_K_LIST),
        filter_already_liked_items=True,
    )
    items_arr = recs_arr[0] if isinstance(recs_arr, tuple) else recs_arr

    def get_recs_cf(u_idx, n):
        return items_arr[u_idx].tolist()

    def filter_train_cf(u_idx):
        return set(purchase_filter[u_idx].indices.tolist())

    metrics = evaluate(get_recs_cf, filter_train_cf, test_ui,
                       idx2product, n_users, label="CF")

    return metrics, model, train_ui, test_ui, purchase_filter

# ══════════════════════════════════════════════════════════════════════
# 5. PURE CBF — TF-IDF + Cosine Similarity
# ══════════════════════════════════════════════════════════════════════
def run_pure_cbf(train_ints, test_ints, products_list, user2idx, product2idx, idx2product,
                 n_users, n_products):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    log("", indent=0)
    log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", C_CYAN)
    log("B. PURE CBF — TF-IDF Content Similarity", C_BOLD)
    log("   (Không dùng interaction behavior — chỉ dùng product content)", C_DIM)
    log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", C_CYAN)

    # Build product text corpus
    log("Build TF-IDF corpus...", indent=1)
    pid_to_idx = product2idx  # alias

    def build_text(p: dict) -> str:
        parts = []
        # Tên sản phẩm — quan trọng nhất → lặp 3 lần để boost TF
        name = (p.get("name") or "").strip()
        parts.extend([name] * 3)
        # Category + brand
        parts.append(p.get("category") or "")
        parts.append(p.get("brand") or "")
        parts.append(p.get("price_range") or "")
        # Tags
        tags = p.get("tags") or []
        if isinstance(tags, list):
            parts.append(" ".join(tags))
        # Description (ngắn gọn)
        desc = (p.get("description") or "")[:500]
        parts.append(desc)
        return " ".join(filter(None, parts))

    # Sắp xếp theo product_idx để index khớp
    prod_map = {p["product_id"]: p for p in products_list}
    corpus = [""] * n_products
    for pid, p_idx in product2idx.items():
        p = prod_map.get(pid, {})
        corpus[p_idx] = build_text(p)

    # TF-IDF với n-gram (1,2) để bắt cụm từ
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=30_000,
        sublinear_tf=True,      # log(1+tf) → giảm bias từ lặp nhiều
        min_df=2,               # bỏ từ chỉ xuất hiện 1 lần
        analyzer="word",
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)   # (n_products, vocab)
    tfidf_norm   = normalize(tfidf_matrix, norm="l2") # cosine similarity = dot product
    log(f"TF-IDF matrix: {tfidf_matrix.shape}  vocab={len(vectorizer.vocabulary_):,}", indent=1)

    # Build user profile từ train interactions
    # user_profile[u_idx] = weighted avg TF-IDF vector của các SP đã tương tác
    log("Build user profiles từ train interactions...", indent=1)

    user_profiles = {}       # u_idx → dense vector (vocab,)
    user_train_items = defaultdict(set)  # u_idx → set of p_idx

    for r in train_ints:
        uid = r["user_id"]; pid = r["product_id"]
        if uid not in user2idx or pid not in product2idx:
            continue
        u_idx = user2idx[uid]; p_idx = product2idx[pid]
        weight = ACTION_CONFIDENCE.get(r.get("action", ""), 1.0)
        user_train_items[u_idx].add(p_idx)
        if u_idx not in user_profiles:
            user_profiles[u_idx] = np.zeros(tfidf_norm.shape[1], dtype=np.float32)
        user_profiles[u_idx] += weight * np.asarray(tfidf_norm[p_idx].todense()).flatten()

    # Normalize user profiles
    for u_idx in user_profiles:
        norm = np.linalg.norm(user_profiles[u_idx])
        if norm > 0:
            user_profiles[u_idx] /= norm

    n_profiles = len(user_profiles)
    log(f"User profiles: {n_profiles:,} users có profile", indent=1)

    # Precompute tất cả cosine scores: user_profile @ tfidf_norm.T
    # Batch để tiết kiệm memory
    log("Compute cosine similarities (batch)...", indent=1)
    t0 = time.time()
    BATCH = 500
    cbf_scores = np.full((n_users, n_products), -1., dtype=np.float32)

    u_indices = sorted(user_profiles.keys())
    for i in range(0, len(u_indices), BATCH):
        batch_u = u_indices[i:i+BATCH]
        # Stack profiles → (batch, vocab)
        profile_mat = np.stack([user_profiles[u] for u in batch_u])
        # (batch, n_products) = (batch, vocab) @ (vocab, n_products)
        scores = profile_mat @ tfidf_norm.T.toarray()
        for j, u_idx in enumerate(batch_u):
            cbf_scores[u_idx] = scores[j]

    log(f"Cosine sim computed: {time.time()-t0:.1f}s", C_GREEN, indent=1)

    # Test matrix
    test_rows, test_cols = [], []
    for r in test_ints:
        uid = r["user_id"]; pid = r["product_id"]
        if uid not in user2idx or pid not in product2idx:
            continue
        test_rows.append(user2idx[uid]); test_cols.append(product2idx[pid])
    test_ui = sp.csr_matrix(
        (np.ones(len(test_rows), dtype=np.float32),
         (np.array(test_rows, dtype=np.int32), np.array(test_cols, dtype=np.int32))),
        shape=(n_users, n_products),
    )

    # Purchase-only train filter (để tránh gợi ý SP đã mua)
    purchase_ints = [r for r in train_ints if r.get("action") == "purchase"]
    p_rows, p_cols = [], []
    for r in purchase_ints:
        uid = r["user_id"]; pid = r["product_id"]
        if uid not in user2idx or pid not in product2idx:
            continue
        p_rows.append(user2idx[uid]); p_cols.append(product2idx[pid])
    purchase_filter_cbf = sp.csr_matrix(
        (np.ones(len(p_rows), dtype=np.float32),
         (np.array(p_rows, dtype=np.int32), np.array(p_cols, dtype=np.int32))),
        shape=(n_users, n_products),
    )

    log("Evaluate...", indent=1)

    def get_recs_cbf(u_idx, n):
        if u_idx not in user_profiles:
            # Cold user: trả về top-n theo popularity (fallback)
            return list(range(min(n, n_products)))
        scores = cbf_scores[u_idx]
        return np.argsort(-scores)[:n].tolist()

    def filter_train_cbf(u_idx):
        return set(purchase_filter_cbf[u_idx].indices.tolist())

    metrics = evaluate(get_recs_cbf, filter_train_cbf, test_ui,
                       idx2product, n_users, label="CBF")

    return metrics, cbf_scores, test_ui, purchase_filter_cbf

# ══════════════════════════════════════════════════════════════════════
# 6. HYBRID — CF score + CBF score (weighted combination)
# ══════════════════════════════════════════════════════════════════════
def run_hybrid(
    cf_model, train_ui_cf, purchase_filter_cf,
    cbf_scores, purchase_filter_cbf,
    test_ui, user2idx, product2idx, idx2product,
    n_users, n_products,
):
    log("", indent=0)
    log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", C_CYAN)
    log("C. HYBRID — CF score + CBF score", C_BOLD)
    log(f"   score = {CF_WEIGHT}×CF + {CBF_WEIGHT}×CBF  (cả hai normalize về [0,1])", C_DIM)
    log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", C_CYAN)

    # Precompute CF scores cho tất cả users
    log("Compute CF scores (batch)...", indent=1)
    t0 = time.time()
    user_factors   = cf_model.user_factors    # (n_users, factors)
    item_factors   = cf_model.item_factors    # (n_products, factors)
    # CF score matrix: (n_users, n_products) = user_factors @ item_factors.T
    BATCH = 1000
    cf_scores = np.zeros((n_users, n_products), dtype=np.float32)
    for i in range(0, n_users, BATCH):
        cf_scores[i:i+BATCH] = user_factors[i:i+BATCH] @ item_factors.T
    log(f"CF scores computed: {time.time()-t0:.1f}s", C_GREEN, indent=1)

    # Normalize CF và CBF scores per-user về [0,1]
    log("Normalize & combine scores...", indent=1)
    hybrid_scores = np.zeros((n_users, n_products), dtype=np.float32)

    for u_idx in range(n_users):
        cf_row  = cf_scores[u_idx]
        cbf_row = cbf_scores[u_idx]

        # Normalize CF
        cf_min, cf_max = cf_row.min(), cf_row.max()
        cf_rng = cf_max - cf_min
        cf_norm = (cf_row - cf_min) / cf_rng if cf_rng > 1e-9 else np.zeros_like(cf_row)

        # Normalize CBF (cbf_scores=-1 nếu không có profile → gán 0)
        cbf_valid = cbf_row.copy()
        cbf_valid[cbf_valid < 0] = 0.
        cbf_min, cbf_max = cbf_valid.min(), cbf_valid.max()
        cbf_rng = cbf_max - cbf_min
        cbf_norm = (cbf_valid - cbf_min) / cbf_rng if cbf_rng > 1e-9 else np.zeros_like(cbf_valid)

        hybrid_scores[u_idx] = CF_WEIGHT * cf_norm + CBF_WEIGHT * cbf_norm

    # Combined purchase filter (union)
    combined_filter = (purchase_filter_cf + purchase_filter_cbf)
    combined_filter.data = np.ones_like(combined_filter.data)

    log("Evaluate...", indent=1)

    def get_recs_hybrid(u_idx, n):
        return np.argsort(-hybrid_scores[u_idx])[:n].tolist()

    def filter_train_hybrid(u_idx):
        return set(combined_filter[u_idx].indices.tolist())

    metrics = evaluate(get_recs_hybrid, filter_train_hybrid, test_ui,
                       idx2product, n_users, label="Hybrid")

    return metrics

# ══════════════════════════════════════════════════════════════════════
# 7. PRINT COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════
def print_comparison(results: dict[str, dict], top_k_list=TOP_K_LIST):
    print()
    print(f"{C_BOLD}{C_CYAN}{'='*80}{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}  KẾT QUẢ SO SÁNH — Pure CF vs Pure CBF vs Hybrid{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}{'='*80}{C_RESET}")
    print()

    models   = list(results.keys())
    metrics  = ["precision", "recall", "ndcg", "hit_rate", "coverage", "composite"]
    m_labels = ["Precision", "Recall", "NDCG", "HitRate", "Coverage", "Composite"]

    for k in top_k_list:
        print(f"  {'─'*78}")
        print(f"  @K = {k}")
        print(f"  {'Metric':<14}", end="")
        for m in models:
            print(f"  {m:>14}", end="")
        print()
        print(f"  {'─'*78}")
        for metric, label in zip(metrics, m_labels):
            print(f"  {label:<14}", end="")
            vals = [results[m][k][metric] for m in models]
            best = max(vals)
            for i, (m, v) in enumerate(zip(models, vals)):
                v_str = f"{v:.1%}" if metric == "coverage" else f"{v:.4f}"
                if abs(v - best) < 1e-6:
                    print(f"  {C_GREEN}{v_str:>14}{C_RESET}", end="")
                else:
                    print(f"  {v_str:>14}", end="")
            print()
        print()

    # Composite winner table
    print(f"  {'─'*78}")
    print(f"  {'Composite@K':<14}", end="")
    for m in models:
        print(f"  {m:>14}", end="")
    print()
    print(f"  {'─'*38}")
    for k in top_k_list:
        vals = {m: results[m][k]["composite"] for m in models}
        winner = max(vals, key=vals.get)
        print(f"  @{k:<13}", end="")
        for m in models:
            v = vals[m]
            tag = f" ✓" if m == winner else ""
            v_str = f"{v:.4f}{tag}"
            color = C_GREEN if m == winner else ""
            print(f"  {color}{v_str:>14}{C_RESET}", end="")
        print()

    # Conclusion
    print()
    print(f"  {'═'*78}")
    print(f"  KẾT LUẬN:")

    # Rank by composite@5
    comp5 = {m: results[m][5]["composite"] for m in models}
    ranked = sorted(comp5.items(), key=lambda x: -x[1])
    for rank, (m, v) in enumerate(ranked, 1):
        medal = ["🥇", "🥈", "🥉"][rank - 1]
        print(f"  {medal} {rank}. {m:<12}  Composite@5 = {v:.4f}", end="")

        if rank == 1:
            # Explain why
            r = results[m][5]
            print(f"  ← Precision={r['precision']:.4f}  Recall={r['recall']:.4f}  "
                  f"NDCG={r['ndcg']:.4f}  Coverage={r['coverage']:.1%}")
        else:
            diff = comp5[ranked[0][0]] - v
            print(f"  (−{diff:.4f} vs top)")

    print()

    # Auto-recommendation
    best_model, best_score = ranked[0]
    cbf_score = comp5.get("Pure CBF", 0)
    cf_score  = comp5.get("Pure CF", 0)
    hy_score  = comp5.get("Hybrid", 0)

    print(f"   Khuyến nghị:")
    if best_model == "Hybrid":
        print(f"     Hybrid tốt nhất → tiếp tục dùng cả CF lẫn CBF.")
        if abs(cf_score - cbf_score) < 0.01:
            print(f"     CF và CBF gần bằng nhau → Hybrid được lợi từ cả hai signal.")
        elif cf_score > cbf_score:
            print(f"     CF mạnh hơn CBF → tăng CF_WEIGHT (hiện {CF_WEIGHT}) để cải thiện thêm.")
        else:
            print(f"     CBF mạnh hơn CF → tăng CBF_WEIGHT (hiện {CBF_WEIGHT}) để cải thiện thêm.")
    elif best_model == "Pure CF":
        print(f"     Pure CF tốt nhất → content features chưa thêm giá trị với dataset này.")
        print(f"     Lý do thường gặp: data thưa, user history ít, content text chưa đủ chất lượng.")
        print(f"     Giải pháp: cải thiện CBF bằng sentence-transformer (PhoBERT) thay TF-IDF.")
    else:
        print(f"     Pure CBF tốt nhất → behavior data quá thưa, content signal mạnh hơn.")
        print(f"     Lý do thường gặp: cold-start nhiều, ít interaction per user.")

    print(f"  {'═'*78}")

def save_results(results: dict, output_path: Path):
    def _to_serializable(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    out = {}
    for model, ks in results.items():
        out[model] = {}
        for k, m in ks.items():
            out[model][str(k)] = {mk: float(mv) for mk, mv in m.items()}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"results": out, "config": {
            "cf_weight": CF_WEIGHT, "cbf_weight": CBF_WEIGHT,
            "cf_factors": CF_FACTORS, "cf_alpha": CF_ALPHA,
            "test_ratio": TEST_RATIO, "top_k": TOP_K_LIST,
        }}, f, ensure_ascii=False, indent=2, default=_to_serializable)

    log(f"Results saved → {output_path}", C_GREEN)

# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n{C_BOLD}{C_MAG}{'='*60}{C_RESET}")
    print(f"{C_BOLD}{C_MAG}  MODEL COMPARISON: CF vs CBF vs Hybrid{C_RESET}")
    print(f"{C_BOLD}{C_MAG}{'='*60}{C_RESET}\n")

    # Check deps
    missing = []
    try:
        from implicit.als import AlternatingLeastSquares
    except ImportError:
        missing.append("implicit")
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        missing.append("scikit-learn")
    if missing:
        print(f"{C_RED} Thiếu thư viện: {', '.join(missing)}{C_RESET}")
        print(f"   Chạy: pip install {' '.join(missing)}")
        sys.exit(1)

    # Connect MongoDB
    try:
        from db.connection import get_db
        db = get_db()
        log(" MongoDB connected", C_GREEN)
    except Exception as e:
        log(f" MongoDB error: {e}", C_RED)
        sys.exit(1)

    # ── Load data ──────────────────────────────────────────────────────
    log("", indent=0)
    log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", C_CYAN)
    log("0. LOAD & SPLIT DATA (dùng chung cho cả 3 model)", C_BOLD)
    log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", C_CYAN)

    interactions, products_list = load_data(db)

    all_user_ids    = sorted({r["user_id"]    for r in interactions})
    all_product_ids = sorted({r["product_id"] for r in interactions})
    user2idx    = {u: i for i, u in enumerate(all_user_ids)}
    product2idx = {p: i for i, p in enumerate(all_product_ids)}
    idx2user    = {i: u for u, i in user2idx.items()}
    idx2product = {i: p for p, i in product2idx.items()}
    n_users     = len(all_user_ids)
    n_products  = len(all_product_ids)
    log(f"Users: {n_users:,}  Products: {n_products:,}", indent=1)

    train_ints, test_ints = time_based_split(interactions)

    all_results = {}

    # ── A. Pure CF ─────────────────────────────────────────────────────
    cf_metrics, cf_model, train_ui_cf, test_ui_cf, purchase_filter_cf = run_pure_cf(
        train_ints, test_ints, user2idx, product2idx, idx2product, n_users, n_products
    )
    all_results["Pure CF"] = cf_metrics

    # ── B. Pure CBF ────────────────────────────────────────────────────
    cbf_metrics, cbf_scores, test_ui_cbf, purchase_filter_cbf = run_pure_cbf(
        train_ints, test_ints, products_list, user2idx, product2idx,
        idx2product, n_users, n_products,
    )
    all_results["Pure CBF"] = cbf_metrics

    # ── C. Hybrid ──────────────────────────────────────────────────────
    # Dùng cùng test_ui (chọn cf version vì chạy trước)
    hybrid_metrics = run_hybrid(
        cf_model, train_ui_cf, purchase_filter_cf,
        cbf_scores, purchase_filter_cbf,
        test_ui_cf,   # same test set
        user2idx, product2idx, idx2product, n_users, n_products,
    )
    all_results["Hybrid"] = hybrid_metrics

    # ── In kết quả ────────────────────────────────────────────────────
    print_comparison(all_results)

    # ── Lưu JSON ──────────────────────────────────────────────────────
    save_results(all_results, BASE_DIR / "comparison_results.json")
    print(f"\n   Chạy xong. Xem chi tiết tại comparison_results.json\n")
