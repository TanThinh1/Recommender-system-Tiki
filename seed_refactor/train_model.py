from __future__ import annotations

import os
import sys
import json
import time
import math
import pickle
import random
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

warnings.filterwarnings("ignore")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# ── Console colors ───────────────────────────────────────────────────
from utils.console import log, C_RESET, C_GREEN, C_RED, C_CYAN, C_BOLD, C_DIM, C_YEL

# ══════════════════════════════════════════════════════════════════════
# ⚙️  CẤU HÌNH
# ══════════════════════════════════════════════════════════════════════
# Mở rộng training signal ngoài purchase
# Confidence = mức độ tin cậy cho mỗi loại action
# Thứ tự: purchase > add_to_cart > click > view
# ALS dùng confidence để biết action nào đáng tin hơn khi factorize
ACTION_CONFIDENCE: dict[str, float] = {
    "purchase"    : 10.0,   # tín hiệu mạnh nhất — user thực sự muốn SP này
    "add_to_cart" :  5.0,   # có ý định mua nhưng chưa hoàn tất
    "click"       :  2.0,   # quan tâm nhưng chưa chắc
    "view"        :  1.0,   # tín hiệu yếu nhất — có thể click nhầm
}
INCLUDE_ACTIONS    = list(ACTION_CONFIDENCE.keys())   # ["purchase","add_to_cart","click","view"]
INCLUDE_SOURCES    = ["real_review"]                  # vẫn giữ filter source để tránh bot traffic

MIN_INTERACTIONS   = 2                     # loại user chỉ 1 tương tác
USE_TEMPORAL_DECAY = True
DECAY_HALF_LIFE_DAYS = 90
USE_IDF            = True
USE_LOG_FREQ       = True

TEST_RATIO         = 0.2
TOP_K_LIST         = [5, 10, 20]
ZERO_SCORE_THRESHOLD = 0.05

BASE_DIR         = Path(__file__).parent
MODEL_PATH       = BASE_DIR / "als_model.pkl"
META_PATH        = BASE_DIR / "als_meta.json"
BEST_PARAMS_PATH = BASE_DIR / "best_params.json"

# Đọc best params
if BEST_PARAMS_PATH.exists():
    with open(BEST_PARAMS_PATH) as f:
        best = json.load(f)
    log(f" Load best params từ {BEST_PARAMS_PATH.name}: {best}", C_CYAN)
    ALPHA          = best.get("alpha", 120)
    FACTORS        = best.get("factors", 64)
    REGULARIZATION = best.get("reg", 0.1)
    ITERATIONS     = 100
else:
    log(" Không tìm thấy best_params.json. Chạy grid_search.py trước.", C_RED)
    sys.exit(1)

# ── DB ───────────────────────────────────────────────────────────────
# Thêm cả thư mục hiện tại (seed_refactor/) LẪN thư mục cha vào sys.path
# để xử lý cả import relative (from db.connection …)
# lẫn import absolute (from seed_refactor.db.connection …)
for _p in (BASE_DIR, BASE_DIR.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
try:
    from db.connection import get_db
    db = get_db()
    log(" Kết nối MongoDB thành công", C_GREEN)
except Exception as e:
    log(f" Không kết nối được MongoDB: {e}", C_RED)
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════
print(f"\n{C_BOLD}{C_CYAN}{'='*60}{C_RESET}")
print(f"{C_BOLD}{C_CYAN}  ALS RECOMMENDATION MODEL v5 — Time-based Split{C_RESET}")
print(f"{C_BOLD}{C_CYAN}{'='*60}{C_RESET}\n")

# ══════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════
log("1. Load data", C_BOLD)
log(f"  source={INCLUDE_SOURCES}  action={INCLUDE_ACTIONS}  min_inter={MIN_INTERACTIONS}", indent=1)
t0 = time.time()

query = {
    "action": {"$in": INCLUDE_ACTIONS},
    "source": {"$in": INCLUDE_SOURCES},
}
raw_interactions = list(db.interactions.find(
    query,
    {"user_id": 1, "product_id": 1, "weight": 1, "action": 1, "source": 1, "timestamp": 1, "_id": 0}
))
log(f"{len(raw_interactions):,} interactions  ({time.time()-t0:.1f}s)", C_GREEN, indent=2)

if not raw_interactions:
    log(" Không có interaction nào.", C_RED)
    sys.exit(1)

log("Load products...", indent=1)
products = list(db.products.find(
    {},
    {"product_id": 1, "popularity_score": 1, "name": 1, "category": 1, "_id": 0}
))
product_pop  = {p["product_id"]: float(p.get("popularity_score") or 0) for p in products}
product_name = {p["product_id"]: p.get("name", "")[:40] for p in products}
product_cat  = {p["product_id"]: p.get("category", "") for p in products}
log(f"{len(products):,} products", C_GREEN, indent=2)

# ══════════════════════════════════════════════════════════════════════
# 2. FILTER & MAPPINGS
# ══════════════════════════════════════════════════════════════════════
log("2. Filter & build mappings", C_BOLD)

user_counts = defaultdict(int)
for r in raw_interactions:
    user_counts[r["user_id"]] += 1

active_users = {u for u, c in user_counts.items() if c >= MIN_INTERACTIONS}
log(f"Users với ≥{MIN_INTERACTIONS} interactions: {len(active_users):,}", indent=2)
log(f"  (Loại bỏ {len(user_counts)-len(active_users):,} cold users)", C_DIM, indent=2)

filtered = [r for r in raw_interactions if r["user_id"] in active_users]
log(f"Interactions sau lọc: {len(filtered):,}", indent=2)

all_user_ids    = sorted({r["user_id"]    for r in filtered})
all_product_ids = sorted({r["product_id"] for r in filtered})
user2idx    = {u: i for i, u in enumerate(all_user_ids)}
product2idx = {p: i for i, p in enumerate(all_product_ids)}
idx2user    = {i: u for u, i in user2idx.items()}
idx2product = {i: p for p, i in product2idx.items()}

n_users    = len(all_user_ids)
n_products = len(all_product_ids)
log(f"Matrix size: {n_users:,} users × {n_products:,} products", indent=2)

# ══════════════════════════════════════════════════════════════════════
# 3. HELPER: Temporal decay
# ══════════════════════════════════════════════════════════════════════
now_ts       = datetime.now(tz=timezone.utc)
decay_lambda = math.log(2) / DECAY_HALF_LIFE_DAYS  # λ = ln2 / half_life

def get_temporal_weight(r: dict) -> float:
    """Tính temporal decay: e^(-λ * days_ago)"""
    if not USE_TEMPORAL_DECAY:
        return 1.0
    ts = r.get("timestamp")
    if ts is None:
        return 1.0
    try:
        if isinstance(ts, datetime):
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        else:
            return 1.0
        days_ago = (now_ts - ts).total_seconds() / 86400.0
        return math.exp(-decay_lambda * max(0, days_ago))
    except Exception:
        return 1.0

# ══════════════════════════════════════════════════════════════════════
# 4. [v5] TIME-BASED SPLIT
# ══════════════════════════════════════════════════════════════════════
log("4. Time-based split", C_BOLD)

def time_based_split(
    interactions: list[dict],
    test_ratio: float = 0.2,
) -> tuple[list[dict], list[dict]]:
    user_ints: dict[str, list] = defaultdict(list)
    for r in interactions:
        user_ints[r["user_id"]].append(r)

    # Kiểm tra tỉ lệ có timestamp
    n_total  = len(interactions)
    n_has_ts = sum(
        1 for r in interactions
        if isinstance(r.get("timestamp"), datetime)
    )
    use_time = (n_has_ts / max(n_total, 1)) >= 0.5

    if use_time:
        log(f"→ Time-based split  ({n_has_ts:,}/{n_total:,} có timestamp)", C_CYAN, indent=2)
    else:
        log(
            f"  Chỉ {n_has_ts:,}/{n_total:,} interactions có timestamp "
            f"→ fallback: per-user holdout (last {TEST_RATIO:.0%} làm test)",
            C_YEL, indent=2,
        )

    train_ints: list[dict] = []
    raw_test_ints: list[dict] = []

    for uid, ints in user_ints.items():
        n = len(ints)

        if use_time:
            def _ts_key(x):
                ts = x.get("timestamp")
                if not isinstance(ts, datetime):
                    return datetime.min.replace(tzinfo=timezone.utc)
                return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            ints_sorted = sorted(ints, key=_ts_key)
        else:
            ints_sorted = ints

        cut = max(1, n - max(1, round(n * test_ratio)))
        train_ints.extend(ints_sorted[:cut])
        raw_test_ints.extend(ints_sorted[cut:])
    test_ints = [r for r in raw_test_ints if r.get("action") == "purchase"]

    n_non_purchase = len(raw_test_ints) - len(test_ints)
    log(
        f"Test ground truth: {len(raw_test_ints):,} raw → {len(test_ints):,} purchase-only "
        f"(bỏ {n_non_purchase:,} non-purchase actions)",
        C_CYAN, indent=2,
    )
    if not test_ints:
        log(
            "⚠️  Test set rỗng — không có purchase nào trong 20% cuối. "
            "Metrics sẽ không khả dụng.",
            C_YEL, indent=2,
        )

    return train_ints, test_ints

train_ints, test_ints = time_based_split(filtered, test_ratio=TEST_RATIO)

# Tóm tắt phân chia timestamp
ts_values = sorted(
    r["timestamp"].replace(tzinfo=timezone.utc) if r.get("timestamp") and
    isinstance(r["timestamp"], datetime) and not r["timestamp"].tzinfo
    else r["timestamp"]
    for r in filtered if isinstance(r.get("timestamp"), datetime)
)
if ts_values:
    log(f"Timestamp range: {ts_values[0].date()} → {ts_values[-1].date()}", C_DIM, indent=2)

n_ts_train = sum(1 for r in train_ints if isinstance(r.get("timestamp"), datetime))
n_ts_test  = sum(1 for r in test_ints  if isinstance(r.get("timestamp"), datetime))
n_no_ts    = sum(1 for r in train_ints if not isinstance(r.get("timestamp"), datetime))

log(f"Train interactions: {len(train_ints):,}  ({n_ts_train:,} có timestamp)", C_GREEN, indent=2)
log(f"Test  interactions: {len(test_ints):,}   ({n_ts_test:,} có timestamp)", C_GREEN, indent=2)
if n_no_ts:
    log(f"No-timestamp (→ train): {n_no_ts:,}", C_DIM, indent=2)

# ══════════════════════════════════════════════════════════════════════
# 5. BUILD CONFIDENCE MATRICES
# ══════════════════════════════════════════════════════════════════════
log("5. Build confidence matrices (v5 weighting + time-based)", C_BOLD)

def build_ui_matrix(
    interactions : list[dict],
    user2idx     : dict,
    product2idx  : dict,
    n_users      : int,
    n_products   : int,
    alpha        : float,
    binary       : bool = False,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """
    Xây dựng user_item và item_user confidence matrix từ danh sách interactions.

    Parameters
    ----------
    interactions : danh sách raw interactions (dict với user_id, product_id, weight, timestamp)
    binary       : True → tất cả confidence = 1.0 (dùng cho test matrix)
                   False → áp dụng đầy đủ IDF + LogFreq + Temporal decay (cho train)

    Returns
    -------
    user_item : sp.csr_matrix (n_users, n_products)
    item_user : sp.csr_matrix (n_products, n_users)
    """
    pair_count      : defaultdict[tuple, int]   = defaultdict(int)
    pair_weight_sum : defaultdict[tuple, float] = defaultdict(float)

    for rec in interactions:
        uid = rec["user_id"]
        pid = rec["product_id"]
        if uid not in user2idx or pid not in product2idx:
            continue
        key = (user2idx[uid], product2idx[pid])
        # (5) Dùng ACTION_CONFIDENCE thay vì weight field cố định
        # Purchase=10, add_to_cart=5, click=2, view=1
        # Nếu action không có trong dict → fallback weight=1.0
        base_w = ACTION_CONFIDENCE.get(rec.get("action", ""), float(rec.get("weight") or 1.0))
        decay  = get_temporal_weight(rec)
        pair_count[key]      += 1
        pair_weight_sum[key] += base_w * decay

    # ── IDF (dựa trên tập interactions này, không leak từ test) ───────
    item_u_count: defaultdict[int, set] = defaultdict(set)
    for (u_idx, p_idx), _ in pair_count.items():
        item_u_count[p_idx].add(u_idx)

    def _idf_raw(pid_idx: int) -> float:
        if not USE_IDF:
            return 1.0
        df = len(item_u_count.get(pid_idx, set()))
        return math.log((n_users + 1) / (df + 1)) + 1.0

    if USE_IDF and not binary:
        idf_vals = [_idf_raw(p) for p in range(n_products)]
        idf_min  = min(idf_vals); idf_max = max(idf_vals)
        idf_rng  = idf_max - idf_min if idf_max > idf_min else 1.0
        def _idf_norm(pid_idx: int) -> float:
            return 0.5 + (_idf_raw(pid_idx) - idf_min) / idf_rng  # [0.5, 1.5]
    else:
        def _idf_norm(pid_idx: int) -> float:
            return 1.0

    # ── Build COO arrays ───────────────────────────────────────────────
    rows_arr, cols_arr, conf_arr = [], [], []

    for (u_idx, p_idx), count in pair_count.items():
        if binary:
            confidence = 1.0
        else:
            w_sum = pair_weight_sum[(u_idx, p_idx)]
            if USE_LOG_FREQ:
                freq_boost = math.log1p(count)
                w_final    = (w_sum / count) * freq_boost
            else:
                w_final = min(w_sum, 1.0)
            idf_scale  = _idf_norm(p_idx)
            confidence = 1.0 + alpha * (w_final * idf_scale)

        rows_arr.append(u_idx)
        cols_arr.append(p_idx)
        conf_arr.append(confidence)

    rows_np = np.array(rows_arr, dtype=np.int32)
    cols_np = np.array(cols_arr, dtype=np.int32)
    conf_np = np.array(conf_arr, dtype=np.float32)

    item_user = sp.csr_matrix(
        (conf_np, (cols_np, rows_np)),
        shape=(n_products, n_users),
        dtype=np.float32,
    )
    user_item = item_user.T.tocsr()
    return user_item, item_user

# Ma trận train (đầy đủ weighting, dùng để train ALS + evaluate known items)
log("  Build train matrix...", indent=1)
train_user_item, train_item_user = build_ui_matrix(
    train_ints, user2idx, product2idx, n_users, n_products, ALPHA
)

# Ma trận test (binary, ground truth cho evaluation)
log("  Build test matrix (binary)...", indent=1)
test_user_item, _ = build_ui_matrix(
    test_ints, user2idx, product2idx, n_users, n_products, ALPHA, binary=True
)

# Ma trận purchase-only trong train — dùng để filter khi evaluate
# Chỉ filter những SP user đã PURCHASE (không filter view/click)
# → model.recommend() vẫn có thể gợi ý SP user đã view/click nhưng chưa mua
log("  Build purchase-filter matrix (purchase-only train)...", indent=1)
purchase_train_ints = [r for r in train_ints if r.get("action") == "purchase"]
purchase_filter_ui, _ = build_ui_matrix(
    purchase_train_ints, user2idx, product2idx, n_users, n_products, ALPHA, binary=True
)

# Ma trận đầy đủ (ALL interactions) — dùng để lưu model, sanity check
log("  Build full matrix (all interactions)...", indent=1)
full_user_item, full_item_user = build_ui_matrix(
    filtered, user2idx, product2idx, n_users, n_products, ALPHA
)
user_item = full_user_item   # alias để tương thích với code phía dưới

nnz      = train_item_user.nnz
sparsity = 1 - nnz / (n_products * n_users)
log(f"Train nnz={nnz:,}  sparsity={sparsity*100:.2f}%", indent=2)
log(f"Test  nnz={test_user_item.nnz:,}  purchase_filter nnz={purchase_filter_ui.nnz:,}", indent=2)
log(f"Full  nnz={full_item_user.nnz:,}", indent=2)

# ══════════════════════════════════════════════════════════════════════
# 6. TRAIN ALS (trên train matrix)
# ══════════════════════════════════════════════════════════════════════
log("6. Train ALS (trên train split)", C_BOLD)
log(f"factors={FACTORS}  iterations={ITERATIONS}  reg={REGULARIZATION}  alpha={ALPHA}", indent=2)

model = AlternatingLeastSquares(
    factors=FACTORS,
    iterations=ITERATIONS,
    regularization=REGULARIZATION,
    use_gpu=False,
    calculate_training_loss=True,
    random_state=42,
)

log("Đang huấn luyện...", indent=2)
t0 = time.time()
#  implicit >= 0.5: fit() nhận user_items (n_users × n_items)
#  trước đó truyền train_item_user (1138×3303) → user_factors size=1138 → IndexError
model.fit(train_user_item, show_progress=True)
elapsed = time.time() - t0
log(f"Hoàn tất: {elapsed:.1f}s", C_GREEN, indent=2)

# ══════════════════════════════════════════════════════════════════════
# 7. EVALUATE (time-based test set)
# ══════════════════════════════════════════════════════════════════════
log("7. Evaluate (time-based holdout)", C_BOLD)

def evaluate_model(model, purchase_filter_ui, test_ui, n_users, idx2product, top_k_list):
    """
    Đánh giá model với time-based holdout:
      - purchase_filter_ui : ma trận purchase-only trong train
                             Chỉ filter SP đã purchase (không filter view/click)
                             → model có thể gợi ý SP user đã view/click nhưng chưa mua
      - test_ui            : purchase ground truth từ test set
    """
    results = {k: {"precision": 0, "recall": 0, "ndcg": 0, "hit_rate": 0}
               for k in top_k_list}
    covered = {k: set() for k in top_k_list}
    n_eval  = 0
    max_k   = max(top_k_list)

    t0 = time.time()
    log(f"Đánh giá {n_users:,} users (batch)...", indent=2)

    recs = model.recommend(
        np.arange(n_users),
        purchase_filter_ui,              # chỉ filter SP đã purchase (không filter view/click)
        N=max_k,
        filter_already_liked_items=True,
    )
    items_arr = recs[0] if isinstance(recs, tuple) else recs

    for u_idx in range(n_users):
        test_items = set(test_ui[u_idx].indices)   # ground truth = new interactions
        if not test_items:
            continue                                # user không có test data → bỏ qua

        rec_ids_raw = items_arr[u_idx]
        for k in top_k_list:
            rec_items = set(rec_ids_raw[:k].tolist())
            hits      = len(rec_items & test_items)
            prec      = hits / k
            recall    = hits / len(test_items)
            hit_rate  = 1.0 if hits > 0 else 0.0

            dcg  = sum(1.0 / math.log2(rank + 1)
                       for rank, item in enumerate(rec_ids_raw[:k].tolist(), start=1)
                       if item in test_items)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(test_items), k)))
            ndcg = dcg / idcg if idcg > 0 else 0.0

            results[k]["precision"] += prec
            results[k]["recall"]    += recall
            results[k]["ndcg"]      += ndcg
            results[k]["hit_rate"]  += hit_rate
            covered[k].update(rec_items)

        n_eval += 1

    elapsed = time.time() - t0
    log(f"Đánh giá {n_eval:,} users có test items  ({elapsed:.1f}s)", indent=2)

    for k in top_k_list:
        if n_eval > 0:
            results[k]["precision"] /= n_eval
            results[k]["recall"]    /= n_eval
            results[k]["ndcg"]      /= n_eval
            results[k]["hit_rate"]  /= n_eval
        results[k]["coverage"]  = len(covered[k]) / len(idx2product) if idx2product else 0
        results[k]["composite"] = (
            0.4 * results[k]["precision"]
            + 0.3 * results[k]["recall"]
            + 0.2 * results[k]["ndcg"]
            + 0.1 * results[k]["hit_rate"]
        )

    return results, n_eval

metrics, n_eval = evaluate_model(
    model, purchase_filter_ui, test_user_item, n_users, idx2product, TOP_K_LIST
)

print()
print(f"  {'K':<6} {'Precision':>10} {'Recall':>10} {'NDCG':>10} {'HitRate':>10} {'Coverage':>10} {'Composite':>10}")
print(f"  {'─'*66}")
for k in TOP_K_LIST:
    m = metrics[k]
    print(f"  @{k:<5} {m['precision']:>10.4f} {m['recall']:>10.4f} "
          f"{m['ndcg']:>10.4f} {m['hit_rate']:>10.4f} "
          f"{m['coverage']:>9.1%} {m['composite']:>10.4f}")
print()

p5         = metrics[5]["precision"]
composite5 = metrics[5]["composite"]
if p5 >= 0.05:
    grade = f"{C_GREEN} TỐT"
elif p5 >= 0.02:
    grade = f"{C_YEL}  KHÁ"
else:
    grade = f"{C_RED} THẤP"
log(f"precision@5 = {p5:.4f}  composite@5 = {composite5:.4f}  {grade}{C_RESET}", indent=1)

# ══════════════════════════════════════════════════════════════════════
# 8. POPULARITY FALLBACK (cold-start)
# ══════════════════════════════════════════════════════════════════════
log("8. Build popularity fallback (cold-start)", C_BOLD)

purchase_count = defaultdict(int)
for rec in filtered:
    # Popularity fallback vẫn chỉ đếm purchase để tránh bias từ view/click
    if rec.get("action") == "purchase":
        purchase_count[rec["product_id"]] += 1

max_pc  = max(purchase_count.values()) if purchase_count else 1
max_pop = max(product_pop.values())    if product_pop    else 1

combined_score = {}
for pid in product_pop:
    pc  = purchase_count.get(pid, 0)
    pop = product_pop.get(pid, 0)
    combined_score[pid] = 0.7 * (pc / max_pc) + 0.3 * (pop / max_pop if max_pop > 0 else 0)

popular_items = sorted(combined_score.items(), key=lambda x: x[1], reverse=True)[:100]

# Dedup giữ thứ tự (combined_score.items() đã unique, nhưng giữ lại để an toàn)
seen = set()
popular_items = [(p, s) for p, s in popular_items if not (p in seen or seen.add(p))]
log(f"Top-100 fallback items sẵn sàng (format: [(pid, score)])", C_GREEN, indent=2)

# ══════════════════════════════════════════════════════════════════════
# 9. LƯU MODEL
# ══════════════════════════════════════════════════════════════════════
log("9. Lưu model", C_BOLD)

weighting_config = {
    "use_log_freq":         USE_LOG_FREQ,
    "use_idf":              USE_IDF,
    "use_temporal_decay":   USE_TEMPORAL_DECAY,
    "decay_half_life_days": DECAY_HALF_LIFE_DAYS,
    "min_interactions":     MIN_INTERACTIONS,
    "zero_score_threshold": ZERO_SCORE_THRESHOLD,
    "split_strategy":       "time_based",
    "test_ratio":           TEST_RATIO,
    "action_confidence":    ACTION_CONFIDENCE,   # (5) ghi rõ confidence theo action
}

model_data = {
    "model":            model,
    "user2idx":         user2idx,
    "product2idx":      product2idx,
    "idx2user":         idx2user,
    "idx2product":      idx2product,
    #  Lưu full_user_item (toàn bộ dữ liệu) để inference trong production
    "user_item":        full_user_item,
    "popular_items":    popular_items,
    "weighting_config": weighting_config,
    "product_name":     product_name,
    "product_cat":      product_cat,
}

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model_data, f, protocol=4)
log(f"Model  → {MODEL_PATH}", C_GREEN, indent=2)

meta = {
    "trained_at":      datetime.now().isoformat(),
    "model_version":   "v5",
    "n_users":         n_users,
    "n_products":      n_products,
    "n_interactions":  len(filtered),
    "n_train_ints":    len(train_ints),
    "n_test_ints":     len(test_ints),
    "n_eval_users":    n_eval,
    "split_strategy":  "time_based",
    "include_sources": INCLUDE_SOURCES,
    "include_actions": INCLUDE_ACTIONS,
    "hyperparams": {
        "factors":        FACTORS,
        "iterations":     ITERATIONS,
        "regularization": REGULARIZATION,
        "alpha":          ALPHA,
    },
    "weighting": weighting_config,
    "metrics": {
        f"precision@{k}": round(metrics[k]["precision"], 4) for k in TOP_K_LIST
    } | {
        f"recall@{k}":    round(metrics[k]["recall"],    4) for k in TOP_K_LIST
    } | {
        f"ndcg@{k}":      round(metrics[k]["ndcg"],      4) for k in TOP_K_LIST
    } | {
        f"hit_rate@{k}":  round(metrics[k]["hit_rate"],  4) for k in TOP_K_LIST
    } | {
        f"coverage@{k}":  round(metrics[k]["coverage"],  4) for k in TOP_K_LIST
    } | {
        f"composite@{k}": round(metrics[k]["composite"], 4) for k in TOP_K_LIST
    },
    "popular_items_count": len(popular_items),
}

with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
log(f"Meta   → {META_PATH}", C_GREEN, indent=2)

# ══════════════════════════════════════════════════════════════════════
# 10. SANITY CHECK với zero-score detection
# ══════════════════════════════════════════════════════════════════════
log("10. Sanity check — recommend thử 3 users", C_BOLD)
print()

sample_users = random.sample(all_user_ids, min(3, len(all_user_ids)))
for uid in sample_users:
    u_idx = user2idx[uid]
    # Dùng full_user_item để sanity check (production scenario)
    rec_ids, rec_scores = model.recommend(
        u_idx, full_user_item[u_idx], N=5, filter_already_liked_items=True
    )
    bought = [idx2product[i] for i in full_user_item[u_idx].indices[:3]]
    bought_names = [product_name.get(p, p)[:30] for p in bought]

    max_score    = float(rec_scores[0]) if len(rec_scores) > 0 else 0.0
    is_degenerate = max_score < ZERO_SCORE_THRESHOLD

    print(f"  User {uid[:12]}...")
    print(f"    Đã mua: {', '.join(bought_names) or '(none)'}")
    if is_degenerate:
        print(f"     ALS scores quá thấp ({max_score:.4f}) → dùng popular fallback")
        for rank, pid in enumerate(popular_items[:5], 1):
            name = product_name.get(pid, pid)[:45]
            cat  = product_cat.get(pid, "")
            print(f"      {rank}. [popular] {name}  [{cat}]")
    else:
        print(f"    Gợi ý:")
        for rank, (item_idx, score) in enumerate(zip(rec_ids, rec_scores), 1):
            pid  = idx2product[item_idx]
            name = product_name.get(pid, pid)[:45]
            cat  = product_cat.get(pid, "")
            print(f"      {rank}. [{score:.3f}] {name}  [{cat}]")
    print()

# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
print(f"{C_BOLD}{C_GREEN} HOÀN TẤT!{C_RESET}")
print(f"   Users train    : {n_users:,}  (min_inter≥{MIN_INTERACTIONS})")
print(f"   Products        : {n_products:,}")
print(f"   Interactions    : {len(filtered):,}  (train={len(train_ints):,} / test={len(test_ints):,})")
print(f"   Split strategy  : time-based  (test_ratio={TEST_RATIO})")
print(f"   precision@5     : {metrics[5]['precision']:.4f}")
print(f"   recall@5        : {metrics[5]['recall']:.4f}")
print(f"   NDCG@5          : {metrics[5]['ndcg']:.4f}")
print(f"   HitRate@5       : {metrics[5]['hit_rate']:.4f}")
print(f"   Coverage@10     : {metrics[10]['coverage']:.1%}")
print(f"   Composite@5     : {metrics[5]['composite']:.4f}")
print(f"   Model file      : {MODEL_PATH.name}  ({MODEL_PATH.stat().st_size/1024/1024:.1f} MB)")
print(f"\n   Bước tiếp theo: chạy uvicorn recommend_api:app --reload\n")