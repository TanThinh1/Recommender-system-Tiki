import itertools
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
from db.connection import get_db

db = get_db()

FACTORS_LIST     = [64, 96, 128]
ALPHA_LIST       = [80, 120, 160, 200]
REG_LIST         = [0.05, 0.1, 0.15, 0.2]
ITERATIONS       = 60
TEST_RATIO       = 0.2
MIN_INTERACTIONS = 2
INCLUDE_SOURCES  = ["real_review"]

# (5) Mở rộng training signal — nhất quán với train_model v6
# Confidence = mức độ tin cậy: purchase mạnh nhất, view yếu nhất
ACTION_CONFIDENCE: dict[str, float] = {
    "purchase"    : 10.0,
    "add_to_cart" :  5.0,
    "click"       :  2.0,
    "view"        :  1.0,
}
INCLUDE_ACTIONS = list(ACTION_CONFIDENCE.keys())

# Temporal decay — copy từ train_model v4
USE_TEMPORAL_DECAY   = True
DECAY_HALF_LIFE_DAYS = 90          # tương tác >90 ngày giảm 50% weight

# IDF — copy từ train_model v4
USE_IDF = True

# Log-frequency — copy từ train_model v4
USE_LOG_FREQ = True

# Early stopping
EARLY_STOP_PATIENCE = 12           # None = tắt


# ══════════════════════════════════════════════════════════════════════
# 📐 HELPERS
# ══════════════════════════════════════════════════════════════════════
def composite_score(p5: float, r5: float, ndcg5: float, hr5: float) -> float:
    return 0.4 * p5 + 0.3 * r5 + 0.2 * ndcg5 + 0.1 * hr5


def evaluate_batch(model, purchase_filter_ui, test_ui, k: int = 5):
    """
    Trả về (precision@k, recall@k, ndcg@k, hit_rate@k).
    purchase_filter_ui : purchase-only train matrix — chỉ filter SP đã mua
                         (không filter view/click → model có thể gợi ý SP đã view)
    test_ui            : purchase ground truth từ test set
    """
    n_users = purchase_filter_ui.shape[0]
    recs = model.recommend(
        np.arange(n_users),
        purchase_filter_ui,              # ✅ [FIX v7] chỉ filter purchase
        N=k,
        filter_already_liked_items=True,
    )
    items_arr = recs[0] if isinstance(recs, tuple) else recs

    total_prec = total_recall = total_ndcg = total_hr = 0.0
    valid_users = 0

    for uid in range(n_users):
        test_items = set(test_ui[uid].indices)
        if not test_items:
            continue
        rec_list = items_arr[uid].tolist()
        rec_set  = set(rec_list)
        hits     = len(rec_set & test_items)

        total_prec   += hits / k
        total_recall += hits / len(test_items)
        total_hr     += 1.0 if hits > 0 else 0.0

        dcg  = sum(
            1.0 / math.log2(rank + 1)
            for rank, item in enumerate(rec_list, start=1)
            if item in test_items
        )
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(test_items), k)))
        total_ndcg += dcg / idcg if idcg > 0 else 0.0
        valid_users += 1

    if valid_users == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        total_prec   / valid_users,
        total_recall / valid_users,
        total_ndcg   / valid_users,
        total_hr     / valid_users,
    )


def _time_based_split(
    records   : list[dict],
    test_ratio: float = 0.2,
) -> tuple[list[dict], list[dict]]:
    """
    Chia records theo thứ tự thời gian.
    Nếu <50% records có timestamp → fallback per-user holdout.

    [FIX] Test ground truth = purchase-only + loại (user,product) đã có trong train
    → tránh precision=0 với multi-action data.
    """
    timestamped = [r for r in records if r.get("timestamp") is not None]
    no_ts       = [r for r in records if r.get("timestamp") is None]

    ts_ratio = len(timestamped) / len(records) if records else 0
    use_time = ts_ratio >= 0.5

    if not use_time:
        print(
            f"  ⚠️  Chỉ {len(timestamped):,}/{len(records):,} records có timestamp "
            f"({ts_ratio:.0%}) → fallback: per-user holdout (last {test_ratio:.0%} làm test)"
        )
        # Per-user holdout
        from collections import defaultdict as _dd
        user_ints: dict = _dd(list)
        for r in records:
            user_ints[r["user_id"]].append(r)
        train_records: list = []
        raw_test:      list = []
        for uid, ints in user_ints.items():
            n   = len(ints)
            cut = max(1, n - max(1, round(n * test_ratio)))
            train_records.extend(ints[:cut])
            raw_test.extend(ints[cut:])
    else:
        timestamped.sort(key=lambda r: r["timestamp"])
        cutoff_idx    = int(len(timestamped) * (1 - test_ratio))
        cutoff_ts     = timestamped[cutoff_idx]["timestamp"] if cutoff_idx < len(timestamped) else None
        train_records = timestamped[:cutoff_idx] + no_ts
        raw_test      = timestamped[cutoff_idx:]
        if cutoff_ts:
            print(f"  Time split: train < {cutoff_ts}  |  test ≥ {cutoff_ts}")

    # [FIX v7] Purchase-only ground truth, KHÔNG lọc overlap với train
    # (cùng logic với train_model v7 — xem giải thích chi tiết ở đó)
    test_records = [r for r in raw_test if r.get("action") == "purchase"]

    print(
        f"  Train records: {len(train_records):,}  |  "
        f"Test records (purchase-only): {len(test_records):,}"
    )
    return train_records, test_records


# ══════════════════════════════════════════════════════════════════════
# 1. LOAD DATA (thêm timestamp để time-split)
# ══════════════════════════════════════════════════════════════════════
print("Load interactions...")
query = {
    "action": {"$in": INCLUDE_ACTIONS},
    "source": {"$in": INCLUDE_SOURCES},
}
raw = list(db.interactions.find(
    query,
    {"user_id": 1, "product_id": 1, "weight": 1, "action": 1, "timestamp": 1, "_id": 0},
))

if not raw:
    sys.exit("Không có dữ liệu. Kiểm tra lại action/source filter.")

# ── Filter MIN_INTERACTIONS ───────────────────────────────────────────
user_counts  = defaultdict(int)
for r in raw:
    user_counts[r["user_id"]] += 1
active_users = {u for u, c in user_counts.items() if c >= MIN_INTERACTIONS}
all_records  = [r for r in raw if r["user_id"] in active_users]
print(f"  {len(all_records):,} interactions / {len(active_users):,} users (min_inter≥{MIN_INTERACTIONS})")

# ── Time-based split ──────────────────────────────────────────────────
train_records, test_records = _time_based_split(all_records, TEST_RATIO)

if not test_records:
    sys.exit("Test set rỗng sau time-split — kiểm tra lại dữ liệu timestamp.")

# ── Mappings từ TRAIN records (không dùng test để build vocab) ────────
all_train_users    = sorted({r["user_id"]    for r in train_records})
all_train_products = sorted({r["product_id"] for r in train_records})
user2idx     = {u: i for i, u in enumerate(all_train_users)}
product2idx  = {p: i for i, p in enumerate(all_train_products)}
n_users      = len(all_train_users)
n_products   = len(all_train_products)
print(f"  Train vocab: {n_users:,} users × {n_products:,} products")

# ── Báo cáo cold-start trong test ─────────────────────────────────────
test_cold_users = sum(1 for r in test_records if r["user_id"]    not in user2idx)
test_cold_items = sum(1 for r in test_records if r["product_id"] not in product2idx)
print(
    f"  Test records bị bỏ qua (cold): "
    f"{test_cold_users:,} user mới, {test_cold_items:,} item mới"
)


# ══════════════════════════════════════════════════════════════════════
# 2. TEMPORAL DECAY SETUP — copy từ train_model v4
# ══════════════════════════════════════════════════════════════════════
now_ts       = datetime.now(tz=timezone.utc)
decay_lambda = math.log(2) / DECAY_HALF_LIFE_DAYS   # λ = ln2 / half_life


def _temporal_weight(ts) -> float:
    """e^(-λ × days_ago). Trả về 1.0 nếu không có timestamp."""
    if not USE_TEMPORAL_DECAY or ts is None:
        return 1.0
    try:
        if isinstance(ts, datetime):
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        else:
            return 1.0
        days_ago = (now_ts - ts).total_seconds() / 86400.0
        return math.exp(-decay_lambda * max(0.0, days_ago))
    except Exception:
        return 1.0


# ══════════════════════════════════════════════════════════════════════
# 3. ACCUMULATE TRAIN PAIRS — với temporal decay
# ══════════════════════════════════════════════════════════════════════
train_pair_count = defaultdict(int)
train_pair_wsum  = defaultdict(float)   # sum of base_w × decay

for rec in train_records:
    uid = rec["user_id"]
    pid = rec["product_id"]
    if uid not in user2idx or pid not in product2idx:
        continue
    key    = (user2idx[uid], product2idx[pid])
    # (5) ACTION_CONFIDENCE thay vì weight field cố định
    base_w = ACTION_CONFIDENCE.get(rec.get("action", ""), float(rec.get("weight") or 1.0))
    decay  = _temporal_weight(rec.get("timestamp"))
    train_pair_count[key]  += 1
    train_pair_wsum[key]   += base_w * decay


# ══════════════════════════════════════════════════════════════════════
# 4. IDF WEIGHTING — copy từ train_model v4
# ══════════════════════════════════════════════════════════════════════
# item_user_count: pid_idx → số user đã tương tác (trong train)
item_user_count = defaultdict(set)
for (u_idx, p_idx) in train_pair_count:
    item_user_count[p_idx].add(u_idx)


def _item_idf(pid_idx: int) -> float:
    """IDF smooth: log((N+1)/(df+1)) + 1"""
    if not USE_IDF:
        return 1.0
    df = len(item_user_count.get(pid_idx, set()))
    return math.log((n_users + 1) / (df + 1)) + 1.0


# Normalize IDF về [0.5, 1.5] — nhất quán với train_model v4
all_idf_vals = [_item_idf(p) for p in range(n_products)]
idf_min  = min(all_idf_vals)
idf_max  = max(all_idf_vals)
idf_range = (idf_max - idf_min) or 1.0


def _idf_normalized(pid_idx: int) -> float:
    return 0.5 + (_item_idf(pid_idx) - idf_min) / idf_range   # → [0.5, 1.5]


# ══════════════════════════════════════════════════════════════════════
# 5. PRECOMPUTE base_scaled_w — tính 1 lần, dùng lại cho mọi alpha
#
#    w_final   = avg_w_decay × log(1+count)   [log-freq + decay]
#    w_scaled  = w_final × idf_normalized(p)  [IDF scaling]
#    confidence = 1 + alpha × w_scaled        [computed per alpha in loop]
# ══════════════════════════════════════════════════════════════════════
train_keys = list(train_pair_count.keys())

base_rows_arr = np.array([k[0] for k in train_keys], dtype=np.int32)
base_cols_arr = np.array([k[1] for k in train_keys], dtype=np.int32)
base_scaled_w = np.array(
    [
        (train_pair_wsum[k] / train_pair_count[k])     # avg_w × decay
        * (math.log1p(train_pair_count[k]) if USE_LOG_FREQ else 1.0)  # log-freq
        * _idf_normalized(k[1])                         # IDF
        for k in train_keys
    ],
    dtype=np.float32,
)

print(
    f"\n  base_scaled_w — min={base_scaled_w.min():.4f}  "
    f"max={base_scaled_w.max():.4f}  "
    f"mean={base_scaled_w.mean():.4f}"
)
print(
    f"  Weighting: IDF={'ON' if USE_IDF else 'OFF'}  "
    f"LogFreq={'ON' if USE_LOG_FREQ else 'OFF'}  "
    f"TemporalDecay={'ON' if USE_TEMPORAL_DECAY else 'OFF'}"
)

# ══════════════════════════════════════════════════════════════════════
# 6. BUILD MATRICES (test + purchase-only filter)
# ══════════════════════════════════════════════════════════════════════
# Test matrix — ground truth (purchase-only)
test_rows, test_cols = [], []
for rec in test_records:
    uid = rec["user_id"]
    pid = rec["product_id"]
    if uid not in user2idx or pid not in product2idx:
        continue
    test_rows.append(user2idx[uid])
    test_cols.append(product2idx[pid])

if not test_rows:
    sys.exit(
        "Test matrix rỗng — không có purchase nào trong test set. "
        "Kiểm tra lại data hoặc giảm TEST_RATIO."
    )

test_matrix = sp.csr_matrix(
    (np.ones(len(test_rows), dtype=np.float32), (test_rows, test_cols)),
    shape=(n_users, n_products),
)
print(f"  Test matrix: {test_matrix.nnz:,} entries  "
      f"({len(set(test_rows)):,} users có ít nhất 1 test item)")

# [FIX v7] Purchase-only filter matrix — dùng khi evaluate để chỉ filter SP đã mua
# Không filter view/click → model có thể gợi ý SP user đã view nhưng chưa mua
purchase_train_records = [r for r in train_records if r.get("action") == "purchase"]
p_rows, p_cols = [], []
for rec in purchase_train_records:
    uid = rec["user_id"]
    pid = rec["product_id"]
    if uid not in user2idx or pid not in product2idx:
        continue
    p_rows.append(user2idx[uid])
    p_cols.append(product2idx[pid])

purchase_filter_matrix = sp.csr_matrix(
    (np.ones(len(p_rows), dtype=np.float32), (p_rows, p_cols)) if p_rows
    else (np.array([]), (np.array([]), np.array([]))),
    shape=(n_users, n_products),
)
print(f"  Purchase-filter matrix: {purchase_filter_matrix.nnz:,} entries")


# ══════════════════════════════════════════════════════════════════════
# 7. GRID SEARCH
# ══════════════════════════════════════════════════════════════════════
best_composite = 0.0
best_params    = None
all_results    = []
no_improve_cnt = 0
total_combos   = len(FACTORS_LIST) * len(ALPHA_LIST) * len(REG_LIST)

print(f"\nGrid search v3: {total_combos} combos  "
      f"(factors×alpha×reg = {len(FACTORS_LIST)}×{len(ALPHA_LIST)}×{len(REG_LIST)})")
print(f"{'─'*82}")
print(
    f"{'factors':>7} {'alpha':>6} {'reg':>6} | "
    f"{'P@5':>7} {'R@5':>7} {'NDCG@5':>8} {'HR@5':>7} {'Composite':>10} {'Time':>6}"
)
print(f"{'─'*82}")

for factors, alpha, reg in itertools.product(FACTORS_LIST, ALPHA_LIST, REG_LIST):
    # confidence thay đổi theo alpha — IDF/decay đã baked vào base_scaled_w
    confidence = (1.0 + alpha * base_scaled_w).astype(np.float32)

    item_user_mat = sp.csr_matrix(
        (confidence, (base_cols_arr, base_rows_arr)),
        shape=(n_products, n_users),
    )
    train_matrix = item_user_mat.T.tocsr()   # user-item (n_users × n_products)

    model = AlternatingLeastSquares(
        factors      = factors,
        iterations   = ITERATIONS,
        regularization = reg,
        random_state = 42,
        use_gpu      = False,
    )
    t0 = time.time()
    model.fit(train_matrix, show_progress=False)
    elapsed = time.time() - t0

    p5, r5, ndcg5, hr5 = evaluate_batch(model, purchase_filter_matrix, test_matrix, k=5)
    comp = composite_score(p5, r5, ndcg5, hr5)

    all_results.append({
        "factors": factors, "alpha": alpha, "reg": reg,
        "p5"     : round(p5,    4),
        "r5"     : round(r5,    4),
        "ndcg5"  : round(ndcg5, 4),
        "hr5"    : round(hr5,   4),
        "composite": round(comp, 4),
    })

    marker = " ◄ BEST" if comp > best_composite else ""
    print(
        f"{factors:>7} {alpha:>6} {reg:>6.3f} | "
        f"{p5:>7.4f} {r5:>7.4f} {ndcg5:>8.4f} "
        f"{hr5:>7.4f} {comp:>10.4f} {elapsed:>5.1f}s{marker}"
    )

    if comp > best_composite:
        best_composite = comp
        best_params    = {"factors": factors, "alpha": alpha, "reg": reg}
        no_improve_cnt = 0
    else:
        no_improve_cnt += 1
        if EARLY_STOP_PATIENCE and no_improve_cnt >= EARLY_STOP_PATIENCE:
            print(f"\n⏹  Early stop: không cải thiện sau {EARLY_STOP_PATIENCE} combos.")
            break

print(f"{'─'*82}")

# ── Top-3 ─────────────────────────────────────────────────────────────
all_results.sort(key=lambda x: x["composite"], reverse=True)
print(f"\n🏆 Top-3 combos:")
for i, res in enumerate(all_results[:3], 1):
    print(
        f"  #{i}: factors={res['factors']}, alpha={res['alpha']}, reg={res['reg']}  "
        f"→ P@5={res['p5']:.4f}  R@5={res['r5']:.4f}  "
        f"NDCG@5={res['ndcg5']:.4f}  Composite={res['composite']:.4f}"
    )

print(f"\n✅ Best: {best_params}  composite = {best_composite:.4f}")

# ── Lưu kết quả ──────────────────────────────────────────────────────
best_params_out = {
    **best_params,
    "_meta": {
        "grid_version"    : "v3",
        "idf"             : USE_IDF,
        "temporal_decay"  : USE_TEMPORAL_DECAY,
        "log_freq"        : USE_LOG_FREQ,
        "split"           : "time_based",
        "test_ratio"      : TEST_RATIO,
        "composite_best"  : round(best_composite, 4),
        "n_train_users"   : n_users,
        "n_train_products": n_products,
    },
}
with open(BASE_DIR / "best_params.json", "w", encoding="utf-8") as f:
    json.dump(best_params_out, f, indent=2)
print("Saved → best_params.json")

with open(BASE_DIR / "grid_results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)
print("Saved → grid_results.json  (full results)")