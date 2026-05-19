"""
processing/interaction.py — Xây dựng interactions (view, cart, purchase) theo tỉ lệ TMĐT.
"""

import random
from collections import defaultdict
from datetime import datetime, timedelta

from seed_refactor.config import RATING_TIERS, CART_WEIGHT_RATIO, MAX_INTERACTIONS, log, C_DIM, C_RED
from seed_refactor.utils.progress import tqdm

_POS_WORDS = {
    "tốt", "tuyệt", "xuất sắc", "đẹp", "chất lượng", "hài lòng", "thích", "ngon",
    "nhanh", "ổn", "ok", "tốt lắm", "đỉnh", "hoàn hảo", "xịn", "chuẩn", "đáng tiền",
    "recommended", "recommend", "great", "good", "excellent", "love", "perfect",
}
_NEG_WORDS = {
    "kém", "tệ", "dở", "xấu", "hỏng", "lỗi", "vỡ", "không như", "thất vọng",
    "fake", "giả", "nhái", "chậm", "thiếu", "sai", "không đúng", "trả hàng",
    "bad", "poor", "terrible", "broken", "disappointed", "wrong", "damaged",
}

def _simple_sentiment(text: str, rating: int) -> str:
    if not text:
        return "positive" if rating >= 4 else "neutral" if rating == 3 else "negative"
    tl = text.lower()
    pos = sum(1 for w in _POS_WORDS if w in tl)
    neg = sum(1 for w in _NEG_WORDS if w in tl)
    if pos > neg and rating >= 3:
        return "positive"
    elif neg > pos or rating <= 2:
        return "negative"
    return "neutral"

def _interaction_weight(rating: int, is_buyer: bool, sentiment: str) -> float:
    base = rating / 5.0
    bonus = (0.15 if is_buyer else 0.0) + (0.05 if sentiment == "positive" else -0.10 if sentiment == "negative" else 0.0)
    return round(min(max(base + bonus, 0.05), 1.0), 4)

def build_interactions(products: list[dict], users: list[dict], review_map: dict[str, list[dict]]) -> list[dict]:
    rng = random.Random(42)
    user_set = {u["user_id"] for u in users}
    prod_map = {p["product_id"]: p for p in products}
    user_map = {u["user_id"]: u for u in users}
    user_ids = list(user_set)
    prod_ids = [p["product_id"] for p in products]

    user_weights = [max(user_map[u].get("review_count", 1), 1) for u in user_ids]
    prod_weights = [max(float(prod_map[p].get("popularity_score") or 0.01), 0.01) for p in prod_ids]

    def _avg_rating_from_reviews(prod_id: str) -> float:
        revs = review_map.get(prod_id, [])
        if not revs:
            return 3.5
        return sum(r["rating"] for r in revs) / len(revs)

    avg_cache = {p: _avg_rating_from_reviews(p) for p in prod_ids}

    real_by_tier = {"1-2": [], "2-3": [], "3-4": [], "4-5": []}
    for prod_id, reviews in review_map.items():
        prod = prod_map.get(prod_id)
        if not prod:
            continue
        avg = avg_cache.get(prod_id, 3.5)
        tier_key = "1-2" if avg < 2 else "2-3" if avg < 3 else "3-4" if avg < 4 else "4-5"
        for rv in reviews:
            uid = rv["reviewer_id"]
            if uid not in user_set or not rv.get("is_buyer", False):
                continue
            real_by_tier[tier_key].append((uid, prod_id, rv, avg))

    tier_stats = {}
    total_estimated = 0
    for tk, tier_cfg in RATING_TIERS.items():
        n_real = len(real_by_tier[tk])
        parts = 1 + tier_cfg["cart_ratio"] + tier_cfg["view_ratio"]
        n_est = int(n_real / tier_cfg["review_rate"]) if n_real > 0 else 0
        n_synth_raw = int((n_est - n_real) * tier_cfg["purchase_weight"]) if n_est > n_real else 0
        tier_stats[tk] = {"n_real": n_real, "n_est": n_est, "n_synth_raw": n_synth_raw, "parts": parts}
        total_estimated += (n_real + n_synth_raw) * parts

    scale = 1.0
    if MAX_INTERACTIONS and total_estimated > MAX_INTERACTIONS:
        scale = MAX_INTERACTIONS / total_estimated
    for tk in tier_stats:
        tier_stats[tk]["n_synth"] = int(tier_stats[tk]["n_synth_raw"] * scale)

    log(f"Ước tính tổng trước cap: {total_estimated:,}", C_DIM, indent=1)
    if MAX_INTERACTIONS:
        log(f"Scale factor: {scale:.3f}  →  mục tiêu {MAX_INTERACTIONS:,}", C_DIM, indent=1)

    interactions = []
    all_purchase_bases = []
    existing_pairs = set()
    existing_view_pairs = set()
    base_ts = datetime.now()

    def _rand_ts(days_back_max: int = 90) -> datetime:
        """Trả về datetime object — để MongoDB lưu đúng kiểu Date, không phải string."""
        offset = timedelta(
            days=rng.randint(0, days_back_max),
            hours=rng.randint(0, 23),
            minutes=rng.randint(0, 59),
        )
        return base_ts - offset

    def _make_base(uid, prod_id, prod, user, rating, sentiment, review_text, ts, source):
        """
        ts: có thể là str (ISO format từ review) hoặc datetime object (từ _rand_ts).
        Luôn chuyển về datetime object để MongoDB lưu đúng kiểu Date.
        Nếu không parse được → dùng datetime.now().
        """
        if isinstance(ts, datetime):
            dt = ts
        else:
            try:
                dt = datetime.fromisoformat(str(ts))
            except Exception:
                dt = datetime.now()
        hour, dow = dt.hour, dt.weekday()
        return {
            "user_id": uid, "product_id": prod_id, "rating": float(rating), "is_buyer": True,
            "category": prod["category"], "price_range": prod.get("price_range", "unknown"),
            "in_pref": prod["category"] in user.get("preference_cats", []),
            "sentiment": sentiment, "review_text": review_text,
            "timestamp": dt,          # ✅ datetime object — MongoDB lưu đúng kiểu Date
            "source": source,
            "context": {"hour": hour, "day_of_week": dow, "device": "unknown", "location": "unknown"},
        }

    now_dt = datetime.now()
    for tk, records in real_by_tier.items():
        pbar = tqdm(records, desc=f"✅ Real {tk}★", unit="rec", ncols=90, leave=False)
        for uid, prod_id, rv, avg in pbar:
            existing_pairs.add((uid, prod_id))
            prod = prod_map[prod_id]; user = user_map[uid]
            rating = int(rv["rating"]); review_txt = rv.get("review_text", "")
            # Parse timestamp từ review — chuyển về datetime object
            raw_ts = rv.get("created_at")
            if isinstance(raw_ts, datetime):
                ts = raw_ts
            elif raw_ts:
                try:
                    ts = datetime.fromisoformat(str(raw_ts))
                except Exception:
                    ts = now_dt
            else:
                ts = now_dt
            sentiment = _simple_sentiment(review_txt, rating)
            weight = _interaction_weight(rating, True, sentiment)
            base = _make_base(uid, prod_id, prod, user, rating, sentiment, review_txt, ts, "real_review")
            all_purchase_bases.append((base, weight, tk))
            interactions.append({**base, "action": "purchase", "weight": weight})
            interactions.append({**base, "action": "add_to_cart", "weight": round(weight * CART_WEIGHT_RATIO, 4)})
        pbar.close()

    for tk, tier_cfg in RATING_TIERS.items():
        n_synth = tier_stats[tk]["n_synth"]
        if n_synth <= 0:
            continue
        tier_prod_ids = [p for p in prod_ids if ( "1-2" if avg_cache.get(p,3.5)<2 else "2-3" if avg_cache.get(p,3.5)<3 else "3-4" if avg_cache.get(p,3.5)<4 else "4-5") == tk]
        if not tier_prod_ids:
            tier_prod_ids = prod_ids
        tier_prod_weights = [max(float(prod_map[p].get("popularity_score") or 0.01), 0.01) for p in tier_prod_ids]
        generated = 0; attempts = 0; max_tries = n_synth * 8
        pbar = tqdm(total=n_synth, desc=f"🔧 Synth {tk}★", unit="rec", ncols=90, leave=False)
        while generated < n_synth and attempts < max_tries:
            attempts += 1
            uid = rng.choices(user_ids, weights=user_weights, k=1)[0]
            prod_id = rng.choices(tier_prod_ids, weights=tier_prod_weights, k=1)[0]
            if (uid, prod_id) in existing_pairs:
                continue
            existing_pairs.add((uid, prod_id))
            prod = prod_map[prod_id]; user = user_map[uid]
            pop = float(prod.get("popularity_score") or 0.3)
            weight = round(min(max(pop * 0.6 + 0.2, 0.05), 1.0), 4)
            sentiment = "neutral"
            rating = min(round(weight * 5), 5)
            base = _make_base(uid, prod_id, prod, user, rating, sentiment, "", _rand_ts(), "synthetic")
            all_purchase_bases.append((base, weight, tk))
            interactions.append({**base, "action": "purchase", "weight": weight})
            interactions.append({**base, "action": "add_to_cart", "weight": round(weight * CART_WEIGHT_RATIO, 4)})
            generated += 1
            pbar.update(1)
        pbar.close()
        if generated < n_synth:
            log(f"⚠️  {tk}★ chỉ sinh được {generated:,}/{n_synth:,} synthetic", C_RED, indent=2)

    log(f"👁️  Sinh views cho {len(all_purchase_bases):,} purchase bases...", C_DIM, indent=1)
    pbar = tqdm(all_purchase_bases, desc="👁️  Views", unit="base", ncols=90)
    for base, p_weight, tk in pbar:
        tier_cfg = RATING_TIERS[tk]
        view_ratio = tier_cfg["view_ratio"]
        prod_id = base["product_id"]
        prod = prod_map[prod_id]
        view_w = round(p_weight * 0.15, 4)
        available_users = [u for u in user_ids if (u, prod_id) not in existing_view_pairs]
        k = min(view_ratio, len(available_users))
        if k <= 0:
            continue
        chosen = rng.sample(available_users, k)
        for viewer_uid in chosen:
            existing_view_pairs.add((viewer_uid, prod_id))
            viewer = user_map[viewer_uid]
            view_ts = _rand_ts(days_back_max=60)   # datetime object
            interactions.append({
                "user_id": viewer_uid, "product_id": prod_id, "rating": 0.0, "is_buyer": False,
                "category": prod["category"], "price_range": prod.get("price_range", "unknown"),
                "in_pref": prod["category"] in viewer.get("preference_cats", []),
                "sentiment": "neutral", "review_text": "", "timestamp": view_ts,  # ✅ datetime
                "source": "synthetic_view",
                "action": "view", "weight": view_w,
                "context": {"hour": rng.randint(7, 23), "day_of_week": rng.randint(0, 6), "device": "unknown", "location": "unknown"},
            })
        pbar.set_postfix_str(f"total={len(interactions):,}")
    pbar.close()
    return interactions

# ... giữ nguyên build_interactions, thêm cuối:

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from seed_refactor.db.connection import get_db
    from seed_refactor.db.storage import insert_interactions_safe

    db = get_db()

    # Load từ MongoDB — không cần JSON file
    products = list(db.products.find({}, {"_id": 0}))
    if not products:
        print("❌ Không có products trong MongoDB. Chạy product.py trước.")
        exit(1)

    users = list(db.users.find({}, {"_id": 0}))
    if not users:
        print("❌ Không có users trong MongoDB. Chạy user.py trước.")
        exit(1)

    # Reconstruct review_map từ db.reviews
    review_docs = list(db.reviews.find({}, {"_id": 0}))
    review_map: dict[str, list[dict]] = {}
    for rv in review_docs:
        pid = rv.get("product_id")
        if pid:
            review_map.setdefault(pid, []).append(rv)

    # Sinh interactions — datetime objects, không serialize JSON
    interactions = build_interactions(products, users, review_map)
    print(f"✅ Sinh được {len(interactions):,} interactions")

    # Dedup theo 5 trường của unique index trước khi insert
    seen = set()
    unique_interactions = []
    for doc in interactions:
        ts_key = doc["timestamp"].isoformat() if isinstance(doc["timestamp"], datetime) else str(doc["timestamp"])
        key = (doc["user_id"], doc["product_id"], doc["action"], doc.get("source", ""), ts_key)
        if key not in seen:
            seen.add(key)
            unique_interactions.append(doc)
    removed = len(interactions) - len(unique_interactions)
    if removed:
        print(f"   Dedup: {len(interactions):,} → {len(unique_interactions):,} (bỏ {removed:,} duplicate)")
    interactions = unique_interactions

    # Xóa collection cũ → tránh duplicate
    db.interactions.drop()
    print("🗑️  Đã xóa collection interactions cũ")

    insert_interactions_safe(db.interactions, interactions)

    # Tạo indexes
    db.interactions.create_index(
        [("user_id", 1), ("product_id", 1), ("action", 1), ("source", 1), ("timestamp", 1)],
        unique=True, name="idx_interaction_unique",
    )
    db.interactions.create_index([("product_id", 1), ("rating", -1)])
    db.interactions.create_index([("timestamp", -1)])
    db.interactions.create_index("action")
    db.interactions.create_index("sentiment")

    # Verify timestamp type
    sample = db.interactions.find_one({"timestamp": {"$exists": True}})
    if sample:
        ts_type = type(sample["timestamp"]).__name__
        status  = "✅" if ts_type == "datetime" else "⚠️ "
        print(f"{status} Timestamp type trong MongoDB: {ts_type}")

    total = db.interactions.count_documents({})
    print(f"✅ Tổng interactions trong MongoDB: {total:,}")
    print(f"   Bước tiếp theo: python generate_embeddings.py")