"""
processing/user.py — Xây dựng users từ review thật.
"""

def _activity_level(review_count: int) -> str:
    if review_count >= 5:
        return "high"
    elif review_count >= 2:
        return "medium"
    return "low"

def _spending_level(prices: list[int]) -> str:
    if not prices:
        return "unknown"
    avg = sum(prices) / len(prices)
    if avg <= 200_000:
        return "budget"
    elif avg <= 1_000_000:
        return "mid"
    return "premium"

def build_users_from_reviews(review_map: dict[str, list[dict]], product_map: dict[str, dict]) -> list[dict]:
    user_acc: dict[str, dict] = {}
    for prod_id, reviews in review_map.items():
        prod = product_map.get(prod_id)
        if not prod:
            continue
        cat = prod.get("category", "")
        price = prod.get("price", 0)
        for rv in reviews:
            rid = rv["reviewer_id"]
            name = rv["reviewer_name"]
            if not rid:
                continue
            if rid not in user_acc:
                user_acc[rid] = {
                    "user_id": rid,
                    "name": name,
                    "age": None,
                    "gender": None,
                    "preference_cats": set(),
                    "reviewed_prices": [],
                    "review_count": 0,
                    "created_at": rv["created_at"],
                }
            acc = user_acc[rid]
            if cat:
                acc["preference_cats"].add(cat)
            if price and price > 0:
                acc["reviewed_prices"].append(price)
            acc["review_count"] += 1
    users = []
    for uid, acc in user_acc.items():
        users.append({
            "user_id": uid,
            "name": acc["name"],
            "age": None,
            "gender": None,
            "preference_cats": list(acc["preference_cats"]),
            "activity_level": _activity_level(acc["review_count"]),
            "spending_level": _spending_level(acc["reviewed_prices"]),
            "review_count": acc["review_count"],
            "created_at": acc["created_at"],
        })
    return users

# ... giữ nguyên build_users_from_reviews, thêm cuối:

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from seed_refactor.db.connection import get_db
    from seed_refactor.db.storage import bulk_upsert

    db = get_db()

    # Load products
    products = list(db.products.find({}, {"_id": 0, "product_id": 1, "category": 1, "price": 1}))
    if not products:
        print("❌ Không có products trong MongoDB. Chạy product.py trước.")
        exit(1)
    product_map = {p["product_id"]: p for p in products}

    # Reconstruct review_map từ MongoDB reviews collection
    # (mỗi doc trong db.reviews có field product_id)
    review_docs = list(db.reviews.find({}, {"_id": 0}))
    if not review_docs:
        print("❌ Không có reviews trong MongoDB. Chạy crawl_reviews.py trước.")
        exit(1)

    review_map: dict[str, list[dict]] = {}
    for rv in review_docs:
        pid = rv.get("product_id")
        if pid:
            review_map.setdefault(pid, []).append(rv)

    users = build_users_from_reviews(review_map, product_map)

    bulk_upsert(db.users, users, key_field="user_id")
    db.users.create_index("user_id",        unique=True)
    db.users.create_index("activity_level")
    db.users.create_index("spending_level")

    print(f"✅ Đã lưu {len(users):,} users vào MongoDB (users)")
    print(f"   Bước tiếp theo: python -m seed_refactor.processing.interaction") 