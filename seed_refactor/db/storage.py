"""
db/storage.py — Upsert an toàn và insert interactions.
"""

from datetime import datetime
from seed_refactor.utils.progress import tqdm

try:
    from pymongo import UpdateOne, errors
except ImportError:
    pass


def _parse_ts(val) -> datetime | None:
    """
    Chuyển timestamp từ bất kỳ dạng nào → datetime object.
    JSON load trả về string; MongoDB cần datetime.
    Trả về None nếu không parse được.
    """
    if isinstance(val, datetime):
        return val
    if isinstance(val, (int, float)):
        try:
            return datetime.fromtimestamp(val)
        except Exception:
            return None
    if isinstance(val, str) and val:
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ):
            try:
                return datetime.strptime(val, fmt)
            except ValueError:
                continue
    return None


def _normalize_interactions(interactions: list[dict]) -> list[dict]:
    """
    Chuẩn hoá interactions trước khi insert vào MongoDB:
      - Chuyển field 'timestamp' từ ISO string → datetime object
        (JSON không lưu được datetime; MongoDB cần datetime để query/sort)
      - Bỏ các doc không parse được timestamp (log cảnh báo)
    """
    result = []
    n_failed = 0
    for doc in interactions:
        raw_ts = doc.get("timestamp")
        ts = _parse_ts(raw_ts)
        if ts is None:
            n_failed += 1
            # Fallback: gán datetime.now() để không mất record
            doc = {**doc, "timestamp": datetime.now()}
        else:
            doc = {**doc, "timestamp": ts}
        result.append(doc)
    if n_failed:
        print(f"  ⚠️  {n_failed:,} interactions không parse được timestamp → dùng datetime.now()")
    return result


def bulk_upsert(collection, docs: list[dict], key_field: str, batch: int = 500):
    if not docs:
        return
    INSERT_ONLY_FIELDS = {"created_at", "name"}
    pbar = tqdm(range(0, len(docs), batch), desc=f"  🔄 {collection.name}", unit="batch", ncols=90)
    n_upserted = n_modified = 0
    for i in pbar:
        chunk = docs[i:i+batch]
        ops = []
        for doc in chunk:
            set_fields      = {k: v for k, v in doc.items() if k not in INSERT_ONLY_FIELDS}
            set_on_insert   = {k: v for k, v in doc.items() if k in INSERT_ONLY_FIELDS}
            update = {"$set": set_fields}
            if set_on_insert:
                update["$setOnInsert"] = set_on_insert
            ops.append(UpdateOne({key_field: doc[key_field]}, update, upsert=True))
        result = collection.bulk_write(ops, ordered=False)
        n_upserted += result.upserted_count
        n_modified  += result.modified_count
        pbar.set_postfix_str(
            f"new={n_upserted:,} upd={n_modified:,} [{min(i+batch, len(docs)):,}/{len(docs):,}]"
        )
    pbar.close()


def insert_interactions_safe(collection, interactions: list[dict], batch: int = 2000):
    if not interactions:
        return
    from pymongo.errors import BulkWriteError

    # ✅ Chuẩn hoá timestamp trước khi insert
    interactions = _normalize_interactions(interactions)

    pbar = tqdm(
        range(0, len(interactions), batch),
        desc=f"  💾 {collection.name}", unit="batch", ncols=90,
    )
    n_inserted = 0
    for i in pbar:
        chunk = interactions[i:i+batch]
        try:
            result = collection.insert_many(chunk, ordered=False)
            n_inserted += len(result.inserted_ids)
        except BulkWriteError as bwe:
            n_inserted += bwe.details.get("nInserted", 0)
        except Exception as e:
            raise RuntimeError(f"Insert error at batch {i}: {e}") from e
        pbar.set_postfix_str(f"inserted={n_inserted:,}")
    pbar.close()


if __name__ == "__main__":
    """
    storage.py __main__ — Tiện ích verify và tạo lại indexes.

    Không còn đọc JSON files nữa — tất cả data đã được từng bước
    (tiki_crawler, product, crawl_reviews, user, interaction) lưu thẳng vào MongoDB.

    Dùng script này để:
      - Kiểm tra số lượng documents trong từng collection
      - Tạo lại indexes nếu cần (vd: sau khi restore backup)
      - Verify timestamp type trong interactions
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from seed_refactor.db.connection import get_db

    db = get_db()

    # ── Verify collections ────────────────────────────────────────────
    collections = ["raw_products", "products", "reviews", "users", "interactions"]
    print("\n📊 Thống kê collections:")
    for col in collections:
        count = db[col].count_documents({})
        print(f"   {col:<20} {count:>10,} docs")

    # ── Verify timestamp ──────────────────────────────────────────────
    sample = db.interactions.find_one({"timestamp": {"$exists": True}})
    if sample:
        ts_type = type(sample["timestamp"]).__name__
        status  = "✅" if ts_type == "datetime" else "⚠️ "
        print(f"\n{status} interactions.timestamp type: {ts_type}")
    else:
        print("\n⚠️  Không tìm thấy interaction nào có timestamp")

    # ── Recreate indexes ──────────────────────────────────────────────
    print("\n🔧 Tạo lại indexes...")

    db.raw_products.create_index("raw_id",    unique=True)
    db.raw_products.create_index("category")

    db.products.create_index("product_id",            unique=True)
    db.products.create_index("category")
    db.products.create_index("brand")
    db.products.create_index([("popularity_score", -1)])

    db.reviews.create_index([("product_id", 1)])
    db.reviews.create_index([("product_id", 1), ("reviewer_id", 1)])
    db.reviews.create_index([("rating", -1)])

    db.users.create_index("user_id",        unique=True)
    db.users.create_index("activity_level")
    db.users.create_index("spending_level")

    try:
        db.interactions.drop_index("idx_interaction_unique")
    except Exception:
        pass
    db.interactions.create_index(
        [("user_id", 1), ("product_id", 1), ("action", 1), ("source", 1), ("timestamp", 1)],
        unique=True, name="idx_interaction_unique",
    )
    db.interactions.create_index([("product_id", 1), ("rating", -1)])
    db.interactions.create_index([("timestamp", -1)])
    db.interactions.create_index("action")
    db.interactions.create_index("sentiment")

    print("✅ Indexes đã được tạo/cập nhật")