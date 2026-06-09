from __future__ import annotations

import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ltr")

BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "ltr_model.pkl"

# Label mapping — action nào coi là "positive" (user thực sự quan tâm)
POSITIVE_ACTIONS = {"purchase", "add_to_cart"}
ALL_ACTIONS      = {"purchase", "add_to_cart", "click", "view"}

FEATURES = ["price_norm", "rating_norm", "stock_flag", "cat_purchase_count_norm"]


# ══════════════════════════════════════════════════════════════════════
# 1. BUILD FEATURE MATRIX
# ══════════════════════════════════════════════════════════════════════
def build_features(db) -> pd.DataFrame:
    """
    Kéo interaction log + product metadata từ MongoDB → feature DataFrame.

    Returns
    -------
    df : DataFrame với columns [user_id, product_id, label, price_norm,
                                 rating_norm, stock_flag, cat_purchase_count_norm]
    """
    log.info("Load interactions...")
    interactions = list(db.interactions.find(
        {"action": {"$in": list(ALL_ACTIONS)}},
        {"_id": 0, "user_id": 1, "product_id": 1, "action": 1},
    ))
    if not interactions:
        raise ValueError("Không có interaction data. Kiểm tra MongoDB.")

    df = pd.DataFrame(interactions)
    df["label"] = df["action"].apply(lambda a: 1 if a in POSITIVE_ACTIONS else 0)
    log.info(f"  {len(df):,} interactions  |  positive={df['label'].sum():,}  negative={(df['label']==0).sum():,}")

    # ── Product features ──────────────────────────────────────────────
    log.info("Load products...")
    products = pd.DataFrame(list(db.products.find(
        {},
        {"_id": 0, "product_id": 1, "price": 1, "rating_avg": 1, "stock": 1, "category": 1},
    )))
    if products.empty:
        raise ValueError("Không có product data.")

    # Đổi tên rating_avg → product_rating để tránh xung đột với cột rating của interactions
    products = products.rename(columns={"rating_avg": "product_rating"})

    df = df.merge(products, on="product_id", how="left")

    # ── User–category affinity: số lần user mua trong category ───────
    # Tính từ purchase interaction + product category
    log.info("Compute user-category affinity...")
    purchase_df = df[df["action"] == "purchase"][["user_id", "category"]].copy()
    cat_aff = (
        purchase_df
        .groupby(["user_id", "category"])
        .size()
        .reset_index(name="cat_purchase_count")
    )
    df = df.merge(cat_aff, on=["user_id", "category"], how="left")

    # ── Normalize features ────────────────────────────────────────────
    MAX_PRICE = 10_000_000   # 10M VND

    df["price_norm"]              = (df["price"].fillna(0).clip(upper=MAX_PRICE) / MAX_PRICE).astype(float)
    df["rating_norm"]             = (df["product_rating"].fillna(0).clip(upper=5.0) / 5.0).astype(float)
    # stock=None nghĩa là chưa có dữ liệu (unknown), KHÔNG phải hết hàng.
    # fillna(0) cũ → None bị coi là stock=0 → stock_flag=0 → zeroes out toàn bộ score.
    # Thay bằng fillna(1): nếu không biết tồn kho, giả định còn hàng (safe default).
    df["stock_flag"]              = (df["stock"].fillna(1) > 0).astype(int)
    df["cat_purchase_count_norm"] = (df["cat_purchase_count"].fillna(0).clip(upper=20) / 20).astype(float)

    log.info(f"Feature matrix: {len(df):,} rows × {len(FEATURES)} features")
    return df


# ══════════════════════════════════════════════════════════════════════
# 2. TRAIN
# ══════════════════════════════════════════════════════════════════════
def train(db) -> dict:
    """
    Train LightGBM classifier và lưu vào ltr_model.pkl.

    Returns
    -------
    artifact : {"model": LGBMClassifier, "features": list[str]}
    """
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        log.error("lightgbm chưa được cài. Chạy: pip install lightgbm")
        sys.exit(1)

    df = build_features(db)

    X = df[FEATURES].values
    y = df["label"].values

    # Class imbalance: positive thường ít hơn negative
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale = n_neg / max(n_pos, 1)
    log.info(f"Class balance: pos={n_pos:,} / neg={n_neg:,}  scale_pos_weight={scale:.1f}")

    model = LGBMClassifier(
        n_estimators      = 300,
        learning_rate     = 0.05,
        num_leaves        = 31,
        max_depth         = -1,
        scale_pos_weight  = scale,    # xử lý class imbalance
        random_state      = 42,
        n_jobs            = -1,
        verbose           = -1,
    )
    model.fit(X, y)

    # Feature importance
    importances = dict(zip(FEATURES, model.feature_importances_))
    log.info("Feature importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        log.info(f"  {feat:<30} {imp:>6}")

    artifact = {"model": model, "features": FEATURES}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f, protocol=4)
    log.info(f"LTR model saved → {MODEL_PATH}")

    return artifact


# ══════════════════════════════════════════════════════════════════════
# 3. LOAD (dùng bởi pipeline.py)
# ══════════════════════════════════════════════════════════════════════
def load() -> dict | None:
    """Trả về artifact hoặc None nếu chưa train."""
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ══════════════════════════════════════════════════════════════════════
# 4. QUICK EVAL (in-sample, dùng để sanity check sau train)
# ══════════════════════════════════════════════════════════════════════
def quick_eval(artifact: dict, db) -> None:
    """In AUC và accuracy trên toàn bộ training data (in-sample, chỉ để verify)."""
    try:
        from sklearn.metrics import roc_auc_score, accuracy_score
    except ImportError:
        log.warning("sklearn không có — bỏ qua eval")
        return

    df    = build_features(db)
    X     = df[FEATURES].values
    y     = df["label"].values
    model = artifact["model"]

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    auc      = roc_auc_score(y, probs)
    accuracy = accuracy_score(y, preds)
    # dùng WARNING thay vì INFO để tránh nhầm đây là offline validation thực
    log.warning(f"In-sample AUC={auc:.4f}  Accuracy={accuracy:.4f}")
    log.warning("(In-sample — KHÔNG phải out-of-sample. Cần offline eval riêng để đánh giá tổng quát hoá)")


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os as _os
    sys.path.insert(0, str(BASE_DIR))

    from dotenv import load_dotenv
    load_dotenv()

    from db.connection import get_db
    db = get_db()

    artifact = train(db)
    quick_eval(artifact, db)

    print(f"\n LTR model sẵn sàng → {MODEL_PATH}")
    print("   pipeline.py sẽ tự load khi khởi động lại recommend_api.")
    print("   Để reload mà không restart: POST /admin/reload")