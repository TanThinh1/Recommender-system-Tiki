# 🛍️ Tiki Recommendation System

Hệ thống gợi ý sản phẩm end-to-end cho sàn thương mại điện tử Tiki, bao gồm toàn bộ pipeline từ thu thập dữ liệu đến triển khai API thời gian thực.

---

## 📐 Kiến trúc tổng quan

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA PIPELINE                               │
│                                                                     │
│  tiki_crawler.py  →  product.py  →  crawl_reviews.py               │
│       (crawl)         (parse)         (reviews + mô tả)            │
│                                           ↓                         │
│                              user.py  →  interaction.py             │
│                            (xây users)   (xây interactions)         │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         MODEL TRAINING                              │
│                                                                     │
│  generate_embeddings.py  →  train_model.py  →  ltr_model.py        │
│     (FAISS + vectors)        (ALS model)      (LightGBM re-rank)   │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         SERVING                                     │
│                                                                     │
│  recommend_api.py (FastAPI :8000)  ←→  server.js (Node.js :3000)   │
│  pipeline.py (post-processing)          index.html (Dashboard)     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Cấu trúc thư mục

```
project/
├── seed_refactor/
│   ├── crawler/
│   │   ├── tiki_crawler.py       # Crawl sản phẩm từ Tiki API (Playwright)
│   │   └── crawl_reviews.py      # Crawl reviews + mô tả sản phẩm
│   ├── processing/
│   │   ├── product.py            # Parse & làm giàu dữ liệu sản phẩm
│   │   ├── user.py               # Xây dựng user profiles từ reviews
│   │   └── interaction.py        # Sinh interaction log (real + synthetic)
│   ├── db/
│   │   ├── connection.py         # MongoDB singleton connection
│   │   └── storage.py            # Bulk upsert / safe insert helpers
│   └── utils/
│       └── progress.py           # tqdm wrapper với fallback
│
├── generate_embeddings.py        # Encode sản phẩm → vector → FAISS index
├── train_model.py                # Train ALS (Implicit Matrix Factorization)
├── ltr_model.py                  # Train LightGBM re-ranker (Learning-to-Rank)
├── grid_search.py                # Hyperparameter tuning cho ALS
├── model_comparison.py           # So sánh CF vs CBF vs Hybrid
│
├── recommend_api.py              # FastAPI — serving endpoints
├── pipeline.py                   # Post-processing: filter → enrich → re-rank
│
├── server.js                     # Node.js frontend server
├── index.html                    # Dashboard UI
├── package.json                  # Node.js dependencies
│
├── .env                          # Biến môi trường (xem INSTALL.md)
│
│   ── Artifacts sinh ra sau training ──
├── als_model.pkl                 # ALS model (implicit)
├── als_meta.json                 # Metadata + metrics
├── faiss_index.bin               # FAISS vector index
├── faiss_id_map.json             # product_id ↔ FAISS index mapping
├── faiss_meta.json               # Loại index (Flat / IVF) + config
└── ltr_model.pkl                 # LightGBM re-ranker
```

---

## 🧠 Các thuật toán sử dụng

| Thành phần | Thuật toán | Thư viện |
|---|---|---|
| Collaborative Filtering | ALS (Alternating Least Squares) | `implicit` |
| Content-Based Filtering | Sentence embeddings + FAISS cosine | `sentence-transformers`, `faiss` |
| Hybrid endpoint | Weighted ALS score + FAISS similarity | — |
| Re-ranking | LightGBM LTR (Learning-to-Rank) | `lightgbm` |
| Post-processing | Filter → Enrich → Re-score | `pipeline.py` |

### Các endpoint API chính

| Method | Endpoint | Mô tả |
|---|---|---|
| `GET` | `/recommend/{user_id}` | Gợi ý cá nhân hoá (ALS) |
| `GET` | `/similar/{product_id}` | Sản phẩm tương tự (FAISS) |
| `GET` | `/hybrid/{user_id}` | Kết hợp ALS + FAISS |
| `POST` | `/feedback` | Ghi nhận tương tác người dùng |
| `GET` | `/metrics/online` | CTR, Conversion Rate, Coverage |
| `GET` | `/health` | Trạng thái model |
| `POST` | `/admin/reload` | Hot-reload model không restart |

---

## 🗄️ MongoDB Collections

| Collection | Mô tả | Key field |
|---|---|---|
| `raw_products` | Dữ liệu thô từ Tiki API | `raw_id` |
| `products` | Sản phẩm đã parse & làm giàu | `product_id` |
| `reviews` | Reviews thật từ Tiki | `product_id` |
| `users` | User profiles từ review data | `user_id` |
| `interactions` | Interaction log (real + synthetic) | *(unique index 5 trường)* |

---

## ✨ Tính năng nổi bật

- **Cold-start handling** — fallback về popularity-based khi user mới
- **Temporal decay** — tương tác cũ nhận confidence thấp hơn khi train
- **Time-based train/test split** — tránh data leakage, phản ánh thực tế
- **LTR re-ranking** — LightGBM học từ interaction log thay vì linear combination cứng
- **Hot-reload model** — `POST /admin/reload` để cập nhật model không restart server
- **Online monitoring** — CTR, Conversion Rate, Coverage qua `/metrics/online`
- **IDF weighting** — giảm nhiễu từ sản phẩm cực phổ biến trong ma trận ALS
- **FAISS auto-detect** — dùng `IndexFlatIP` khi N ≤ 50K, `IndexIVFFlat` khi lớn hơn

---

## ⚙️ Biến môi trường

```env
# MongoDB
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=tiki_recommendation

# Node.js server
PORT=3000
ML_API_URL=http://localhost:8000

# API bảo mật
RELOAD_SECRET=change-me-in-production
```

---

## 📊 Metrics đánh giá model

Offline metrics (in-sample, chạy sau `train_model.py`):

- **Precision@5** — tỉ lệ gợi ý đúng trong top 5
- **Recall@5** — tỉ lệ sản phẩm liên quan được tìm thấy
- **NDCG@5** — chất lượng xếp hạng (Normalized Discounted Cumulative Gain)
- **HitRate@5** — tỉ lệ user có ít nhất 1 gợi ý đúng trong top 5

Composite metric = `0.4×P@5 + 0.3×R@5 + 0.2×NDCG@5 + 0.1×HitRate@5`

Online metrics (live, qua `/metrics/online`):

- **CTR** — Click-Through Rate
- **Conversion Rate** — tỉ lệ purchase / impression
- **Coverage** — % catalogue sản phẩm được gợi ý
