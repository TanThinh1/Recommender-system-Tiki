# ▶️ Hướng dẫn chạy tuần tự

Toàn bộ pipeline gồm **3 giai đoạn** chạy theo thứ tự. Mỗi bước phụ thuộc vào kết quả của bước trước — **không được bỏ qua hay đảo thứ tự**.

```
[Giai đoạn 1] Thu thập dữ liệu  →  [Giai đoạn 2] Train model  →  [Giai đoạn 3] Khởi động server
```

> Trước khi chạy: đảm bảo MongoDB đang chạy và `.env` đã được tạo theo [INSTALL.md](./INSTALL.md).
> Kích hoạt virtual environment: `source .venv/bin/activate` (Linux/macOS) hoặc `.venv\Scripts\activate` (Windows).

---

## ⏱️ Thời gian ước tính

| Bước | Thời gian | Ghi chú |
|---|---|---|
| Crawl sản phẩm | 5–15 phút | Phụ thuộc số category & tốc độ mạng |
| Parse sản phẩm | < 1 phút | — |
| Crawl reviews + mô tả | 15–60 phút | Phụ thuộc số sản phẩm |
| Xây users & interactions | 1–5 phút | — |
| Generate embeddings | 5–20 phút | Nhanh hơn nếu có GPU |
| Train ALS | 2–10 phút | — |
| Train LTR | 1–3 phút | — |

---

## 🗄️ Giai đoạn 1 — Thu thập & xây dựng dữ liệu

Chạy từ **thư mục gốc** của project.

### Bước 1.1 — Crawl sản phẩm thô từ Tiki
```bash
python -m seed_refactor.crawler.tiki_crawler
```
Kết quả: lưu sản phẩm thô vào MongoDB collection `raw_products`.

Kiểm tra:
```bash
python -c "
from seed_refactor.db.connection import get_db
db = get_db()
print('raw_products:', db.raw_products.count_documents({}))
"
```

---

### Bước 1.2 — Parse & làm giàu sản phẩm
```bash
python -m seed_refactor.processing.product
```
Kết quả: collection `products` với đầy đủ fields (price_range, tags, popularity_score, …).

---

### Bước 1.3 — Crawl reviews và mô tả đầy đủ
```bash
python -m seed_refactor.crawler.crawl_reviews
```
Kết quả: collection `reviews` + cập nhật `description`, `rating_avg` vào `products`.

> Bước này dùng Playwright mở trình duyệt để gọi Tiki API. Nếu muốn thấy trình duyệt hoạt động, đặt `HEADLESS=false` trong `.env`.

---

### Bước 1.4 — Xây dựng user profiles
```bash
python -m seed_refactor.processing.user
```
Kết quả: collection `users` với `activity_level`, `spending_level`, `preference_cats`.

---

### Bước 1.5 — Sinh interaction log
```bash
python -m seed_refactor.processing.interaction
```
Kết quả: collection `interactions` với các action `view`, `add_to_cart`, `purchase` (real + synthetic).

---

### ✅ Kiểm tra toàn bộ dữ liệu sau Giai đoạn 1
```bash
python -m seed_refactor.db.storage
```
Output mẫu:
```
📊 Thống kê collections:
   raw_products              1,500 docs
   products                  1,200 docs
   reviews                   8,400 docs
   users                     2,100 docs
   interactions             45,000 docs

✅ interactions.timestamp type: datetime
✅ Indexes đã được tạo/cập nhật
```

---

## 🤖 Giai đoạn 2 — Train model

### Bước 2.1 — Generate item embeddings + FAISS index
```bash
python generate_embeddings.py
```
Kết quả:
- `faiss_index.bin` — vector index (IndexFlatIP hoặc IndexIVFFlat)
- `faiss_id_map.json` — mapping FAISS index ↔ product_id
- `faiss_meta.json` — metadata index
- Cập nhật field `item_embedding` trong MongoDB `products`

---

### Bước 2.2 — Train ALS model
```bash
python train_model.py
```
Kết quả:
- `als_model.pkl` — ALS model + user/product mappings + sparse matrix
- `als_meta.json` — metrics (Precision@5, Recall@5, NDCG@5, HitRate@5)

Output mẫu sau training:
```
[12:34:56] ✅ Train done  factors=128  alpha=120  reg=0.1
[12:34:57]    Precision@5 = 0.1823
[12:34:57]    Recall@5    = 0.0941
[12:34:57]    NDCG@5      = 0.1654
[12:34:57]    HitRate@5   = 0.4210
```

---

### Bước 2.3 — Train LTR re-ranker (tuỳ chọn nhưng khuyến nghị)
```bash
python ltr_model.py
```
Kết quả: `ltr_model.pkl` — LightGBM classifier, tự động được `pipeline.py` load khi khởi động API.

---

### (Tuỳ chọn) Hyperparameter tuning cho ALS
```bash
python grid_search.py
```
Tìm bộ `factors`, `alpha`, `regularization` tối ưu. Kết quả ghi ra `grid_search_results.json`.

---

### (Tuỳ chọn) So sánh các phương pháp gợi ý
```bash
python model_comparison.py
```
So sánh Pure CF vs Pure CBF vs Hybrid trên cùng train/test split. Kết quả ghi ra `comparison_results.json`.

---

## 🚀 Giai đoạn 3 — Khởi động server

### Bước 3.1 — Khởi động Python API (FastAPI)

**Terminal 1:**
```bash
uvicorn recommend_api:app --host 0.0.0.0 --port 8000 --reload
```

Kiểm tra API đang chạy:
```bash
curl http://localhost:8000/health
```

Output mẫu:
```json
{
  "status": "ok",
  "als_users": 2100,
  "als_products": 1200,
  "faiss_vectors": 1200,
  "popular_count": 50
}
```

---

### Bước 3.2 — Khởi động Node.js frontend

**Terminal 2:**
```bash
node server.js
```

hoặc dùng nodemon để tự reload khi thay đổi code:
```bash
npx nodemon server.js
```

Mở trình duyệt: **http://localhost:3000**

---

## 🔁 Sau khi retrain model

Nếu đã retrain mà không muốn restart API server, dùng hot-reload:

```bash
curl -X POST "http://localhost:8000/admin/reload?secret=your-reload-secret"
```

Thay `your-reload-secret` bằng giá trị `RELOAD_SECRET` trong `.env`.

---

## 🧪 Kiểm tra nhanh các endpoint

```bash
# Gợi ý cho user
curl "http://localhost:8000/recommend/USER_ID_HERE?n=5"

# Sản phẩm tương tự
curl "http://localhost:8000/similar/SP00001?n=5"

# Hybrid (ALS + FAISS)
curl "http://localhost:8000/hybrid/USER_ID_HERE?n=5&als_weight=0.7"

# Online metrics (7 ngày gần nhất)
curl "http://localhost:8000/metrics/online?days=7"

# Ghi nhận tương tác
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"user_id":"U001","product_id":"SP00001","action":"purchase","weight":1.0}'
```

---

## 📋 Tóm tắt lệnh (copy-paste toàn bộ)

```bash
# ── Giai đoạn 1: Dữ liệu ──
python -m seed_refactor.crawler.tiki_crawler
python -m seed_refactor.processing.product
python -m seed_refactor.crawler.crawl_reviews
python -m seed_refactor.processing.user
python -m seed_refactor.processing.interaction

# ── Kiểm tra ──
python -m seed_refactor.db.storage

# ── Giai đoạn 2: Model ──
python generate_embeddings.py
python train_model.py
python ltr_model.py

# ── Giai đoạn 3: Server (mỗi lệnh chạy trong terminal riêng) ──
uvicorn recommend_api:app --host 0.0.0.0 --port 8000 --reload
node server.js
```
