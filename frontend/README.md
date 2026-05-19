# 🛍 Tiki Recommendation System — Frontend Dashboard

Giao diện web quản trị cho hệ thống gợi ý sản phẩm Tiki, xây dựng trên **Node.js + Express**.

---

## 📐 Kiến trúc

```
Browser
  ↓ HTTP
Node.js Express Server  (port 3000)
  ├── /api/products/*     → MongoDB trực tiếp
  ├── /api/stats          → MongoDB trực tiếp
  ├── /api/interactions   → MongoDB + proxy → FastAPI /feedback
  ├── /api/recommend/user/:id    → proxy → FastAPI /hybrid/:id
  └── /api/recommend/similar/:id → proxy → FastAPI /similar/:id

FastAPI ML Server  (port 8000)
  ├── /hybrid/:user_id    ALS + FAISS hybrid
  ├── /similar/:product_id FAISS content-based
  └── /health
```

---

## 🚀 Cài đặt & Chạy

### 1. Cài dependencies

```bash
cd tiki-frontend
npm install
```

### 2. Cấu hình `.env`

```env
PORT=3000
MONGO_URI=mongodb://localhost:27017
DB_NAME=tiki_recommendation
ML_API_URL=http://localhost:8000
```

### 3. Khởi động FastAPI (nếu chưa chạy)

```bash
cd seed_refactor
uvicorn recommend_api:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Khởi động Node.js server

```bash
# Production
npm start

# Development (auto-reload)
npm run dev
```

Mở trình duyệt: **http://localhost:3000**

---

## 📋 Use Cases

| UC | Tên | Route | Mô tả |
|----|-----|-------|-------|
| UC-01 | Xem danh sách sản phẩm | `GET /api/products` | Phân trang, lọc danh mục, tìm kiếm, sắp xếp |
| UC-02 | Xem chi tiết sản phẩm | `GET /api/products/:id` | Thông tin đầy đủ + reviews gần đây |
| UC-03 | Gợi ý cá nhân hóa | `GET /api/recommend/user/:id` | Hybrid ALS+FAISS, alpha động, enriched với metadata |
| UC-04 | Sản phẩm tương tự | `GET /api/recommend/similar/:id` | FAISS cosine similarity, enriched |
| UC-05 | Ghi nhận tương tác | `POST /api/interactions` | Upsert theo 24h window + forward sang ML API |
| UC-06 | Thống kê hệ thống | `GET /api/stats` | Totals, top-5, categories, ML health |

---

## 🔌 API Reference

### GET /api/products
```
Query params:
  page     int    (default: 1)
  limit    int    (default: 20, max: 50)
  category string (filter theo danh mục)
  search   string (tìm theo tên / brand)
  sort_by  string (popularity_score | price_asc | price_desc | rating | name)

Response:
  { products: [...], total, page, limit, total_pages }
```

### GET /api/products/:id
```
Response:
  { product_id, name, price, rating, stock, category, brand, ... reviews: [...] }
```

### GET /api/recommend/user/:id
```
Query params:
  n     int   (default: 10)
  alpha float (0–1, trọng số ALS trong hybrid, default: 0.7)

Response:
  { user_id, is_cold, latency_ms, alpha, total, items: [...] }
```

### GET /api/recommend/similar/:id
```
Query params:
  n int (default: 10)

Response:
  { product_id, latency_ms, total, items: [...] }
```

### POST /api/interactions
```json
{
  "user_id":    "user_abc",
  "product_id": "prod_xyz",
  "action":     "view | click | add_to_cart | purchase | review",
  "rating":     4.5,
  "source":     "dashboard"
}
```

### GET /api/stats
```
Response:
  { totals, categories, top_products, interactions, ml_api, generated_at }
```

---

## 🗂 Cấu trúc thư mục

```
tiki-frontend/
├── server.js          Express server + tất cả API routes
├── package.json
├── .env               Cấu hình (MONGO_URI, ML_API_URL, PORT)
├── README.md
└── public/
    └── index.html     Single-page frontend (HTML/CSS/JS thuần)
```

---

## 🛠 Troubleshooting

**MongoDB không kết nối được**
```bash
# Kiểm tra MongoDB đang chạy
mongosh --eval "db.runCommand({ping:1})"
```

**ML API offline**
- Trạng thái hiển thị trên sidebar (đỏ = offline)
- Các tính năng recommendation sẽ trả lỗi 503
- Phần còn lại (products, stats, interactions) vẫn hoạt động bình thường

**Không thấy sản phẩm**
```bash
# Kiểm tra dữ liệu trong DB
mongosh tiki_recommendation --eval "db.products.countDocuments()"
```
