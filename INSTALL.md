# 🛠️ Hướng dẫn cài đặt

## Yêu cầu hệ thống

| Thành phần | Phiên bản tối thiểu |
|---|---|
| Python | 3.10+ |
| Node.js | 18+ |
| MongoDB | 6.0+ |
| RAM | 4 GB (khuyến nghị 8 GB nếu dataset lớn) |
| Disk | 2 GB trống (cho model artifacts + FAISS index) |

---

## Bước 1 — Cài MongoDB

### macOS (Homebrew)
```bash
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community
```

### Ubuntu / Debian
```bash
curl -fsSL https://pgp.mongodb.com/server-7.0.asc | sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
sudo apt-get update && sudo apt-get install -y mongodb-org
sudo systemctl start mongod && sudo systemctl enable mongod
```

### Windows
Tải installer từ https://www.mongodb.com/try/download/community và cài đặt theo hướng dẫn.

### Kiểm tra MongoDB đang chạy
```bash
mongosh --eval "db.runCommand({ connectionStatus: 1 })"
```

---

## Bước 2 — Tạo Python virtual environment

```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt (Linux / macOS)
source .venv/bin/activate

# Kích hoạt (Windows)
.venv\Scripts\activate
```

---

## Bước 3 — Cài Python dependencies

```bash
pip install --upgrade pip

# Core crawler
pip install playwright pymongo python-dotenv tqdm

# Playwright browser engine
playwright install chromium

# ML & embeddings
pip install implicit faiss-cpu sentence-transformers

# API serving
pip install fastapi uvicorn

# LTR model (tuỳ chọn — cần để dùng /ltr_model.py)
pip install lightgbm scikit-learn

# Numpy / Scipy (thường đã có theo implicit)
pip install numpy scipy pandas
```

> **Lưu ý GPU**: nếu máy có GPU và muốn tăng tốc FAISS, cài `faiss-gpu` thay vì `faiss-cpu`. Với `implicit`, cài thêm `cudatoolkit` theo hướng dẫn tại https://github.com/benfred/implicit

---

## Bước 4 — Cài Node.js dependencies

```bash
npm install
```

Package `package.json` đã khai báo sẵn: `express`, `axios`, `cors`, `dotenv`, `mongodb`, `http-proxy-middleware`.

---

## Bước 5 — Tạo file `.env`

Tạo file `.env` ở thư mục gốc (cùng cấp với `recommend_api.py`):

```env
# ── MongoDB ──────────────────────────────────────────────
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=tiki_recommendation

# ── Node.js Frontend ─────────────────────────────────────
PORT=3000
MONGO_URI=mongodb://localhost:27017
DB_NAME=tiki_recommendation

# ── Python API ───────────────────────────────────────────
ML_API_URL=http://localhost:8000

# ── Bảo mật hot-reload ───────────────────────────────────
RELOAD_SECRET=change-me-in-production
```

> `MONGODB_URI` dùng cho Python (`recommend_api.py`, `connection.py`).
> `MONGO_URI` dùng cho Node.js (`server.js`).
> Hai biến trỏ cùng một MongoDB instance.

---

## Bước 6 — Kiểm tra cài đặt

```bash
# Kiểm tra Python packages
python -c "import pymongo, implicit, faiss, sentence_transformers, fastapi; print('✅ Python OK')"

# Kiểm tra Playwright
python -c "from playwright.sync_api import sync_playwright; print('✅ Playwright OK')"

# Kiểm tra Node.js
node -e "require('express'); require('mongodb'); console.log('✅ Node.js OK')"

# Kiểm tra MongoDB
python -c "from pymongo import MongoClient; MongoClient('mongodb://localhost:27017').admin.command('ping'); print('✅ MongoDB OK')"
```

---

## Cấu trúc thư mục sau khi cài

```
project/
├── .venv/                  # Python virtual environment
├── .env                    # Biến môi trường (tự tạo ở Bước 5)
├── node_modules/           # Node.js packages (sau npm install)
├── seed_refactor/          # Python package chính
├── recommend_api.py
├── pipeline.py
├── train_model.py
├── generate_embeddings.py
├── ltr_model.py
├── server.js
├── index.html
└── package.json
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'seed_refactor'`**
```bash
# Chạy từ thư mục gốc của project, không phải từ bên trong seed_refactor/
cd /path/to/project
python -m seed_refactor.crawler.tiki_crawler
```

**`playwright._impl._errors.Error: Executable doesn't exist`**
```bash
playwright install chromium
```

**`faiss.swigfaiss.FaissException` hoặc lỗi FAISS trên Apple Silicon**
```bash
# Trên M1/M2/M3, cài conda version
conda install -c conda-forge faiss-cpu
```

**MongoDB `Authentication failed`**
Nếu MongoDB yêu cầu xác thực, cập nhật URI:
```env
MONGODB_URI=mongodb://username:password@localhost:27017/tiki_recommendation
```

**`implicit` báo lỗi `BLAS`**
```bash
pip install implicit[cpu]
# hoặc
conda install -c conda-forge implicit
```
