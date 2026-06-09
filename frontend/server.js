require("dotenv").config();
const express = require("express");
const cors    = require("cors");
const axios   = require("axios");
const path    = require("path");
const { MongoClient } = require("mongodb");

const app = express();

// ── Config ──────────────────────────────────────────────────────────
const PORT       = process.env.PORT        || 3000;
const MONGO_URI  = process.env.MONGO_URI   || "mongodb://localhost:27017";
const DB_NAME    = process.env.DB_NAME     || "tiki_recommendation";
const ML_API     = (process.env.ML_API_URL || "http://localhost:8000").replace(/\/$/, "");
const ML_API_KEY = process.env.API_KEY     || "";

// Debug: in ra để xác nhận .env được đọc đúng
console.log(`[config] ML_API     = ${ML_API}`);
console.log(`[config] API_KEY    = ${ML_API_KEY ? ML_API_KEY.slice(0,8) + "..." : "⚠️  CHƯA SET — kiểm tra .env"}`);


// ── Escape regex metacharacters để phòng ReDoS ──────────────────────
// Áp dụng cho mọi input từ query string trước khi dùng $regex trong MongoDB.
function escapeRegex(str) {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// ── Middleware ───────────────────────────────────────────────────────
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

// ── MongoDB Connection (lazy singleton với pool + reconnect) ─────────
let _client = null;
let _connecting = false;

async function getDb() {
  if (_client) return _client.db(DB_NAME);

  // Tránh tạo nhiều connection song song khi cold-start
  if (_connecting) {
    await new Promise(resolve => setTimeout(resolve, 100));
    return getDb();
  }

  _connecting = true;
  try {
    _client = new MongoClient(MONGO_URI, {
      serverSelectionTimeoutMS : 5_000,
      connectTimeoutMS         : 5_000,
      socketTimeoutMS          : 30_000,
      // pool config — tránh connection leak dưới tải
      minPoolSize              : 2,
      maxPoolSize              : 20,
      maxIdleTimeMS            : 60_000,
      waitQueueTimeoutMS       : 5_000,
    });
    await _client.connect();
    console.log(" MongoDB connected →", MONGO_URI);

    // tự reset client khi server đóng connection
    _client.on("serverClosed", () => {
      console.warn(" MongoDB serverClosed — sẽ reconnect ở request tiếp theo");
      _client = null;
    });
    _client.on("error", (err) => {
      console.error(" MongoDB error:", err.message);
    });
  } finally {
    _connecting = false;
  }

  return _client.db(DB_NAME);
}

// ── Validate query param n (1–50, default 10) ───────────────────────
// server.js không được dựa vào ML API để bắt n không hợp lệ.
// Validate độc lập để fail-fast và tránh coupling với schema ML API.
function parseN(raw, defaultVal = 10) {
  const n = parseInt(raw || String(defaultVal), 10);
  if (isNaN(n) || n < 1) return 1;
  if (n > 50)            return 50;
  return n;
}
async function mlGet(path, params = {}) {
  const res = await axios.get(`${ML_API}${path}`, {
    params,
    timeout: 15000,
    headers: { "X-API-Key": ML_API_KEY },
  });
  return res.data;
}

// ══════════════════════════════════════════════════════════════════════
// UC-01  GET /api/products — Danh sách sản phẩm (phân trang + lọc)
// ══════════════════════════════════════════════════════════════════════
app.get("/api/products", async (req, res) => {
  try {
    const db       = await getDb();
    const page     = Math.max(1, parseInt(req.query.page  || "1"));
    const limit    = Math.min(50, Math.max(1, parseInt(req.query.limit || "20")));
    const category = req.query.category || null;
    const rawSearch = req.query.search   || null;
    const search    = rawSearch ? rawSearch.trim().slice(0, 100) : null;  // cap 100 ký tự
    const sort_by  = req.query.sort_by  || "popularity_score";

    // Build filter
    const filter = {};
    if (category) filter.category = { $regex: `^${escapeRegex(category)}$`, $options: "i" };
    if (search)   filter.$or = [
      { name:  { $regex: escapeRegex(search), $options: "i" } },
      { brand: { $regex: escapeRegex(search), $options: "i" } },
    ];

    // Sort map
    const sortMap = {
      popularity_score : { popularity_score: -1 },
      price_asc        : { price: 1 },
      price_desc       : { price: -1 },
      rating           : { rating: -1 },
      name             : { name: 1 },
    };
    const sortObj = sortMap[sort_by] || { popularity_score: -1 };

    const [products, total] = await Promise.all([
      db.collection("products")
        .find(filter)
        .sort(sortObj)
        .skip((page - 1) * limit)
        .limit(limit)
        .project({ _id: 0 })
        .toArray(),
      db.collection("products").countDocuments(filter),
    ]);

    res.json({
      products,
      total,
      page,
      limit,
      total_pages: Math.ceil(total / limit),
    });
  } catch (err) {
    console.error("[UC-01]", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ══════════════════════════════════════════════════════════════════════
// UC-02  GET /api/products/:id — Chi tiết sản phẩm
// ══════════════════════════════════════════════════════════════════════
app.get("/api/products/:id", async (req, res) => {
  try {
    const db      = await getDb();
    const product = await db.collection("products").findOne(
      { product_id: req.params.id },
      { projection: { _id: 0 } }
    );

    if (!product) return res.status(404).json({ error: "Không tìm thấy sản phẩm" });

    // Lấy reviews có review_text, deduplicate theo user (giữ mới nhất mỗi user)
    const reviews = await db.collection("interactions").aggregate([
      { $match: {
          product_id : req.params.id,
          review_text: { $exists: true, $ne: "" },
      }},
      { $sort: { timestamp: -1 } },
      { $group: { _id: "$user_id", doc: { $first: "$$ROOT" } } },
      { $replaceRoot: { newRoot: "$doc" } },
      { $sort: { timestamp: -1 } },
      { $limit: 15 },
      { $project: { _id: 0, user_id: 1, rating: 1, review_text: 1, sentiment: 1, timestamp: 1 } },
    ]).toArray();

    // Lookup tên user cho reviews
    const reviewUserIds = [...new Set(reviews.map(r => r.user_id))];
    const userDocs = reviewUserIds.length > 0
      ? await db.collection("users")
          .find({ user_id: { $in: reviewUserIds } })
          .project({ _id: 0, user_id: 1, name: 1 })
          .toArray()
      : [];
    const userNameMap = Object.fromEntries(userDocs.map(u => [u.user_id, u.name]));
    const reviewsWithName = reviews.map(r => ({
      ...r,
      user_name: userNameMap[r.user_id] || null,
    }));

    res.json({ ...product, reviews: reviewsWithName });
  } catch (err) {
    console.error("[UC-02]", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ══════════════════════════════════════════════════════════════════════
// UC-03  GET /api/recommend/user/:id — Gợi ý cá nhân hoá (Hybrid)
// ══════════════════════════════════════════════════════════════════════
app.get("/api/recommend/user/:id", async (req, res) => {
  try {
    const { id }   = req.params;
    const n        = parseN(req.query.n);
    const alpha    = Math.min(1, Math.max(0, parseFloat(req.query.alpha || "0.7")));
    const debug    = req.query.debug === "true";

    // Gọi hybrid endpoint (ALS + FAISS)
    const data = await mlGet(`/hybrid/${encodeURIComponent(id)}`, {
      n,
      als_weight   : alpha,
      filter_owned : true,
    });

    // Enrich với product metadata từ MongoDB
    const db  = await getDb();
    const ids = (data.items || []).map(i => i.product_id);
    let metaMap = {};
    if (ids.length > 0) {
      const docs = await db.collection("products")
        .find({ product_id: { $in: ids } })
        .project({ _id: 0, product_id: 1, name: 1, price: 1, rating_avg: 1, rating: 1,
                   category: 1, brand: 1, image_url: 1, stock: 1 })
        .toArray();
      metaMap = Object.fromEntries(docs.map(d => [d.product_id, d]));
    }

    const enriched = (data.items || []).map(item => ({
      ...item,
      ...(metaMap[item.product_id] || {}),
    }));

    res.json({
      user_id     : id,
      is_cold     : data.is_cold,
      latency_ms  : data.latency_ms,
      alpha       : alpha,
      total       : enriched.length,
      items       : enriched,
    });
  } catch (err) {
    const status = err.response?.status || 500;
    const detail = err.response?.data?.detail || err.response?.data?.error || err.message;
    console.error(`[UC-03] HTTP ${status} →`, detail);
    res.status(status).json({ error: detail });
  }
});

// ══════════════════════════════════════════════════════════════════════
// UC-04  GET /api/recommend/similar/:id — Sản phẩm tương tự
// ══════════════════════════════════════════════════════════════════════
app.get("/api/recommend/similar/:id", async (req, res) => {
  try {
    const { id }  = req.params;
    const n       = parseN(req.query.n);

    const data = await mlGet(`/similar/${encodeURIComponent(id)}`, { n });

    // Enrich metadata — /similar trả về key "similar" (không phải "items")
    const db  = await getDb();
    const similarItems = data.similar || data.items || [];
    const ids = similarItems.map(i => i.product_id);
    let metaMap = {};
    if (ids.length > 0) {
      const docs = await db.collection("products")
        .find({ product_id: { $in: ids } })
        .project({ _id: 0, product_id: 1, name: 1, price: 1, rating_avg: 1, rating: 1,
                   category: 1, brand: 1, image_url: 1 })
        .toArray();
      metaMap = Object.fromEntries(docs.map(d => [d.product_id, d]));
    }

    const enriched = similarItems.map(item => ({
      ...item,
      ...(metaMap[item.product_id] || {}),
    }));

    res.json({
      product_id  : id,
      latency_ms  : data.latency_ms,
      total       : enriched.length,
      items       : enriched,
    });
  } catch (err) {
    const status = err.response?.status || 500;
    const detail = err.response?.data?.detail || err.response?.data?.error || err.message;
    console.error(`[UC-04] HTTP ${status} →`, detail);
    res.status(status).json({ error: detail });
  }
});

// ══════════════════════════════════════════════════════════════════════
// UC-05  POST /api/interactions — Ghi nhận tương tác
// ══════════════════════════════════════════════════════════════════════
app.post("/api/interactions", async (req, res) => {
  try {
    const { user_id, product_id, action, rating, source } = req.body;

    if (!user_id || !product_id || !action) {
      return res.status(400).json({ error: "Thiếu user_id, product_id hoặc action" });
    }

    const VALID_ACTIONS = ["view", "click", "add_to_cart", "purchase", "review"];
    if (!VALID_ACTIONS.includes(action)) {
      return res.status(400).json({ error: `action phải là một trong: ${VALID_ACTIONS.join(", ")}` });
    }

    const db  = await getDb();
    const now = new Date();

    const doc = {
      user_id,
      product_id,
      action,
      source    : source || "frontend",
      timestamp : now,
      ...(rating !== undefined ? { rating: Number(rating) } : {}),
    };

    // Upsert: cập nhật nếu cùng user+product+action trong 24 giờ
    const windowStart = new Date(now - 86400_000);
    await db.collection("interactions").updateOne(
      { user_id, product_id, action, timestamp: { $gte: windowStart } },
      { $set: doc },
      { upsert: true }
    );

    // Gửi feedback sang ML API để log (non-blocking)
    const weightMap = { view: 0.1, click: 0.2, add_to_cart: 0.5, purchase: 1.0, review: 0.7 };
    mlGet && axios.post(`${ML_API}/feedback`, {
      user_id,
      product_id,
      action,
      weight: weightMap[action] || 0.3,
      source : source || "frontend",
    }, {
      headers: { "X-API-Key": ML_API_KEY },
    }).catch(() => {}); // ignore ML API errors

    res.json({ status: "ok", message: "Tương tác đã được ghi nhận", doc });
  } catch (err) {
    console.error("[UC-05]", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ══════════════════════════════════════════════════════════════════════
// UC-06  GET /api/stats — Thống kê hệ thống
// ══════════════════════════════════════════════════════════════════════
app.get("/api/stats", async (req, res) => {
  try {
    const db = await getDb();

    const [
      totalProducts,
      totalUsers,
      totalInteractions,
      categories,
      topProducts,
      interactionBreakdown,
      mlHealth,
    ] = await Promise.all([
      db.collection("products").countDocuments(),
      db.collection("users").countDocuments(),
      db.collection("interactions").countDocuments(),

      // Danh sách danh mục + số lượng
      db.collection("products").aggregate([
        { $group: { _id: "$category", count: { $sum: 1 } } },
        { $sort: { count: -1 } },
      ]).toArray(),

      // Top 5 sản phẩm theo popularity_score
      db.collection("products")
        .find({})
        .sort({ popularity_score: -1 })
        .limit(5)
        .project({ _id: 0, product_id: 1, name: 1, category: 1,
                   price: 1, rating: 1, popularity_score: 1 })
        .toArray(),

      // Breakdown tương tác theo action
      db.collection("interactions").aggregate([
        { $group: { _id: "$action", count: { $sum: 1 } } },
        { $sort: { count: -1 } },
      ]).toArray(),

      // Trạng thái ML API
      axios.get(`${ML_API}/health`, { timeout: 3000 })
        .then(r => r.data)
        .catch(() => ({ status: "offline" })),
    ]);

    res.json({
      totals: {
        products     : totalProducts,
        users        : totalUsers,
        interactions : totalInteractions,
      },
      categories  : categories.map(c => ({ name: c._id, count: c.count })),
      top_products: topProducts,
      interactions: interactionBreakdown.map(i => ({ action: i._id, count: i.count })),
      ml_api      : mlHealth,
      generated_at: new Date().toISOString(),
    });
  } catch (err) {
    console.error("[UC-06]", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ══════════════════════════════════════════════════════════════════════
// GET /api/users — Tìm kiếm người dùng theo tên (cho autocomplete)
// ══════════════════════════════════════════════════════════════════════
app.get("/api/users", async (req, res) => {
  try {
    const db     = await getDb();
    const rawSearch = req.query.search || null;
    const search    = rawSearch ? rawSearch.trim().slice(0, 100) : null;
    const limit  = Math.min(20, parseInt(req.query.limit || "20"));

    const filter = {};
    if (search) {
      const escaped = escapeRegex(search);
      filter.$or = [
        { name:    { $regex: escaped, $options: "i" } },
        { user_id: { $regex: escaped, $options: "i" } },
      ];
    }

    const users = await db.collection("users")
      .find(filter)
      .limit(limit)
      .project({ _id: 0, user_id: 1, name: 1 })
      .toArray();

    res.json({ users, total: users.length });
  } catch (err) {
    console.error("[users]", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ── Health check ─────────────────────────────────────────────────────
app.get("/api/health", (_, res) => res.json({ status: "ok", ts: new Date() }));

// ── SPA fallback ─────────────────────────────────────────────────────
app.get("*", (_, res) =>
  res.sendFile(path.join(__dirname, "public", "index.html"))
);

// ── Start ─────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n╔══════════════════════════════════════════════╗`);
  console.log(`║  🛍  Tiki Recommendation Frontend            ║`);
  console.log(`║  http://localhost:${PORT}                       ║`);
  console.log(`║  ML API → ${ML_API}               ║`);
  console.log(`╚══════════════════════════════════════════════╝\n`);
});

// ── Graceful shutdown ─────────────────────────────────────────────────
process.on("SIGINT",  () => { _client?.close(); process.exit(); });
process.on("SIGTERM", () => { _client?.close(); process.exit(); });