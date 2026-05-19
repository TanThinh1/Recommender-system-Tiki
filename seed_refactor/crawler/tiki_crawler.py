"""
crawler/tiki_crawler.py — TikiAPICrawler + crawl_reviews_for_product.
"""

import asyncio
import re
from datetime import datetime
from playwright.async_api import async_playwright, Page, Route

from seed_refactor.config import HEADLESS, MAX_PAGES_PER_CAT, MAX_REVIEW_PAGES_PER_PROD, log, C_DIM, C_RED, C_GREEN
from seed_refactor.utils.progress import tqdm


def _strip_html(html: str) -> str:
    """Bỏ HTML tags, decode entities cơ bản, chuẩn hoá whitespace."""
    if not html:
        return ""
    text = re.sub(r"<[^>]+>", " ", html)
    text = (text
            .replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            .replace("&nbsp;", " ").replace("&quot;", '"').replace("&#39;", "'"))
    return re.sub(r"\s+", " ", text).strip()


class TikiAPICrawler:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self._last_result = None

    async def start(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=HEADLESS)
        self.context = await self.browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
        )
        self.page = await self.context.new_page()
        await self.page.route("**/api/v2/products?**", self._intercept_products)
        await self.page.goto("https://tiki.vn", timeout=30_000)
        await self.page.wait_for_timeout(2000)
        log("✅ Trình duyệt đã sẵn sàng", C_GREEN)

    async def _intercept_products(self, route: Route):
        response = await route.fetch()
        try:
            body = await response.json()
        except Exception:
            body = {}
        self._last_result = {"url": response.url, "data": body}
        await route.continue_()

    async def fetch_category_products(self, catid: int, limit: int = 100) -> list[dict]:
        products = []
        page_num = 1
        while True:
            api_url = f"https://tiki.vn/api/v2/products?category={catid}&limit={limit}&page={page_num}"
            log(f"📡 {api_url}", C_DIM, 2)
            self._last_result = None
            for _attempt in range(3):
                await self.page.goto(api_url, wait_until="networkidle", timeout=30_000)
                await self.page.wait_for_timeout(800)
                if self._last_result:
                    break
                if _attempt < 2:
                    log(f"⚠️  Retry {_attempt+1}/2 (catid={catid} page={page_num})", C_RED, 2)
                    await self.page.wait_for_timeout(1200 * (_attempt + 1))
            result = self._last_result
            if not result:
                log(f"⚠️  Không nhận được response (catid={catid} page={page_num})", C_RED, 2)
                break
            resp_url = result["url"]
            if f"category={catid}&" not in resp_url and not resp_url.endswith(f"category={catid}"):
                log(f"⚠️  URL không khớp: {resp_url}", C_RED, 2)
                break
            data = result["data"]
            items = data.get("data", [])
            if not items:
                break
            products.extend(items)
            log(f"   page {page_num}: +{len(items)} sp  (tổng {len(products)})", C_DIM, 2)
            paging = data.get("paging", {})
            last_page = paging.get("last_page", page_num)
            if page_num >= last_page or page_num >= MAX_PAGES_PER_CAT:
                break
            page_num += 1
        return products

    async def fetch_product_detail(self, page: Page, tiki_product_id: str) -> dict:
        """
        Lấy chi tiết sản phẩm từ Tiki API — bao gồm mô tả đầy đủ và thông số kỹ thuật.

        Endpoint: GET /api/v2/products/{tiki_product_id}
        Trả về dict gồm:
          - description : str — mô tả dạng plain text (đã strip HTML), tối đa 3000 ký tự
          - specifications : list[dict] — thông số kỹ thuật [{name, attributes:[{code,name,value}]}]
          - short_description : str — mô tả ngắn (fallback nếu không có description dài)

        Nếu request thất bại → trả về dict rỗng (không raise exception).
        """
        url = f"https://tiki.vn/api/v2/products/{tiki_product_id}"
        try:
            resp = await page.request.get(url, timeout=20_000)
            if resp.status != 200:
                return {}
            data = await resp.json()

            # Mô tả đầy đủ (có thể chứa HTML)
            raw_desc = data.get("description") or data.get("short_description") or ""
            description = _strip_html(raw_desc)[:3000]   # giới hạn 3000 ký tự

            # Thông số kỹ thuật: list[{name: str, attributes: [{code, name, value}]}]
            specifications = data.get("specifications") or []

            return {
                "description"   : description,
                "specifications": specifications,
            }
        except Exception as e:
            log(f"⚠️  fetch_product_detail pid={tiki_product_id}: {e}", C_RED, 2)
            return {}

    async def close(self):
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()


async def crawl_reviews_for_product(
    page: Page,
    product_id: str,
    max_pages: int = MAX_REVIEW_PAGES_PER_PROD,
) -> list[dict]:
    reviews = []
    for page_num in range(1, max_pages + 1):
        url = (
            f"https://tiki.vn/api/v2/reviews?product_id={product_id}"
            f"&limit=20&page={page_num}&sort=score%7Cdesc,id%7Cdesc,stars%7Call"
        )
        try:
            resp = await page.request.get(url, timeout=30_000)
            if resp.status != 200:
                break
            data = await resp.json()
            items = data.get("data", [])
            if not items:
                break
            for r in items:
                rating = r.get("rating", 0)
                if not (1 <= rating <= 5):
                    continue
                review_text = (r.get("content") or "").strip()[:500]
                text_len = len(review_text)
                if rating <= 2:
                    is_buyer = text_len > 10
                elif rating <= 3:
                    is_buyer = text_len > 50
                else:
                    is_buyer = text_len > 20
                created_by   = r.get("created_by") or {}
                reviewer_id  = str(created_by.get("id", "") or "")
                reviewer_name = (created_by.get("name") or "").strip()

                # ✅ FIX: lưu timestamp dạng datetime object thay vì string
                # → MongoDB sẽ lưu đúng kiểu Date, train_model.py nhận diện được
                raw_ts = r.get("created_at", None)
                if isinstance(raw_ts, int):
                    ts = datetime.fromtimestamp(raw_ts)        # unix timestamp → datetime
                elif isinstance(raw_ts, str) and raw_ts:
                    try:
                        ts = datetime.fromisoformat(raw_ts)    # ISO string → datetime
                    except Exception:
                        ts = datetime.now()
                else:
                    ts = datetime.now()

                if not reviewer_id:
                    continue
                reviews.append({
                    "reviewer_id"  : reviewer_id,
                    "reviewer_name": reviewer_name,
                    "rating"       : rating,
                    "is_buyer"     : is_buyer,
                    "review_text"  : review_text,
                    "created_at"   : ts,                       # ✅ datetime object
                })
            await asyncio.sleep(0.4)
        except Exception as e:
            log(f"⚠️  Review error pid={product_id} p{page_num}: {e}", C_RED, 2)
            break
    return reviews
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self._last_result = None

    async def start(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=HEADLESS)
        self.context = await self.browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
        )
        self.page = await self.context.new_page()
        await self.page.route("**/api/v2/products?**", self._intercept_products)
        await self.page.goto("https://tiki.vn", timeout=30_000)
        await self.page.wait_for_timeout(2000)
        log("✅ Trình duyệt đã sẵn sàng", C_GREEN)

    async def _intercept_products(self, route: Route):
        response = await route.fetch()
        try:
            body = await response.json()
        except Exception:
            body = {}
        self._last_result = {"url": response.url, "data": body}
        await route.continue_()

    async def fetch_category_products(self, catid: int, limit: int = 100) -> list[dict]:
        products = []
        page_num = 1
        while True:
            api_url = f"https://tiki.vn/api/v2/products?category={catid}&limit={limit}&page={page_num}"
            log(f"📡 {api_url}", C_DIM, 2)
            self._last_result = None
            for _attempt in range(3):
                await self.page.goto(api_url, wait_until="networkidle", timeout=30_000)
                await self.page.wait_for_timeout(800)
                if self._last_result:
                    break
                if _attempt < 2:
                    log(f"⚠️  Retry {_attempt+1}/2 (catid={catid} page={page_num})", C_RED, 2)
                    await self.page.wait_for_timeout(1200 * (_attempt + 1))
            result = self._last_result
            if not result:
                log(f"⚠️  Không nhận được response (catid={catid} page={page_num})", C_RED, 2)
                break
            resp_url = result["url"]
            if f"category={catid}&" not in resp_url and not resp_url.endswith(f"category={catid}"):
                log(f"⚠️  URL không khớp: {resp_url}", C_RED, 2)
                break
            data = result["data"]
            items = data.get("data", [])
            if not items:
                break
            products.extend(items)
            log(f"   page {page_num}: +{len(items)} sp  (tổng {len(products)})", C_DIM, 2)
            paging = data.get("paging", {})
            last_page = paging.get("last_page", page_num)
            if page_num >= last_page or page_num >= MAX_PAGES_PER_CAT:
                break
            page_num += 1
        return products

    async def close(self):
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()


async def crawl_reviews_for_product(page: Page, product_id: str, max_pages: int = MAX_REVIEW_PAGES_PER_PROD) -> list[dict]:
    reviews = []
    for page_num in range(1, max_pages + 1):
        url = f"https://tiki.vn/api/v2/reviews?product_id={product_id}&limit=20&page={page_num}&sort=score%7Cdesc,id%7Cdesc,stars%7Call"
        try:
            resp = await page.request.get(url, timeout=30_000)
            if resp.status != 200:
                break
            data = await resp.json()
            items = data.get("data", [])
            if not items:
                break
            for r in items:
                rating = r.get("rating", 0)
                if not (1 <= rating <= 5):
                    continue
                review_text = (r.get("content") or "").strip()[:500]
                text_len = len(review_text)
                if rating <= 2:
                    is_buyer = text_len > 10
                elif rating <= 3:
                    is_buyer = text_len > 50
                else:
                    is_buyer = text_len > 20
                created_by = r.get("created_by") or {}
                reviewer_id = str(created_by.get("id", "") or "")
                reviewer_name = (created_by.get("name") or "").strip()
                raw_ts = r.get("created_at", None)
                if isinstance(raw_ts, int):
                    ts = datetime.fromtimestamp(raw_ts).isoformat()
                elif isinstance(raw_ts, str) and raw_ts:
                    ts = raw_ts
                else:
                    ts = datetime.now().isoformat()
                if not reviewer_id:
                    continue
                reviews.append({
                    "reviewer_id": reviewer_id,
                    "reviewer_name": reviewer_name,
                    "rating": rating,
                    "is_buyer": is_buyer,
                    "review_text": review_text,
                    "created_at": ts,
                })
            await asyncio.sleep(0.4)
        except Exception as e:
            log(f"⚠️  Review error pid={product_id} p{page_num}: {e}", C_RED, 2)
            break
    return reviews

# ... giữ nguyên các class và hàm hiện có, chỉ thêm vào cuối file:

if __name__ == "__main__":
    import asyncio
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from seed_refactor.config import CATEGORY_CONFIG, catid_from_url, log, C_RED, C_GREEN
    from seed_refactor.db.connection import get_db
    from seed_refactor.db.storage import bulk_upsert

    async def main_crawl():
        print("🕷️ Bắt đầu crawl sản phẩm thô → MongoDB...")
        crawler = TikiAPICrawler()
        await crawler.start()

        all_raw = []
        for cat_name, cfg in CATEGORY_CONFIG.items():
            for url in cfg["urls"]:
                try:
                    catid = catid_from_url(url)
                except ValueError as e:
                    log(f"Bỏ qua {url}: {e}", C_RED)
                    continue
                log(f"Crawl {cat_name} / catid={catid}")
                raw_list = await crawler.fetch_category_products(catid, limit=100)
                for raw in raw_list:
                    raw["category"] = cat_name
                    # raw_id = tiki product id dạng string, dùng làm upsert key
                    raw["raw_id"] = str(raw.get("id", ""))
                all_raw.extend(raw_list)

        await crawler.close()

        # Bỏ item không có id
        all_raw = [r for r in all_raw if r.get("raw_id")]

        db = get_db()
        db.raw_products.drop()   # xóa cũ để tránh stale data
        bulk_upsert(db.raw_products, all_raw, key_field="raw_id")
        db.raw_products.create_index("raw_id", unique=True)
        db.raw_products.create_index("category")

        print(f"✅ Đã lưu {len(all_raw):,} sản phẩm thô vào MongoDB (raw_products)")
        print(f"   Bước tiếp theo: python -m seed_refactor.processing.product")

    asyncio.run(main_crawl())