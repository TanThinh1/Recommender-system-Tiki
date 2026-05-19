"""
crawler/crawl_reviews.py — Crawl reviews + mô tả sản phẩm từ MongoDB (không dùng JSON file).
"""

import asyncio
from playwright.async_api import async_playwright

from seed_refactor.config import HEADLESS, MAX_REVIEW_PAGES_PER_PROD, log, C_GREEN, C_RED
from seed_refactor.crawler.tiki_crawler import TikiAPICrawler, crawl_reviews_for_product
from seed_refactor.utils.progress import tqdm


async def main():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from seed_refactor.db.connection import get_db
    from pymongo import UpdateOne

    db = get_db()

    products = list(db.products.find({}, {"_id": 0}))
    if not products:
        print("❌ Không có products trong MongoDB. Chạy product.py trước.")
        return

    all_reviews   = []   # flat list — mỗi review là 1 doc có product_id
    desc_fetched  = 0
    desc_failed   = 0
    product_updates = []  # bulk update descriptions

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=HEADLESS)
        ctx = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        )
        page = await ctx.new_page()
        await page.goto("https://tiki.vn")
        await page.wait_for_timeout(2000)

        crawler = TikiAPICrawler()
        crawler.page = page

        crawlable = [p for p in products if p.get("tiki_product_id")]
        pbar = tqdm(crawlable, desc="📝 Crawl reviews + mô tả", unit="SP", ncols=90)

        for prod in pbar:
            tiki_pid   = prod["tiki_product_id"]
            product_id = prod["product_id"]

            # ── 1. Crawl reviews ─────────────────────────────────────
            reviews = await crawl_reviews_for_product(page, tiki_pid, MAX_REVIEW_PAGES_PER_PROD)
            for rv in reviews:
                rv["product_id"] = product_id   # thêm product_id vào mỗi review doc

            all_reviews.extend(reviews)

            if reviews:
                avg = sum(r["rating"] for r in reviews) / len(reviews)
                product_updates.append(UpdateOne(
                    {"product_id": product_id},
                    {"$set": {"rating_avg": round(avg, 1)}},
                ))

            # ── 2. Fetch mô tả đầy đủ ────────────────────────────────
            detail = await crawler.fetch_product_detail(page, tiki_pid)
            if detail.get("description"):
                product_updates.append(UpdateOne(
                    {"product_id": product_id},
                    {"$set": {
                        "description"   : detail["description"],
                        "specifications": detail.get("specifications", []),
                    }},
                ))
                desc_fetched += 1
            else:
                desc_failed += 1

            await asyncio.sleep(0.3)

            pbar.set_postfix_str(
                f"rev={len(all_reviews):,} "
                f"desc={desc_fetched} fail={desc_failed}"
            )

        await browser.close()

    # ── Lưu reviews vào MongoDB ───────────────────────────────────────
    # Mỗi review là 1 document với product_id, reviewer_id, rating, ...
    db.reviews.drop()
    if all_reviews:
        db.reviews.insert_many(all_reviews, ordered=False)
        db.reviews.create_index([("product_id", 1)])
        db.reviews.create_index([("product_id", 1), ("reviewer_id", 1)])
        db.reviews.create_index([("rating", -1)])

    # ── Cập nhật description + rating_avg vào products ───────────────
    if product_updates:
        from pymongo import UpdateOne as _UO
        db.products.bulk_write(product_updates, ordered=False)

    total_reviews = len(all_reviews)
    print(f"\n✅ Reviews: {total_reviews:,} → MongoDB (reviews)")
    print(f"✅ Mô tả: {desc_fetched:,} SP có mô tả / {desc_failed:,} SP không có")
    print(f"✅ Products cập nhật description + rating_avg → MongoDB")
    print(f"   Bước tiếp theo: python -m seed_refactor.processing.user")


if __name__ == "__main__":
    asyncio.run(main())