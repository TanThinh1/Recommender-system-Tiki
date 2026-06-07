from collections import Counter
from seed_refactor.config import C_BOLD, C_CYAN, C_RESET

def print_quality_report(products: list, users: list, interactions: list) -> dict:
    print(f"\n{C_BOLD}{C_CYAN}{'═'*60}{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}  📊 BÁO CÁO CHẤT LƯỢNG DỮ LIỆU{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}{'═'*60}{C_RESET}")

    with_pop = sum(1 for p in products if p.get("popularity_score",0)>0)
    print(f"\n🛍️  Sản phẩm: {len(products):,}")
    price_range_counts = Counter(p.get("price_range") for p in products)
    for pr in ["budget","mid","premium"]:
        cnt = price_range_counts.get(pr,0)
        print(f"     {pr}: {cnt:,} ({cnt/len(products)*100:.1f}%)" if products else f"     {pr}: 0")
    print(f"   Có popularity_score > 0: {with_pop:,}")

    with_pref = sum(1 for u in users if u.get("preference_cats"))
    multi_rev = sum(1 for u in users if u.get("review_count",0)>=2)
    print(f"\n👥 Users: {len(users):,}")
    if users:
        print(f"   Có preference_cats: {with_pref:,} ({with_pref/len(users)*100:.1f}%)")
    print(f"   Có ≥2 reviews (dùng được cho CF): {multi_rev:,}")
    activity_counts = Counter(u.get("activity_level") for u in users)
    for lvl in ["low","medium","high"]:
        print(f"     {lvl}: {activity_counts.get(lvl,0):,}")

    action_counts = Counter(i["action"] for i in interactions)
    purchases = action_counts.get("purchase",0)
    carts = action_counts.get("add_to_cart",0)
    views = action_counts.get("view",0)
    pos_senti = sum(1 for i in interactions if i.get("sentiment")=="positive")
    has_text = sum(1 for i in interactions if i.get("review_text"))
    unique_pairs = len({(i["user_id"], i["product_id"]) for i in interactions})
    sparsity_row = 1 - len(interactions)/(len(users)*len(products)) if users and products else 0
    sparsity_cf = 1 - unique_pairs/(len(users)*len(products)) if users and products else 0
    avg_weight = sum(i["weight"] for i in interactions)/len(interactions) if interactions else 0

    print(f"\n Interactions: {len(interactions):,}")
    print(f"   purchase    : {purchases:,} ({purchases/len(interactions)*100:.1f}%)")
    print(f"   add_to_cart : {carts:,}     ({carts/len(interactions)*100:.1f}%)")
    print(f"   view        : {views:,}      ({views/len(interactions)*100:.1f}%)")
    print(f"   Có review_text: {has_text:,} ({has_text/len(interactions)*100:.1f}%)")
    print(f"   sentiment positive: {pos_senti:,} ({pos_senti/len(interactions)*100:.1f}%)")
    print(f"   Row sparsity: {sparsity_row*100:.2f}%")
    print(f"   CF sparsity : {sparsity_cf*100:.2f}%")
    print(f"   Avg weight: {avg_weight:.3f}")
    print(f"{C_BOLD}{C_CYAN}{'═'*60}{C_RESET}\n")

    return {
        "num_products": len(products),
        "num_users": len(users),
        "num_interactions": len(interactions),
        "action_counts": action_counts,
        "sparsity_row": sparsity_row,
        "sparsity_cf": sparsity_cf,
        "avg_weight": avg_weight,
    }


if __name__ == "__main__":
    import json
    from pathlib import Path
    for fname in ["products.json", "users.json", "interactions.json"]:
        if not Path(fname).exists():
            print(f" Thiếu {fname}. Hãy chạy đầy đủ các bước trước.")
            exit(1)

    products = json.load(open("products.json", encoding="utf-8"))
    users = json.load(open("users.json", encoding="utf-8"))
    interactions = json.load(open("interactions.json", encoding="utf-8"))
    print_quality_report(products, users, interactions)