from __future__ import annotations

import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline import (
    _filter,
    _rescore,
    run_pipeline,
    DEFAULT_SCORE_WEIGHT,
    DEFAULT_RATING_WEIGHT,
)


# ══════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════

def _candidates_with_tight_scores() -> list[dict]:
    """FAISS scores cho SP tương tự — cluster sát nhau như thực tế."""
    return [
        {"product_id": "A", "score": 0.96, "source": "similar",
         "meta": {"rating_avg": 4.2, "rating": 0.0, "stock": 10, "price": 100_000}},
        {"product_id": "B", "score": 0.95, "source": "similar",
         "meta": {"rating_avg": 3.8, "rating": 0.0, "stock": 5,  "price": 90_000}},
        {"product_id": "C", "score": 0.94, "source": "similar",
         "meta": {"rating_avg": 4.5, "rating": 0.0, "stock": 8,  "price": 120_000}},
    ]


def _candidates_no_meta() -> list[dict]:
    """Candidates chưa qua enrichment — meta rỗng."""
    return [
        {"product_id": "X", "score": 0.92, "source": "similar"},
        {"product_id": "Y", "score": 0.90, "source": "similar"},
    ]

class TestBug1WrongDefaultWeights:

    def test_default_weights_hurt_similar_endpoint(self):
        """
        DEFAULT score_weight=0.2 / rating_weight=0.8 khiến final_score thấp
        khi rating thiếu — minh hoạ tại sao /similar phải override weights.
        """
        cands = _candidates_with_tight_scores()
        with_defaults = _rescore(
            [c.copy() for c in cands],
            score_weight=DEFAULT_SCORE_WEIGHT,   # 0.2
            rating_weight=DEFAULT_RATING_WEIGHT, # 0.8
        )
        with_similar_weights = _rescore(
            [c.copy() for c in cands],
            score_weight=0.8,
            rating_weight=0.2,
        )
        # Với trọng số đúng, score phải cao hơn đáng kể
        avg_default = sum(c["final_score"] for c in with_defaults) / len(with_defaults)
        avg_similar = sum(c["final_score"] for c in with_similar_weights) / len(with_similar_weights)
        assert avg_similar > avg_default, (
            f"score_weight=0.8 phải cho score cao hơn score_weight=0.2\n"
            f"  avg_similar={avg_similar:.4f}  avg_default={avg_default:.4f}"
        )

    def test_similar_weights_produce_nonzero_scores(self):
        """Với score_weight=0.8, kết quả phải > 0 ngay cả khi meta thiếu."""
        cands = _candidates_no_meta()
        result = _rescore(cands, score_weight=0.8, rating_weight=0.2)
        for item in result:
            assert item["final_score"] > 0, (
                f"score không được = 0 khi score_weight=0.8: "
                f"product_id={item['product_id']} score={item['final_score']}"
            )

class TestBug2EnrichDbFieldMismatch:

    def test_enrich_skipped_when_db_none(self):
        """Khi db=None, pipeline vẫn chạy bình thường — không crash."""
        result = run_pipeline(
            raw_candidates=_candidates_no_meta(),
            user_id="test_user",
            db=None,
            n=10,
            score_weight=0.8,
            rating_weight=0.2,
        )
        assert len(result.items) > 0, "Pipeline phải trả về items dù không có DB"
        for item in result.items:
            assert item["score"] > 0, (
                f"score phải > 0 dù db=None: product_id={item['product_id']}"
            )

    def test_enrich_db_wrong_field_causes_zero_meta(self):
        """
        Minh hoạ: nếu enrich dùng sai field, meta = {} → rating = 0.
        Sau fix, pipeline vẫn cho score > 0 nhờ score_weight cao.
        """
        # Simulate kết quả của enrich bị lỗi: meta rỗng
        cands = [
            {"product_id": "A", "score": 0.95, "meta": {}},
            {"product_id": "B", "score": 0.93, "meta": {}},
        ]
        result = _rescore(cands, score_weight=0.8, rating_weight=0.2)
        for item in result:
            # Với score_weight=0.8, dù rating=0, vẫn phải có score
            assert item["final_score"] > 0.7, (
                f"score_weight=0.8 phải đảm bảo final_score > 0.7 dù rating=0: "
                f"{item['product_id']} = {item['final_score']}"
            )

class TestBug3MinMaxCollapse:

    def test_tight_faiss_scores_not_collapsed(self):
        """
        Bug 3 core: scores 0.94/0.95/0.96 không được normalize về 0.
        Absolute clamp phải giữ nguyên giá trị gần với 1.0.
        """
        cands = _candidates_with_tight_scores()
        result = _rescore(cands, score_weight=0.8, rating_weight=0.2)
        for item in result:
            assert item["final_score"] > 0.5, (
                f"FAISS score 0.94-0.96 bị collapse về 0 — Bug 3 chưa fix: "
                f"{item['product_id']} = {item['final_score']}"
            )

    def test_score_ordering_preserved(self):
        """Thứ tự rank phải tương quan với input score khi rating bằng nhau."""
        cands = [
            {"product_id": "HIGH", "score": 0.96,
             "meta": {"rating_avg": 4.0, "stock": 5}},
            {"product_id": "LOW",  "score": 0.90,
             "meta": {"rating_avg": 4.0, "stock": 5}},
        ]
        result = _rescore(cands, score_weight=0.8, rating_weight=0.2)
        scores = {item["product_id"]: item["final_score"] for item in result}
        assert scores["HIGH"] > scores["LOW"], (
            "SP có FAISS score cao hơn phải có final_score cao hơn khi rating bằng nhau"
        )

    def test_extreme_tight_scores_not_zero(self):
        """Edge case: scores cực kỳ gần nhau (ví dụ batch đồng nhất)."""
        cands = [
            {"product_id": str(i), "score": 0.999 - i * 0.001,
             "meta": {"rating_avg": 4.0, "stock": 1}}
            for i in range(10)
        ]
        result = _rescore(cands, score_weight=0.8, rating_weight=0.2)
        zeros = [c for c in result if c["final_score"] == 0.0
                 and c.get("meta", {}).get("stock", 1) > 0]
        assert len(zeros) == 0, (
            f"Bug 3: {len(zeros)} items bị collapse về 0 dù có stock: "
            f"{[c['product_id'] for c in zeros]}"
        )

class TestBug4FaissScoreRemap:

    def test_remap_formula_correct(self):
        """(score + 1) / 2 phải map [-1,1] → [0,1] chính xác."""
        test_cases = [
            (-1.0, 0.0),
            ( 0.0, 0.5),
            ( 1.0, 1.0),
            ( 0.92, 0.96),
            (-0.1, 0.45),
        ]
        for raw, expected in test_cases:
            remapped = (raw + 1.0) / 2.0
            assert abs(remapped - expected) < 1e-6, (
                f"Remap sai: ({raw} + 1) / 2 = {remapped}, expected {expected}"
            )

    def test_negative_raw_score_does_not_collapse_to_zero(self):
        """
        Score âm từ FAISS (trước remap) không được gây ra 0% sau pipeline.
        """
        # Giả sử FAISS trả về -0.1 → sau remap = 0.45 → vẫn phải có score
        remapped_score = (-0.1 + 1.0) / 2.0   # = 0.45
        cands = [
            {"product_id": "A", "score": remapped_score,
             "meta": {"rating_avg": 3.5, "stock": 5}},
        ]
        result = _rescore(cands, score_weight=0.8, rating_weight=0.2)
        assert result[0]["final_score"] > 0, (
            "Score từ FAISS âm (sau remap = 0.45) vẫn phải cho final_score > 0"
        )

    def test_without_remap_negative_score_causes_zero(self):
        """
        Minh hoạ bug cũ: score âm không remap → clamp về 0 → final_score = 0.
        Test này pass khi BUG tồn tại — dùng để document hành vi cũ.
        """
        raw_negative = -0.1   # inner product từ IndexFlatIP trước khi remap
        clamped = min(max(raw_negative, 0.0), 1.0)   # = 0.0 — lỗi cũ
        assert clamped == 0.0, "Minh hoạ: clamp score âm → 0.0 (lỗi cũ)"

        remapped = (raw_negative + 1.0) / 2.0        # = 0.45 — đúng
        assert remapped > 0.0, "Sau remap: score âm → 0.45 (đúng)"


class TestBug5RatingFieldMismatch:

    def test_rating_avg_used_when_rating_absent(self):
        """Pipeline phải đọc được rating_avg khi rating không có."""
        cands = [
            {"product_id": "A", "score": 0.92,
             "meta": {"rating_avg": 5.0, "stock": 10}},   # chỉ có rating_avg
            {"product_id": "B", "score": 0.91,
             "meta": {"rating_avg": 1.0, "stock": 10}},
        ]
        result = _rescore(cands, score_weight=0.2, rating_weight=0.8)
        scores = {c["product_id"]: c["final_score"] for c in result}
        assert scores["A"] > scores["B"], (
            "SP có rating_avg=5.0 phải rank cao hơn rating_avg=1.0\n"
            "Nếu bằng nhau → Bug 5 chưa fix (pipeline không đọc được rating_avg)"
        )

    def test_rating_field_fallback_chain(self):
        """Thứ tự ưu tiên: rating_avg → rating → 0."""
        cases = [
            # (meta dict, expected_rating_norm > 0)
            ({"rating_avg": 4.0, "stock": 5}, True),
            ({"rating": 4.0,     "stock": 5}, True),   # chỉ có rating
            ({                   "stock": 5}, False),  # không có gì → 0
        ]
        for meta, expect_nonzero in cases:
            cands = [{"product_id": "T", "score": 0.5, "meta": meta}]
            result = _rescore(cands, score_weight=0.2, rating_weight=0.8)
            score = result[0]["final_score"]
            if expect_nonzero:
                assert score > 0.0, (
                    f"Với meta={meta}, phải đọc được rating → score > 0, got {score}"
                )
            else:
                # score chỉ từ score_weight=0.2 × 0.5 = 0.1
                assert abs(score - 0.1) < 0.01, (
                    f"Không có rating, score phải = 0.1 (chỉ từ score_weight), got {score}"
                )

    def test_both_fields_present_prefers_rating_avg(self):
        """Khi có cả hai, rating_avg (aggregate SP) được ưu tiên hơn rating (interaction)."""
        meta_both = {"rating_avg": 5.0, "rating": 1.0, "stock": 5}
        meta_avg_only = {"rating_avg": 5.0, "stock": 5}
        for meta in [meta_both, meta_avg_only]:
            cands = [{"product_id": "T", "score": 0.5, "meta": meta}]
            result = _rescore(cands, score_weight=0.2, rating_weight=0.8)
            # rating_avg=5.0 → norm_rating=1.0 → final = 0.2×0.5 + 0.8×1.0 = 0.9
            assert result[0]["final_score"] >= 0.88, (
                f"rating_avg=5.0 phải cho final_score ≈ 0.9, meta={meta}, "
                f"got {result[0]['final_score']}"
            )


class TestIntegrationNoZeroScore:

    def test_full_pipeline_no_zero_score_similar_usecase(self):
        """
        End-to-end: simulate toàn bộ /similar flow với FAISS scores đã remap.
        Tất cả SP có stock phải có final_score > 0.
        """
        # Giả lập output của _faiss_similar (đã remap về [0,1])
        faiss_output = [
            {"product_id": f"SP{i:03d}", "score": round(0.96 - i * 0.005, 3),
             "source": "similar"}
            for i in range(10)
        ]
        result = run_pipeline(
            raw_candidates = faiss_output,
            user_id        = "test_user",
            db             = None,      # không có DB — test worst case
            n              = 10,
            score_weight   = 0.8,       # FIX Bug 1
            rating_weight  = 0.2,
        )
        assert len(result.items) == 10
        for item in result.items:
            assert item["score"] > 0, (
                f" score=0 với /similar weights — Bug chưa fix hoàn toàn: "
                f"product_id={item['product_id']}"
            )

    def test_out_of_stock_items_rank_last(self):
        """SP hết hàng phải có final_score=0 và xếp cuối."""
        cands = [
            {"product_id": "IN_STOCK",    "score": 0.90,
             "meta": {"rating_avg": 4.0, "stock": 10}},
            {"product_id": "OUT_STOCK",   "score": 0.95,
             "meta": {"rating_avg": 5.0, "stock": 0}},   # score cao hơn nhưng hết hàng
        ]
        result = _rescore(cands, score_weight=0.8, rating_weight=0.2)
        scores = {c["product_id"]: c["final_score"] for c in result}
        assert scores["OUT_STOCK"] == 0.0, "SP hết hàng phải có final_score=0"
        assert scores["IN_STOCK"] > 0.0,   "SP còn hàng phải có final_score > 0"

    def test_filter_removes_owned_and_excluded(self):
        """Candidates đã owned/excluded phải bị loại trước khi rescore."""
        result = run_pipeline(
            raw_candidates=[
                {"product_id": "OWNED",    "score": 0.99},
                {"product_id": "EXCLUDED", "score": 0.98},
                {"product_id": "KEEP",     "score": 0.95},
            ],
            user_id     = "u1",
            db          = None,
            n           = 10,
            owned_ids   = {"OWNED"},
            exclude_ids = {"EXCLUDED"},
            score_weight = 0.8,
            rating_weight = 0.2,
        )
        ids = [item["product_id"] for item in result.items]
        assert "OWNED"    not in ids, "SP owned phải bị loại"
        assert "EXCLUDED" not in ids, "SP excluded phải bị loại"
        assert "KEEP"     in ids,     "SP hợp lệ phải được giữ"
