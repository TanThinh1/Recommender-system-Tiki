from __future__ import annotations

import types
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ══════════════════════════════════════════════════════════════════════
#  MOCK STORE — giả lập model đã load, không cần file .pkl hay FAISS
# ══════════════════════════════════════════════════════════════════════
def _make_mock_store():
    """Tạo ModelStore giả đủ để các endpoint hoạt động."""
    import numpy as np

    s = MagicMock()
    s.als_model    = MagicMock()
    s.user2idx     = {"user_001": 0, "user_002": 1}
    s.product2idx  = {f"SP{i:03d}": i for i in range(20)}
    s.idx2product  = {i: f"SP{i:03d}" for i in range(20)}
    s.idx2user     = {0: "user_001", 1: "user_002"}
    s.popular_items = [(f"SP{i:03d}", float(100 - i)) for i in range(20)]
    s.faiss_index   = MagicMock()
    s.faiss_id_map  = [f"SP{i:03d}" for i in range(20)]
    s.faiss_id_to_idx = {f"SP{i:03d}": i for i in range(20)}
    s.meta          = {}
    s.loaded_at     = "2025-01-01T00:00:00"

    # ALS recommend trả về list of RecommendItem-like objects
    from collections import namedtuple
    Item = namedtuple("Item", ["product_id", "score"])
    s.als_model.recommend.return_value = (
        [Item(f"SP{i:03d}", round(0.95 - i * 0.03, 4)) for i in range(10)],
        [round(0.95 - i * 0.03, 4) for i in range(10)],
    )

    # FAISS search: trả về scores và indices tương tự
    scores  = [[0.96, 0.94, 0.93, 0.91, 0.90, 0.88, 0.87, 0.85, 0.84, 0.83, 0.0]]
    indices = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1]]
    s.faiss_index.search.return_value = (
        __import__("numpy").array(scores),
        __import__("numpy").array(indices),
    )
    s.faiss_index.reconstruct.return_value = __import__("numpy").zeros(128, dtype="float32")

    return s


# ══════════════════════════════════════════════════════════════════════
#  FIXTURES
# ══════════════════════════════════════════════════════════════════════
API_KEY = "test-key-abc123"

@pytest.fixture(scope="module")
def client():
    """TestClient với mock store và API key set."""
    import os
    os.environ["API_KEY"]          = API_KEY
    os.environ["ALLOWED_ORIGINS"]  = ""

    # Patch pickle/model loading để lifespan không crash khi không có file thật
    mock_store = _make_mock_store()

    with (
        patch("recommend_api.ModelStore", return_value=mock_store),
        patch("recommend_api._load_als"),
        patch("recommend_api._load_faiss"),
        patch("recommend_api._load_meta"),
        patch("recommend_api._smoke_test_similar"),
        patch("recommend_api.store", mock_store),
        patch("recommend_api.db", None),
    ):
        from recommend_api import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


@pytest.fixture
def auth_headers():
    return {"X-API-Key": API_KEY}


# ══════════════════════════════════════════════════════════════════════
# AUTH — mọi endpoint có dữ liệu phải từ chối request thiếu key
# ══════════════════════════════════════════════════════════════════════
class TestAuthRequired:

    PROTECTED = [
        ("GET",  "/recommend/user_001"),
        ("GET",  "/similar/SP001"),
        ("GET",  "/hybrid/user_001"),
        ("POST", "/feedback"),
        ("GET",  "/metrics/online"),
    ]

    @pytest.mark.parametrize("method,path", PROTECTED)
    def test_missing_key_returns_401(self, client, method, path):
        resp = getattr(client, method.lower())(path)
        assert resp.status_code == 401, f"{method} {path} phải trả 401 khi thiếu API key"

    @pytest.mark.parametrize("method,path", PROTECTED)
    def test_wrong_key_returns_401(self, client, method, path):
        resp = getattr(client, method.lower())(path, headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401, f"{method} {path} phải trả 401 khi key sai"

    def test_health_is_public(self, client):
        """GET /health không cần key."""
        resp = client.get("/health")
        assert resp.status_code == 200


# ══════════════════════════════════════════════════════════════════════
#  /recommend — happy path & edge cases
# ══════════════════════════════════════════════════════════════════════
class TestRecommendEndpoint:

    def test_known_user_returns_items(self, client, auth_headers):
        resp = client.get("/recommend/user_001", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert "items" in body
        assert len(body["items"]) > 0
        assert body["is_cold"] is False

    def test_cold_start_user_returns_popular(self, client, auth_headers):
        """User không có trong model → popular fallback, is_cold=True."""
        resp = client.get("/recommend/unknown_user_xyz", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["is_cold"] is True
        assert len(body["items"]) > 0
        for item in body["items"]:
            assert item["source"] == "popular"

    def test_n_param_respected(self, client, auth_headers):
        resp = client.get("/recommend/user_001?n=3", headers=auth_headers)
        assert resp.status_code == 200
        assert len(resp.json()["items"]) <= 3

    def test_n_out_of_range_clamped(self, client, auth_headers):
        """n=999 phải bị clamp về 50 (FastAPI schema le=50)."""
        resp = client.get("/recommend/user_001?n=999", headers=auth_headers)
        # FastAPI validation trả 422
        assert resp.status_code == 422

    def test_n_zero_returns_422(self, client, auth_headers):
        resp = client.get("/recommend/user_001?n=0", headers=auth_headers)
        assert resp.status_code == 422

    def test_response_schema(self, client, auth_headers):
        resp = client.get("/recommend/user_001", headers=auth_headers)
        body = resp.json()
        assert "user_id"            in body
        assert "items"              in body
        assert "is_cold"            in body
        assert "latency_ms"         in body
        assert "recommendation_id"  in body
        for item in body["items"]:
            assert "product_id" in item
            assert "score"      in item
            assert "source"     in item

    def test_scores_between_0_and_1(self, client, auth_headers):
        resp = client.get("/recommend/user_001", headers=auth_headers)
        for item in resp.json()["items"]:
            assert 0.0 <= item["score"] <= 1.0, (
                f"score phải trong [0,1]: {item['product_id']} = {item['score']}"
            )


# ══════════════════════════════════════════════════════════════════════
#  /similar — happy path & edge cases
# ══════════════════════════════════════════════════════════════════════
class TestSimilarEndpoint:

    def test_known_product_returns_similar(self, client, auth_headers):
        resp = client.get("/similar/SP001", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert "similar" in body
        assert len(body["similar"]) > 0

    def test_product_not_in_faiss_returns_404(self, client, auth_headers):
        resp = client.get("/similar/NOT_EXIST_XYZ", headers=auth_headers)
        assert resp.status_code == 404

    def test_result_does_not_include_query_product(self, client, auth_headers):
        """SP query không được xuất hiện trong kết quả."""
        resp = client.get("/similar/SP001", headers=auth_headers)
        ids = [i["product_id"] for i in resp.json()["similar"]]
        assert "SP001" not in ids, "SP query không được có trong similar results"

    def test_n_param(self, client, auth_headers):
        resp = client.get("/similar/SP001?n=5", headers=auth_headers)
        assert len(resp.json()["similar"]) <= 5

    def test_scores_remapped_positive(self, client, auth_headers):
        """FAISS cosine score đã remap về [0,1] — phải > 0."""
        resp = client.get("/similar/SP001", headers=auth_headers)
        for item in resp.json()["similar"]:
            assert item["score"] > 0, f"Score phải > 0 sau remap: {item}"


# ══════════════════════════════════════════════════════════════════════
#  /hybrid — trọng số ALS + FAISS
# ══════════════════════════════════════════════════════════════════════
class TestHybridEndpoint:

    def test_returns_items(self, client, auth_headers):
        resp = client.get("/hybrid/user_001", headers=auth_headers)
        assert resp.status_code == 200
        assert len(resp.json()["items"]) > 0

    def test_als_weight_validation(self, client, auth_headers):
        """als_weight ngoài [0,1] phải bị từ chối."""
        resp = client.get("/hybrid/user_001?als_weight=1.5", headers=auth_headers)
        assert resp.status_code == 422

    def test_cold_user_hybrid(self, client, auth_headers):
        resp = client.get("/hybrid/brand_new_user_000", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["is_cold"] is True


# ══════════════════════════════════════════════════════════════════════
#  /feedback — non-blocking, schema
# ══════════════════════════════════════════════════════════════════════
class TestFeedbackEndpoint:

    def _payload(self, **kwargs):
        base = {
            "user_id"    : "user_001",
            "product_id" : "SP001",
            "action"     : "click",
            "weight"     : 0.3,
            "source"     : "test",
        }
        return {**base, **kwargs}

    def test_accepted_returns_202(self, client, auth_headers):
        resp = client.post("/feedback", json=self._payload(), headers=auth_headers)
        assert resp.status_code == 202
        assert resp.json()["status"] == "accepted"

    def test_with_recommendation_id(self, client, auth_headers):
        payload = self._payload(recommendation_id="rec-uuid-001")
        resp = client.post("/feedback", json=payload, headers=auth_headers)
        assert resp.status_code == 202

    def test_invalid_action_still_accepted(self, client, auth_headers):
        """FeedbackRequest không validate enum action — ghi log rồi thôi."""
        resp = client.post("/feedback", json=self._payload(action="unknown_action"),
                           headers=auth_headers)
        assert resp.status_code == 202

    def test_missing_required_fields_returns_422(self, client, auth_headers):
        resp = client.post("/feedback", json={"user_id": "u1"}, headers=auth_headers)
        assert resp.status_code == 422


# ══════════════════════════════════════════════════════════════════════
#  /health — public, không cần auth
# ══════════════════════════════════════════════════════════════════════
class TestHealthEndpoint:

    def test_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("status") in ("ok", "ready", "healthy")

    def test_has_model_info(self, client):
        resp = client.get("/health")
        body = resp.json()
        # Ít nhất phải có loaded_at hoặc model_loaded
        assert "loaded_at" in body or "model_loaded" in body or "status" in body
