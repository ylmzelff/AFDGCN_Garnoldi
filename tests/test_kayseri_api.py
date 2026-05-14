"""
Phase API Entegrasyon Testleri
================================

Çalıştırma (proje kökünden):
    pytest tests/test_phase_api.py -v
    pytest tests/test_phase_api.py -v -s   # stdout ile
"""

import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import AsyncMock, MagicMock, patch

# Proje kökünden import
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.main import app
from backend.app.schemas.phases import RegionPhaseResponse

MOCK_REGION_DATA = {
    89: [
        {"edge_direction": "A", "edge_name": "Sivas Blv", "0": 12.0, "1": 15.0},
        {"edge_direction": "B", "edge_name": "Test Cad",  "0":  8.0, "1":  9.0},
        {"edge_direction": "C", "edge_name": "Test Sok",  "0":  5.0, "1":  6.0},
        {"edge_direction": "D", "edge_name": "Test Yol",  "0":  3.0, "1":  3.0},
    ]
}

MOCK_PREDICTIONS = {
    89:  {"A": 13.0, "B": 9.0,  "C": 6.0,  "D": 3.5},
    95:  {"A": 20.0, "B": 18.0, "C": 14.0, "D": 10.0},
    117: {"A": 8.0,  "C": 5.0,  "D": 4.0},
    121: {"A": 15.0, "B": 12.0, "C": 10.0, "D": 8.0},
    184: {"A": 22.0, "B": 18.0, "D": 30.0},
    187: {"A": 10.0, "B": 8.0,  "C": 7.0,  "D": 5.0},
    188: {"A": 25.0, "B": 20.0, "C": 15.0, "D": 6.0},
    192: {"A": 12.0, "B": 6.0,  "C": 10.0, "D": 9.0},
    194: {"A": 18.0, "B": 14.0, "C": 16.0, "D": 12.0},
}


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcı: Token al
# ─────────────────────────────────────────────────────────────────────────────

async def _get_token(ac: AsyncClient) -> str:
    resp = await ac.post("/auth/login", json={"username": "demo", "password": "demo123"})
    assert resp.status_code == 200, f"Login hatası: {resp.text}"
    return resp.json()["access_token"]


# ─────────────────────────────────────────────────────────────────────────────
# Testler
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_health_endpoint():
    """GET /health — auth gerekmez."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        resp = await ac.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "model" in data
    assert "version" in data


@pytest.mark.anyio
async def test_login_success():
    """POST /auth/login — geçerli kimlik."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        resp = await ac.post("/auth/login", json={"username": "demo", "password": "demo123"})
    assert resp.status_code == 200
    assert "access_token" in resp.json()
    assert resp.json()["token_type"] == "bearer"


@pytest.mark.anyio
async def test_login_wrong_password():
    """POST /auth/login — yanlış şifre → 401."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        resp = await ac.post("/auth/login", json={"username": "demo", "password": "wrong"})
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_phases_requires_auth():
    """GET /phases/ildem — token olmadan → 401/403."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        resp = await ac.get("/phases/ildem")
    assert resp.status_code in (401, 403)


@pytest.mark.anyio
@patch("backend.app.api.v1.phases.build_ildem_phases")
async def test_get_ildem_phases(mock_build):
    """GET /phases/ildem — başarılı yanıt."""
    from backend.phase_api import RegionPhaseResponse
    mock_response = MagicMock(spec=RegionPhaseResponse)
    mock_response.model_dump.return_value = {
        "region": "ildem",
        "city": "kayseri",
        "timestamp": "2026-04-24T11:00:00",
        "time_label": "11:00",
        "prediction_source": "moving_average",
        "kayseri_api_status": "unavailable",
        "junctions": [],
    }
    mock_build.return_value = (mock_response, False)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        token = await _get_token(ac)
        resp = await ac.get(
            "/phases/ildem",
            headers={"Authorization": f"Bearer {token}"},
        )
    assert resp.status_code == 200


@pytest.mark.anyio
@patch("backend.app.api.v1.phases.build_tuna_phases")
async def test_get_tuna_phases(mock_build):
    """GET /phases/tuna — her zaman moving_average."""
    from backend.phase_api import RegionPhaseResponse
    mock_response = MagicMock(spec=RegionPhaseResponse)
    mock_response.model_dump.return_value = {
        "region": "tuna",
        "city": "kayseri",
        "timestamp": "2026-04-24T11:00:00",
        "time_label": "11:00",
        "prediction_source": "moving_average",
        "kayseri_api_status": "unavailable",
        "junctions": [],
    }
    mock_build.return_value = (mock_response, False)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        token = await _get_token(ac)
        resp = await ac.get(
            "/phases/tuna",
            headers={"Authorization": f"Bearer {token}"},
        )
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_auth_me():
    """GET /auth/me — token sahibi döner."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        token = await _get_token(ac)
        resp = await ac.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
    assert resp.status_code == 200
    assert resp.json()["username"] == "demo"

        assert "devre dışı" in data["detail"].lower()


class TestAutoPredictService:
    """Otomatik tahmin servisi testleri."""
    
    @pytest.mark.asyncio
    @patch("api.kayseri_db.get_available_junctions")
    @patch("api.kayseri_db.fetch_latest_traffic_data")
    @patch("api.inference.predict")
    async def test_auto_predict_run(
        self,
        mock_predict,
        mock_fetch_data,
        mock_get_junctions,
        mock_kayseri_data,
        mock_prediction_result,
    ):
        """Otomatik tahmin servisinin tek çalışması."""
        from api.auto_predict import AutoPredictionService
        
        mock_get_junctions.return_value = ["test_junction"]
        mock_fetch_data.return_value = mock_kayseri_data
        mock_predict.return_value = mock_prediction_result
        
        service = AutoPredictionService(
            interval_seconds=1,
            enabled=True,
        )
        
        # Tek çalıştırma
        await service._run_predictions()
        
        # Tahmin fonksiyonu çağrıldı mı?
        mock_predict.assert_called_once()


# Pytest configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
