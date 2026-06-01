"""
AFDGCN Model Sunucusu (Port 9002)
===================================
TypeScript backend'in HTTP ile çağırdığı minimal model servisi.

Tek görevi:
  POST /predict/next  →  AFDGCN çalıştır  →  araç sayısı tahmini döndür

Başlatmak için (proje kökünden):
  python model_server.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Proje kökünü path'e ekle (model/ ve saved_models/ klasörlerine erişim için)
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("model_server")

# ─────────────────────────────────────────────────────────────────────────────
# İstek / Yanıt Modelleri
# ─────────────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    data_by_junction: Dict[int, List[dict]]
    minute_index: int


class PredictResponse(BaseModel):
    predictions: Dict[int, Dict[str, float]]
    source: str  # "AFDGCN" | "moving_average"


class PredictSeriesRequest(BaseModel):
    data_by_junction: Dict[int, List[dict]]
    completed_idx: int


class PredictSeriesResponse(BaseModel):
    prediction_series: Dict[int, Dict[str, List[float]]]
    source: str  # "AFDGCN" | "moving_average"


class LoadModelRequest(BaseModel):
    path: str
    num_nodes: int = 34
    lag: int = 12
    horizon: int = 6
    scaler_mean: float = 28.53
    scaler_std: float = 38.72
    input_dim: int = 3
    tod_enabled: bool = True


class LoadModelFromBytesRequest(BaseModel):
    num_nodes: int = 34
    lag: int = 12
    horizon: int = 6
    scaler_mean: float = 28.53
    scaler_std: float = 38.72
    input_dim: int = 3
    tod_enabled: bool = True


class PredictDayAheadRequest(BaseModel):
    seed_data_by_junction: Dict[int, List[dict]]
    seed_completed_idx: int = 143


class PredictDayAheadResponse(BaseModel):
    prediction_series: Dict[int, Dict[str, List[float]]]
    source: str  # "AFDGCN" | "moving_average"


class LoadModelResponse(BaseModel):
    success: bool
    message: str


# ─────────────────────────────────────────────────────────────────────────────
# Uygulama
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AFDGCN Model Server",
    version="1.0.0",
    description="TypeScript backend için minimal AFDGCN model servisi",
)


@app.on_event("startup")
async def on_startup():
    logger.info("Model sunucusu başlatılıyor...")
    from ml.prediction_wrapper import ensure_model_loaded
    await ensure_model_loaded()
    logger.info("Model yükleme tamamlandı")


@app.post("/predict/next", response_model=PredictResponse)
async def predict_next(body: PredictRequest):
    """
    Bir sonraki 10 dakikalık zaman dilimine ait araç sayısı tahminini döndürür.
    Moving average fallback otomatik olarak devreye girer.
    """
    from ml.prediction_wrapper import (
        predict_next_timestep,
        get_model_status,
    )

    try:
        predictions = await predict_next_timestep(
            body.data_by_junction,
            body.minute_index,
        )
        status = get_model_status()
        source = "moving_average" if status.get("fallback_active") else "AFDGCN"
        return PredictResponse(predictions=predictions, source=source)
    except Exception as exc:
        logger.error("Tahmin hatası: %s", exc)
        # Fallback: sıfır tahmin döndür, TypeScript tarafı moving average'a geçer
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc)},
        )


@app.post("/predict/series", response_model=PredictSeriesResponse)
async def predict_series(body: PredictSeriesRequest):
    """
    Gün başından completed_idx'e kadar her slot için gerçek AFDGCN tahminini döndürür.
    Her slot için tahmin, o slottan önceki lag gerçek değeri kullanılarak üretilir
    (data leakage yok).
    """
    from ml.prediction_wrapper import (
        predict_rolling_series,
        get_model_status,
    )

    try:
        series = await predict_rolling_series(
            body.data_by_junction,
            body.completed_idx,
        )
        status = get_model_status()
        source = "moving_average" if status.get("fallback_active") else "AFDGCN"
        return PredictSeriesResponse(prediction_series=series, source=source)
    except Exception as exc:
        logger.error("Seri tahmin hatası: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc)},
        )


@app.post("/predict/day-ahead", response_model=PredictDayAheadResponse)
async def predict_day_ahead(body: PredictDayAheadRequest):
    """
    Bir onceki gunun gercek verisi tohum alinarak bir sonraki gun icin
    tum 144 slot otoregresif tahmin uretir.
    """
    from ml.prediction_wrapper import predict_full_day, get_model_status

    try:
        series = await predict_full_day(
            body.seed_data_by_junction,
            body.seed_completed_idx,
        )
        status = get_model_status()
        source = "moving_average" if status.get("fallback_active") else "AFDGCN"
        return PredictDayAheadResponse(prediction_series=series, source=source)
    except Exception as exc:
        logger.error("Gun-oncesi tahmin hatasi: %s", exc)
        return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/model/status")
async def model_status():
    """Model durumunu döndürür (TypeScript PythonModelService bu endpoint'i çağırır)."""
    from ml.prediction_wrapper import get_model_status
    return get_model_status()


@app.post("/model/load", response_model=LoadModelResponse)
async def load_model(body: LoadModelRequest):
    """
    Verilen .pth dosyasını yükler. TypeScript backend model aktivasyonunda çağırır.
    Hot-reload: sunucu yeniden başlatılmaz.
    """
    from ml.prediction_wrapper import reload_model
    try:
        success, message = await reload_model(
            model_path=body.path,
            num_nodes=body.num_nodes,
            lag=body.lag,
            horizon=body.horizon,
            scaler_mean=body.scaler_mean,
            scaler_std=body.scaler_std,
        )
        logger.info("Model yükleme: success=%s | %s", success, message)
        return LoadModelResponse(success=success, message=message)
    except Exception as exc:
        logger.error("Model yükleme hatası: %s", exc)
        return LoadModelResponse(success=False, message=str(exc))


@app.post("/model/load-from-bytes", response_model=LoadModelResponse)
async def load_model_from_bytes(
    request: Request,
    num_nodes: int = 34,
    lag: int = 1,
    horizon: int = 1,
    scaler_mean: float = 28.53,
    scaler_std: float = 38.72,
):
    """
    Ham .pth byte akışından model yükler (dosya sistemi gerekmez).
    Content-Type: application/octet-stream
    Query params: num_nodes, lag, horizon, scaler_mean, scaler_std
    """
    from ml.prediction_wrapper import reload_model_from_bytes
    try:
        data = await request.body()
        if not data:
            return LoadModelResponse(success=False, message="Boş istek gövdesi")
        success, message = await reload_model_from_bytes(
            weights=data,
            num_nodes=num_nodes,
            lag=lag,
            horizon=horizon,
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
        )
        logger.info("load-from-bytes: success=%s | %s", success, message)
        return LoadModelResponse(success=success, message=message)
    except Exception as exc:
        logger.error("load-from-bytes hatasi: %s", exc)
        return LoadModelResponse(success=False, message=str(exc))


@app.get("/health")
async def health():
    from ml.prediction_wrapper import get_model_status
    return {"status": "ok", "model": get_model_status()}


# ─────────────────────────────────────────────────────────────────────────────
# Başlatma
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "model_server:app",
        host="0.0.0.0",
        port=9002,
        reload=False,
        log_level="info",
    )
