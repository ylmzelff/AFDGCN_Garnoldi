"""
AFDGCN Model Sunucusu (Port 9002)
===================================
TypeScript backend'in HTTP ile çağırdığı minimal model servisi.

Tek görevi:
  POST /predict/next  →  AFDGCN çalıştır  →  araç sayısı tahmini döndür

Birden fazla bölge (region) aynı anda, birbirinden bağımsız modellerle
serve edilebilir — her istek hangi bölge için olduğunu `region` alanıyla
belirtir (bkz. ml/prediction_wrapper.py).

Başlatmak için (proje kökünden):
  python model_server.py
"""

from __future__ import annotations

import json
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
    region: str
    data_by_junction: Dict[int, List[dict]]
    minute_index: int


class PredictResponse(BaseModel):
    predictions: Dict[int, Dict[str, float]]
    source: str  # "AFDGCN" | "moving_average"


class PredictSeriesRequest(BaseModel):
    region: str
    data_by_junction: Dict[int, List[dict]]
    completed_idx: int


class PredictSeriesResponse(BaseModel):
    prediction_series: Dict[int, Dict[str, List[float]]]
    source: str  # "AFDGCN" | "moving_average"


class LoadModelRequest(BaseModel):
    region: str
    path: str
    num_nodes: int
    lag: int = 1
    horizon: int = 1
    scaler_mean: float = 0.0
    scaler_std: float = 1.0
    node_map: Optional[Dict[int, Dict[str, int]]] = None
    graph_edges: Optional[List[List[int]]] = None


class LoadModelResponse(BaseModel):
    success: bool
    message: str


# ─────────────────────────────────────────────────────────────────────────────
# Uygulama
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AFDGCN Model Server",
    version="2.0.0",
    description="TypeScript backend için çok-bölgeli AFDGCN model servisi",
)


@app.post("/predict/next", response_model=PredictResponse)
async def predict_next(body: PredictRequest):
    """
    Bir sonraki zaman dilimine ait araç sayısı tahminini döndürür.
    Moving average fallback otomatik olarak devreye girer.
    """
    from ml.prediction_wrapper import predict_next_timestep, get_model_status

    try:
        predictions = await predict_next_timestep(
            body.region,
            body.data_by_junction,
            body.minute_index,
        )
        status = get_model_status(body.region)
        source = "moving_average" if status.get("fallback_active") else "AFDGCN"
        return PredictResponse(predictions=predictions, source=source)
    except Exception as exc:
        logger.error("Tahmin hatası (%s): %s", body.region, exc)
        # Fallback: hata döndür, TypeScript tarafı moving average'a geçer
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
    from ml.prediction_wrapper import predict_rolling_series, get_model_status

    try:
        series = await predict_rolling_series(
            body.region,
            body.data_by_junction,
            body.completed_idx,
        )
        status = get_model_status(body.region)
        source = "moving_average" if status.get("fallback_active") else "AFDGCN"
        return PredictSeriesResponse(prediction_series=series, source=source)
    except Exception as exc:
        logger.error("Seri tahmin hatası (%s): %s", body.region, exc)
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc)},
        )


@app.get("/model/status")
async def model_status(region: Optional[str] = None):
    """
    Model durumunu döndürür (TypeScript PythonModelService bu endpoint'i çağırır).
    region verilmezse yüklü tüm bölgelerin durumu döner.
    """
    from ml.prediction_wrapper import get_model_status as _status
    return _status(region)


@app.post("/model/load", response_model=LoadModelResponse)
async def load_model(body: LoadModelRequest):
    """
    Verilen .pth dosyasını, belirtilen bölge için yükler.
    TypeScript backend model aktivasyonunda çağırır. Hot-reload: sunucu yeniden başlatılmaz.
    """
    from ml.prediction_wrapper import reload_model
    try:
        success, message = await reload_model(
            region=body.region,
            model_path=body.path,
            num_nodes=body.num_nodes,
            lag=body.lag,
            horizon=body.horizon,
            scaler_mean=body.scaler_mean,
            scaler_std=body.scaler_std,
            node_map=body.node_map,
            graph_edges=[tuple(e) for e in body.graph_edges] if body.graph_edges else None,
        )
        logger.info("Model yükleme (%s): success=%s | %s", body.region, success, message)
        return LoadModelResponse(success=success, message=message)
    except Exception as exc:
        logger.error("Model yükleme hatası (%s): %s", body.region, exc)
        return LoadModelResponse(success=False, message=str(exc))


@app.post("/model/load-from-bytes", response_model=LoadModelResponse)
async def load_model_from_bytes(
    request: Request,
    region: str,
    num_nodes: int,
    lag: int = 1,
    horizon: int = 1,
    scaler_mean: float = 0.0,
    scaler_std: float = 1.0,
    node_map: Optional[str] = None,
    graph_edges: Optional[str] = None,
):
    """
    Ham .pth byte akışından, belirtilen bölge için model yükler (dosya sistemi gerekmez).
    Content-Type: application/octet-stream
    Query params: region, num_nodes, lag, horizon, scaler_mean, scaler_std,
                  node_map (JSON string, {"junction_id": {"arm": node_index}}),
                  graph_edges (JSON string, [[from, to], ...])
    """
    from ml.prediction_wrapper import reload_model_from_bytes
    try:
        data = await request.body()
        if not data:
            return LoadModelResponse(success=False, message="Boş istek gövdesi")

        parsed_node_map = None
        if node_map:
            parsed_node_map = {int(k): v for k, v in json.loads(node_map).items()}
        parsed_graph_edges = None
        if graph_edges:
            parsed_graph_edges = [tuple(e) for e in json.loads(graph_edges)]

        success, message = await reload_model_from_bytes(
            region=region,
            weights=data,
            num_nodes=num_nodes,
            lag=lag,
            horizon=horizon,
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
            node_map=parsed_node_map,
            graph_edges=parsed_graph_edges,
        )
        logger.info("load-from-bytes (%s): success=%s | %s", region, success, message)
        return LoadModelResponse(success=success, message=message)
    except Exception as exc:
        logger.error("load-from-bytes hatasi (%s): %s", region, exc)
        return LoadModelResponse(success=False, message=str(exc))


@app.get("/health")
async def health():
    from ml.prediction_wrapper import get_model_status
    return {"status": "ok", "models": get_model_status()}


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
