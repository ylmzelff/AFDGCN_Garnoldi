"""
Çok-Bölgeli AFDGCN Tahmin Wrapper
====================================

Belediye API verisi → AFDGCN model → sonraki zaman dilimi araç sayısı tahmini.

Her bölge (region) kendi model, node haritası ve adjacency grafiyle ayrı ayrı
yönetilir (_MODELS[region]) — bu sayede birden fazla şehir/bölge aynı Python
sürecinde, birbirini etkilemeden, aynı anda AFDGCN ile tahmin üretebilir.
node_map / graph_edges / scaler / num_nodes hepsi model aktivasyonu sırasında
(TypeScript backend, RegionConfig/ModelVersion tablolarından) parametre olarak
gelir — bu dosyada hiçbir şehre özel sabit yoktur.

Model yoksa veya boyut uyuşmazlığı varsa moving average fallback devreye girer.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# ml/ → proje kökü
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

GraphEdge = Tuple[int, int]


# ─────────────────────────────────────────────────────────────────────────────
# Bölge başına model durumu
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelState:
    model: Optional[object] = None
    model_ready: bool = False
    model_path: Optional[str] = None

    num_nodes: int = 0
    lag: int = 1
    horizon: int = 1
    scaler_mean: float = 0.0
    scaler_std: float = 1.0
    # True ise model zaten raw (denormalize) değer çıkarıyor, _denormalize() uygulanmaz.
    model_outputs_raw: bool = True

    node_map: Dict[int, Dict[str, int]] = field(default_factory=dict)
    graph_edges: List[GraphEdge] = field(default_factory=list)

    max_history: int = 12
    history: List[np.ndarray] = field(default_factory=list)


_MODELS: Dict[str, ModelState] = {}
_MODELS_LOCK = asyncio.Lock()
_INFERENCE_LOCK = asyncio.Lock()


def _state_for(region: str) -> ModelState:
    """Bölgenin ModelState'ini döner, yoksa boş biri oluşturur."""
    state = _MODELS.get(region)
    if state is None:
        state = ModelState()
        _MODELS[region] = state
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Graf yükleme
# ─────────────────────────────────────────────────────────────────────────────

def _build_adj(num_nodes: int, edges: Sequence[GraphEdge]) -> torch.Tensor:
    """Verilen kenar listesinden num_nodes×num_nodes adjacency tensörü oluşturur."""
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for edge in edges:
        i, j = int(edge[0]), int(edge[1])
        if 0 <= i < num_nodes and 0 <= j < num_nodes:
            A[i, j] = 1.0
    logger.info("Graf oluşturuldu: %d kenar (%d node)", int(A.sum()), num_nodes)
    return torch.tensor(A)


# ─────────────────────────────────────────────────────────────────────────────
# Model yükleme (ortak çekirdek: dosyadan veya bytes'tan aynı mantık)
# ─────────────────────────────────────────────────────────────────────────────

def _load_state_dict_payload(raw) -> Tuple[dict, Optional[float], Optional[float]]:
    """Checkpoint dict formatını (scaler bilgisi içerebilir) çözer."""
    ckpt_scaler_mean = None
    ckpt_scaler_std = None
    if isinstance(raw, dict) and 'state_dict' in raw:
        ckpt_scaler_mean = raw.get('scaler_mean')
        ckpt_scaler_std = raw.get('scaler_std')
        raw = raw['state_dict']
    return raw, ckpt_scaler_mean, ckpt_scaler_std


def _build_net_from_state_dict(
    state_dict: dict,
    num_nodes: int,
    lag_default: int,
    horizon_default: int,
    edges: Sequence[GraphEdge],
):
    """
    state_dict'i verilen num_nodes ile doğrular, uyumsuzsa None döner.
    Uyumluysa (net, resolved_lag, resolved_horizon) döner.
    """
    import model.AFDGCN as afdgcn_mod

    if "node_embedding" not in state_dict:
        logger.warning("node_embedding bulunamadı")
        return None

    n_nodes, e_dim = state_dict["node_embedding"].shape
    if n_nodes != num_nodes:
        logger.warning(
            "checkpoint %d node iceriyor, %d bekleniyor - atlaniyor",
            n_nodes, num_nodes,
        )
        return None

    pe_key = "MultiHeadAttention.positional_encoding.pe"
    nconv_key = "nconv.weight"
    lag = int(state_dict[pe_key].shape[1]) if pe_key in state_dict else lag_default
    horizon = int(state_dict[nconv_key].shape[0]) if nconv_key in state_dict else horizon_default

    A = _build_adj(num_nodes, edges)
    saved_algo = afdgcn_mod.ALGO
    try:
        afdgcn_mod.ALGO = "Garnoldi"
        net = afdgcn_mod.Model(
            num_node   = num_nodes,
            input_dim  = 1,
            hidden_dim = 64,
            output_dim = 1,
            embed_dim  = e_dim,
            cheb_k     = 2,
            horizon    = horizon,
            num_layers = 1,
            heads      = 4,
            timesteps  = lag,
            A          = A,
            kernel_size= 5,
        )
    finally:
        afdgcn_mod.ALGO = saved_algo

    net.load_state_dict(state_dict, strict=False)
    net.eval()
    return net, lag, horizon


def _try_load(path: Path, num_nodes: int, lag_default: int, horizon_default: int, edges: Sequence[GraphEdge]):
    """Verilen .pth dosyasından AFDGCN Model yüklemeyi dener."""
    raw = torch.load(path, map_location="cpu", weights_only=True)
    state_dict, ckpt_mean, ckpt_std = _load_state_dict_payload(raw)
    result = _build_net_from_state_dict(state_dict, num_nodes, lag_default, horizon_default, edges)
    if result is None:
        return None
    net, lag, horizon = result
    raw_flag = state_dict if isinstance(state_dict, dict) else {}
    outputs_raw = bool(raw_flag.pop('real_value_output', True))
    logger.info("Model yüklendi: %s (%d node, lag=%d)", path.name, num_nodes, lag)
    return net, lag, horizon, ckpt_mean, ckpt_std, outputs_raw


def _try_load_from_bytes(data: bytes, num_nodes: int, lag_default: int, horizon_default: int, edges: Sequence[GraphEdge]):
    """Ham .pth bytes'ından AFDGCN modelini yükler. _try_load() ile aynı mantık."""
    import io

    buf = io.BytesIO(data)
    raw = torch.load(buf, map_location="cpu", weights_only=True)
    state_dict, ckpt_mean, ckpt_std = _load_state_dict_payload(raw)
    result = _build_net_from_state_dict(state_dict, num_nodes, lag_default, horizon_default, edges)
    if result is None:
        return None
    net, lag, horizon = result
    raw_flag = state_dict if isinstance(state_dict, dict) else {}
    outputs_raw = bool(raw_flag.pop('real_value_output', True))
    logger.info("Model bytes'dan yüklendi (%d node, lag=%d)", num_nodes, lag)
    return net, lag, horizon, ckpt_mean, ckpt_std, outputs_raw


async def reload_model(
    region: str,
    model_path: str,
    num_nodes: int,
    lag: int = 1,
    horizon: int = 1,
    scaler_mean: float = 0.0,
    scaler_std: float = 1.0,
    node_map: Optional[Dict[int, Dict[str, int]]] = None,
    graph_edges: Optional[Sequence[GraphEdge]] = None,
) -> tuple:
    """Verilen .pth dosyasından modeli sıcak yükler. Returns (success, message)."""
    path_obj = Path(model_path)
    if not path_obj.exists():
        return False, f"Dosya bulunamadı: {model_path}"

    async with _MODELS_LOCK:
        state = _state_for(region)
        state.num_nodes = num_nodes
        state.scaler_mean = scaler_mean
        state.scaler_std = scaler_std
        state.model_path = str(path_obj)
        if node_map is not None:
            state.node_map = node_map
        if graph_edges is not None:
            state.graph_edges = list(graph_edges)

        state.model = None
        state.model_ready = False
        state.history = []

        try:
            result = _try_load(path_obj, num_nodes, lag, horizon, state.graph_edges)
            if result is None:
                state.model_ready = True
                return False, f"Model yüklenemedi: {path_obj.name} (node boyutu uyumsuz)"
            net, resolved_lag, resolved_horizon, ckpt_mean, ckpt_std, outputs_raw = result
            state.model = net
            state.lag = resolved_lag
            state.horizon = resolved_horizon
            state.model_outputs_raw = outputs_raw
            if ckpt_mean is not None and ckpt_std is not None:
                state.scaler_mean = float(ckpt_mean)
                state.scaler_std = float(ckpt_std)
            state.model_ready = True
            return True, f"Model yüklendi: {path_obj.name} ({num_nodes} node, lag={resolved_lag})"
        except Exception as exc:
            state.model_ready = True
            return False, f"Model yükleme hatası: {exc}"


async def reload_model_from_bytes(
    region: str,
    weights: bytes,
    num_nodes: int,
    lag: int = 1,
    horizon: int = 1,
    scaler_mean: float = 0.0,
    scaler_std: float = 1.0,
    node_map: Optional[Dict[int, Dict[str, int]]] = None,
    graph_edges: Optional[Sequence[GraphEdge]] = None,
) -> tuple:
    """Ham .pth bytes'ından modeli sıcak yükler (dosya sistemi gerekmez). Returns (success, message)."""
    async with _MODELS_LOCK:
        state = _state_for(region)
        state.num_nodes = num_nodes
        state.scaler_mean = scaler_mean
        state.scaler_std = scaler_std
        if node_map is not None:
            state.node_map = node_map
        if graph_edges is not None:
            state.graph_edges = list(graph_edges)

        state.model = None
        state.model_ready = False
        state.history = []

        try:
            result = _try_load_from_bytes(weights, num_nodes, lag, horizon, state.graph_edges)
            if result is None:
                state.model_ready = True
                return False, "Model bytes'dan yuklenemedi (node boyutu uyumsuz)"
            net, resolved_lag, resolved_horizon, ckpt_mean, ckpt_std, outputs_raw = result
            state.model = net
            state.lag = resolved_lag
            state.horizon = resolved_horizon
            state.model_outputs_raw = outputs_raw
            if ckpt_mean is not None and ckpt_std is not None:
                state.scaler_mean = float(ckpt_mean)
                state.scaler_std = float(ckpt_std)
            state.model_ready = True
            return True, f"Model bytes'dan yuklendi ({num_nodes} node, lag={resolved_lag})"
        except Exception as exc:
            state.model_ready = True
            return False, f"Model bytes yükleme hatası: {exc}"


async def ensure_model_loaded(region: str) -> None:
    """Bölge için model hazır değilse hazır olarak işaretler (fallback moving-average)."""
    state = _state_for(region)
    if state.model_ready:
        return
    async with _MODELS_LOCK:
        if state.model_ready:
            return
        logger.warning("⚠️ '%s' için yüklü model yok → moving average kullanılacak", region)
        state.model_ready = True


# ─────────────────────────────────────────────────────────────────────────────
# Normalizasyon
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(x: np.ndarray, state: ModelState) -> np.ndarray:
    return (x - state.scaler_mean) / (state.scaler_std + 1e-8)


def _denormalize(x: np.ndarray, state: ModelState) -> np.ndarray:
    return x * state.scaler_std + state.scaler_mean


# ─────────────────────────────────────────────────────────────────────────────
# Veri dönüşümü
# ─────────────────────────────────────────────────────────────────────────────

def _belediye_to_node_vector(
    data_by_junction: Dict[int, List[dict]],
    minute_index: int,
    state: ModelState,
) -> np.ndarray:
    """Belediye API cevabından belirli bir zaman dilimi için node vektörü üretir."""
    vec = np.zeros(state.num_nodes, dtype=np.float32)
    for jid, arms in data_by_junction.items():
        arm_map = state.node_map.get(jid, {})
        for arm_data in arms:
            direction = str(arm_data.get("edge_direction", "")).strip().upper()
            node_idx = arm_map.get(direction)
            if node_idx is not None:
                vec[node_idx] = float(arm_data.get(str(minute_index), 0))
    return vec


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def _run_forward(net, x_norm: np.ndarray) -> np.ndarray:
    """Senkron AFDGCN forward pass. x_norm: (lag, num_nodes)"""
    import model.AFDGCN as afdgcn_mod

    x_t = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    saved = afdgcn_mod.ALGO
    try:
        afdgcn_mod.ALGO = "Garnoldi"
        with torch.no_grad():
            out = net(x_t)  # (1, horizon, N, 1) veya (1, horizon, N)
    finally:
        afdgcn_mod.ALGO = saved

    out_np = out.squeeze(0).squeeze(-1).detach().cpu().numpy()
    if out_np.ndim == 1:
        out_np = out_np[np.newaxis, :]
    return out_np  # (horizon, N)


def _moving_average_predict(history: np.ndarray, num_nodes: int) -> np.ndarray:
    """Basit moving average + trend tahmini. history: (lag, num_nodes)"""
    if history.shape[0] >= 2:
        trend = (history[-1] - history[-2]) * 0.5
    else:
        trend = np.zeros(num_nodes, dtype=np.float32)
    pred = np.clip(history[-1] + trend, 0, None)
    return pred[np.newaxis, :]


# ─────────────────────────────────────────────────────────────────────────────
# Ana Tahmin Fonksiyonları
# ─────────────────────────────────────────────────────────────────────────────

async def predict_next_timestep(
    region: str,
    data_by_junction: Dict[int, List[dict]],
    current_minute_index: int,
) -> Dict[int, Dict[str, float]]:
    """
    Sonraki zaman dilimi için kavşak kolu bazlı araç sayısı tahmini.

    Returns
    -------
    {junction_id: {arm_letter: predicted_count}}
    """
    await ensure_model_loaded(region)
    state = _state_for(region)

    lag_needed = state.lag if state.model is not None else 1

    if len(state.history) < lag_needed:
        state.history = []
        start_idx = max(0, current_minute_index - lag_needed)
        for i in range(start_idx, current_minute_index):
            state.history.append(_belediye_to_node_vector(data_by_junction, i, state))

    current_vec = _belediye_to_node_vector(data_by_junction, current_minute_index, state)
    state.history.append(current_vec)
    if len(state.history) > state.max_history:
        state.history.pop(0)

    if len(state.history) < lag_needed:
        pad = [state.history[0]] * (lag_needed - len(state.history))
        history_arr = np.stack(pad + state.history, axis=0)
    else:
        history_arr = np.stack(state.history[-lag_needed:], axis=0)

    history_norm = _normalize(history_arr, state)

    if state.model is not None:
        async with _INFERENCE_LOCK:
            loop = asyncio.get_event_loop()
            model_out = await loop.run_in_executor(
                None, _run_forward, state.model, history_norm
            )
        pred = np.clip(model_out, 0, None) if state.model_outputs_raw else np.clip(_denormalize(model_out, state), 0, None)
    else:
        pred_norm = _moving_average_predict(history_norm, state.num_nodes)
        pred = np.clip(_denormalize(pred_norm, state), 0, None)

    result: Dict[int, Dict[str, float]] = {}
    for jid, arm_map in state.node_map.items():
        result[jid] = {
            arm: float(pred[0, node_idx])
            for arm, node_idx in arm_map.items()
        }
    return result


def _predict_rolling_series_sync(
    region: str,
    data_by_junction: Dict[int, List[dict]],
    completed_idx: int,
) -> Dict[int, Dict[str, List[float]]]:
    """
    Gün başından completed_idx dahil her slot için gerçek bir-adım-ileride tahmin üretir.

    Slot i tahmini: model(actual[i-lag .. i-1]) → slot i araç sayısı
    Yani tahmin üretilirken slot i'nin gerçek değeri KULLANILMAZ (data leakage yok).

    Returns: {jid: {arm: [pred_slot_0, pred_slot_1, ..., pred_slot_completed_idx]}}
    """
    state = _state_for(region)
    n_slots = completed_idx + 1
    lag = state.lag if state.model is not None else 1

    actual = np.zeros((n_slots, state.num_nodes), dtype=np.float32)
    for s in range(n_slots):
        actual[s] = _belediye_to_node_vector(data_by_junction, s, state)

    result_series = np.zeros((n_slots, state.num_nodes), dtype=np.float32)

    for i in range(n_slots):
        if i == 0:
            window = np.zeros((lag, state.num_nodes), dtype=np.float32)
        elif i < lag:
            pad = np.zeros((lag - i, state.num_nodes), dtype=np.float32)
            window = np.vstack([pad, actual[:i]])
        else:
            window = actual[i - lag : i]

        window_norm = _normalize(window, state)

        if state.model is not None:
            model_out = _run_forward(state.model, window_norm)
            pred = np.clip(model_out, 0, None) if state.model_outputs_raw else np.clip(_denormalize(model_out, state), 0, None)
        else:
            pred_norm = _moving_average_predict(window_norm, state.num_nodes)
            pred = np.clip(_denormalize(pred_norm, state), 0, None)

        result_series[i] = pred[0]

    out: Dict[int, Dict[str, List[float]]] = {}
    for jid, arm_map in state.node_map.items():
        out[jid] = {
            arm: [float(result_series[s, node_idx]) for s in range(n_slots)]
            for arm, node_idx in arm_map.items()
        }
    return out


async def predict_rolling_series(
    region: str,
    data_by_junction: Dict[int, List[dict]],
    completed_idx: int,
) -> Dict[int, Dict[str, List[float]]]:
    """
    Async wrapper — rolling tahmin serisini döndürür.
    Tüm forward pass'lar inference lock altında tek bir iş parçacığında çalışır.
    """
    await ensure_model_loaded(region)
    async with _INFERENCE_LOCK:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            _predict_rolling_series_sync,
            region,
            data_by_junction,
            completed_idx,
        )


def get_model_status(region: Optional[str] = None) -> dict:
    """Wrapper durumunu döndürür (sağlık endpoint'i için). region=None ise tüm bölgeler."""
    def _status(state: ModelState) -> dict:
        return {
            "model_loaded": state.model is not None,
            "model_path": state.model_path,
            "num_nodes": state.num_nodes,
            "lag": state.lag,
            "horizon": state.horizon,
            "scaler_mean": state.scaler_mean,
            "scaler_std": state.scaler_std,
            "fallback_active": state.model is None,
            "history_length": len(state.history),
        }

    if region is not None:
        return _status(_state_for(region))
    return {r: _status(s) for r, s in _MODELS.items()}


def clear_history(region: str) -> None:
    """Geçmiş tamponu temizler (test/yeniden başlatma için)."""
    _state_for(region).history = []
