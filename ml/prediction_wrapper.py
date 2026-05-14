"""
İldem Bölgesi AFDGCN Tahmin Wrapper
====================================

Belediye API verisi → AFDGCN model → sonraki 10 dakika araç sayısı tahmini.

34 node mapping:
  Gesi       (89)  : A=0, B=1, C=2, D=3
  Serkent    (187) : A=4, B=5, C=6, D=7
  Beyazşehir (95)  : A=8, B=9, C=10, D=11
  Toki       (121) : A=12, B=13, C=14, D=15
  İldem 1    (184) : A=16, B=17, D=18          (C yok)
  İldem 2    (188) : A=19, B=20, C=21, D=22
  İldem 3    (117) : A=23, C=24, D=25           (B yok)
  İldem 4    (192) : A=26, B=27, C=28, D=29
  İldem 5    (194) : A=30, B=31, C=32, D=33

Model yoksa veya boyut uyuşmazlığı varsa moving average fallback devreye girer.
Kayseri_Serit conf: lag=1, horizon=1, normalizer=std
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# ml/ → proje kökü
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Varsayılan Sabitler (reload_model() ile değiştirilebilir)
# ─────────────────────────────────────────────────────────────────────────────

NUM_NODES: int = 34
LAG: int = 1
HORIZON: int = 1
SCALER_MEAN: float = 28.53
SCALER_STD: float = 38.72

# junction_id → {arm_letter: node_index}
ILDEM_NODE_MAP: Dict[int, Dict[str, int]] = {
    89:  {"A": 0,  "B": 1,  "C": 2,  "D": 3},
    187: {"A": 4,  "B": 5,  "C": 6,  "D": 7},
    95:  {"A": 8,  "B": 9,  "C": 10, "D": 11},
    121: {"A": 12, "B": 13, "C": 14, "D": 15},
    184: {"A": 16, "B": 17,           "D": 18},   # C yok
    188: {"A": 19, "B": 20, "C": 21, "D": 22},
    117: {"A": 23,           "C": 24, "D": 25},   # B yok
    192: {"A": 26, "B": 27, "C": 28, "D": 29},
    194: {"A": 30, "B": 31, "C": 32, "D": 33},
}

# Kavşak adları
JUNCTION_NAMES: Dict[int, str] = {
    89: "Gesi", 187: "Serkent", 95: "Beyazşehir", 121: "Toki",
    184: "İldem 1", 188: "İldem 2", 117: "İldem 3",
    192: "İldem 4", 194: "İldem 5",
}

# Kol adları
ARM_NAMES: Dict[int, Dict[str, str]] = {
    89: {
        "A": "SİVAS BULVARI-SİVAS YÖNÜ",
        "B": "GESİ CAD.",
        "C": "SİVAS BULV-ŞEHİR MERKEZİ",
        "D": "381. SOKAK",
    },
    187: {
        "A": "822. SK",
        "B": "GESİ CAD. DOĞU",
        "C": "KOCASİNAN CAD.",
        "D": "GESİ CAD. BATI",
    },
    95: {
        "A": "OSMAN ÖZCAN CAD.",
        "B": "GESİ CAD DOĞU",
        "C": "MARKETLER ÇIKIŞ",
        "D": "GESİ CAD BATI",
    },
    121: {
        "A": "832. CD",
        "B": "GESİ CAD. DOĞU",
        "C": "KADİR HAS BUL.",
        "D": "GESİ CAD BATI",
    },
    184: {
        "A": "DİNÇER SOKAK",
        "B": "ALPARSLAN TÜRKEŞ BUL.",
        "D": "GESİ CAD",
    },
    188: {
        "A": "DİNÇER SOKAK KUZEY",
        "B": "HANEDAN SOKAK",
        "C": "DİNÇER SOKAK GÜNEY",
        "D": "FETİH SOKAK",
    },
    117: {
        "A": "DÜNDAR TAŞER CAD. KUZEY",
        "C": "DÜNDAR TAŞER CAD. GÜNEY",
        "D": "HANEDAN SOKAK",
    },
    192: {
        "A": "YAVUZ SULTAN SELİM CAD. KUZEY",
        "B": "VATAN SOKAK",
        "C": "YAVUZ SULTAN SELİM CAD. GÜNEY",
        "D": "ORKUN SOKAK",
    },
    194: {
        "A": "S.A BEDÜK CAD. KUZEY",
        "B": "VATAN SOKAK BATI",
        "C": "S.A BEDÜK CAD. GÜNEY",
        "D": "VATAN SOKAK DOĞU",
    },
}

# DB'den yüklenen model buraya önbelleğe alınır (Python yeniden başladığında kullanılır)
_CACHED_MODEL_PATH = PROJECT_ROOT / "saved_models" / "_active_model_cache.pth"

# Varsayılan model arama sırası — önce önbellek, sonra gerçek model dosyaları
_DEFAULT_MODEL_PATHS = [
    _CACHED_MODEL_PATH,
    PROJECT_ROOT / "saved_models" / "kayseri_ildem_34.pth",
    PROJECT_ROOT / "saved_models" / "kayseri_ildem_34_dummy.pth",
]

# Aktif model dosya yolu (DB'den aktivasyon yapıldığında güncellenir)
_requested_model_path: Optional[Path] = None

GRAPH_PATH = PROJECT_ROOT / "data" / "Kayseri" / "kol_bazli_graph.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Global model state
# ─────────────────────────────────────────────────────────────────────────────
_model: Optional[object] = None
_model_ready: bool = False
_model_lag: int = LAG
_model_lock = asyncio.Lock()

_MAX_HISTORY = 12
_history: List[np.ndarray] = []

# Aktif node map (region'a göre değişebilir)
_node_map: Dict[int, Dict[str, int]] = ILDEM_NODE_MAP.copy()

# ─────────────────────────────────────────────────────────────────────────────
# Graf yükleme
# ─────────────────────────────────────────────────────────────────────────────

def _build_adj() -> torch.Tensor:
    """kol_bazli_graph.csv'den 34×34 adjacency tensörü oluşturur."""
    A = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)
    try:
        import pandas as pd
        df = pd.read_csv(GRAPH_PATH)
        fc, tc = df.columns[0], df.columns[1]
        for _, row in df.iterrows():
            i, j = int(row[fc]), int(row[tc])
            if 0 <= i < NUM_NODES and 0 <= j < NUM_NODES:
                A[i, j] = 1.0
        logger.info("Graf yüklendi: %d kenar", int(A.sum()))
    except Exception as exc:
        logger.warning("Graf yüklenemedi, sıfır matris kullanılıyor: %s", exc)
    return torch.tensor(A)


# ─────────────────────────────────────────────────────────────────────────────
# Model yükleme
# ─────────────────────────────────────────────────────────────────────────────

def _try_load(path: Path):
    """
    Verilen .pth dosyasından AFDGCN Model yüklemeyi dener.
    node_embedding boyutu NUM_NODES ile uyuşmuyorsa None döner.
    """
    import model.AFDGCN as afdgcn_mod

    state_dict = torch.load(path, map_location="cpu", weights_only=True)

    # Checkpoint dict formatı (scaler bilgisi içerebilir)
    _ckpt_scaler_mean = None
    _ckpt_scaler_std = None
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        _ckpt_scaler_mean = state_dict.get('scaler_mean')
        _ckpt_scaler_std = state_dict.get('scaler_std')
        state_dict = state_dict['state_dict']

    if "node_embedding" not in state_dict:
        logger.warning("%s: node_embedding bulunamadı", path.name)
        return None

    n_nodes, e_dim = state_dict["node_embedding"].shape
    if n_nodes != NUM_NODES:
        logger.warning(
            "%s: %d node iceriyor, %d bekleniyor - atlaniyor",
            path.name, n_nodes, NUM_NODES,
        )
        return None

    pe_key = "MultiHeadAttention.positional_encoding.pe"
    nconv_key = "nconv.weight"
    lag = int(state_dict[pe_key].shape[1]) if pe_key in state_dict else LAG
    horizon = int(state_dict[nconv_key].shape[0]) if nconv_key in state_dict else HORIZON

    A = _build_adj()
    saved_algo = afdgcn_mod.ALGO
    try:
        afdgcn_mod.ALGO = "Garnoldi"
        net = afdgcn_mod.Model(
            num_node   = NUM_NODES,
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
    if _ckpt_scaler_mean is not None and _ckpt_scaler_std is not None:
        global SCALER_MEAN, SCALER_STD
        SCALER_MEAN = float(_ckpt_scaler_mean)
        SCALER_STD = float(_ckpt_scaler_std)
        logger.info("Scaler .pth'dan yüklendi: mean=%.4f std=%.4f", SCALER_MEAN, SCALER_STD)
    logger.info("✅ Model yüklendi: %s (%d node, lag=%d)", path.name, NUM_NODES, lag)
    return net, lag


async def ensure_model_loaded() -> None:
    """İlk istekte modeli lazy-load eder."""
    global _model, _model_ready, _model_lag
    if _model_ready:
        return
    async with _model_lock:
        if _model_ready:
            return
        paths_to_try = [_requested_model_path] if _requested_model_path else _DEFAULT_MODEL_PATHS
        for path in paths_to_try:
            if path.exists():
                try:
                    result = _try_load(path)
                    if result is not None:
                        _model, _model_lag = result
                        _model_ready = True
                        return
                except Exception as exc:
                    logger.warning("Model yükleme hatası (%s): %s", path.name, exc)
        logger.warning("⚠️ Hiçbir model bulunamadı → moving average kullanılacak")
        _model_ready = True


def _try_load_from_bytes(data: bytes):
    """
    Ham .pth bytes'ından AFDGCN modelini yükler.
    _try_load() ile aynı doğrulama mantığını izler.
    """
    import io
    import model.AFDGCN as afdgcn_mod

    buf = io.BytesIO(data)
    state_dict = torch.load(buf, map_location="cpu", weights_only=True)

    # Checkpoint dict formatı (scaler bilgisi içerebilir)
    _ckpt_scaler_mean = None
    _ckpt_scaler_std = None
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        _ckpt_scaler_mean = state_dict.get('scaler_mean')
        _ckpt_scaler_std = state_dict.get('scaler_std')
        state_dict = state_dict['state_dict']

    if "node_embedding" not in state_dict:
        logger.warning("bytes model: node_embedding bulunamadi")
        return None

    n_nodes, e_dim = state_dict["node_embedding"].shape
    if n_nodes != NUM_NODES:
        logger.warning(
            "bytes model: %d node iceriyor, %d bekleniyor - atlaniyor",
            n_nodes, NUM_NODES,
        )
        return None

    pe_key = "MultiHeadAttention.positional_encoding.pe"
    nconv_key = "nconv.weight"
    lag = int(state_dict[pe_key].shape[1]) if pe_key in state_dict else LAG
    horizon = int(state_dict[nconv_key].shape[0]) if nconv_key in state_dict else HORIZON

    A = _build_adj()
    saved_algo = afdgcn_mod.ALGO
    try:
        afdgcn_mod.ALGO = "Garnoldi"
        net = afdgcn_mod.Model(
            num_node   = NUM_NODES,
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
    if _ckpt_scaler_mean is not None and _ckpt_scaler_std is not None:
        global SCALER_MEAN, SCALER_STD
        SCALER_MEAN = float(_ckpt_scaler_mean)
        SCALER_STD = float(_ckpt_scaler_std)
        logger.info("Scaler bytes'dan yüklendi: mean=%.4f std=%.4f", SCALER_MEAN, SCALER_STD)
    logger.info("Model bytes'dan yuklendi (%d node, lag=%d)", NUM_NODES, lag)
    return net, lag


async def reload_model(
    model_path: str,
    num_nodes: int = 34,
    lag: int = 1,
    horizon: int = 1,
    scaler_mean: float = 28.53,
    scaler_std: float = 38.72,
    node_map: Optional[Dict[int, Dict[str, int]]] = None,
) -> tuple:
    """
    Verilen .pth dosyasından modeli sıcak yeniden yükler.
    Returns (success: bool, message: str)
    """
    global _model, _model_ready, _model_lag, _history
    global NUM_NODES, LAG, HORIZON, SCALER_MEAN, SCALER_STD
    global _requested_model_path, _node_map

    path_obj = Path(model_path)
    if not path_obj.exists():
        return False, f"Dosya bulunamadı: {model_path}"

    async with _model_lock:
        NUM_NODES = num_nodes
        LAG = lag
        HORIZON = horizon
        SCALER_MEAN = scaler_mean
        SCALER_STD = scaler_std
        _requested_model_path = path_obj
        if node_map is not None:
            _node_map = node_map

        _model = None
        _model_ready = False
        _history = []

        try:
            result = _try_load(path_obj)
            if result is not None:
                _model, _model_lag = result
                _model_ready = True
                return True, f"Model yüklendi: {path_obj.name} ({num_nodes} node, lag={_model_lag})"
            else:
                _model_ready = True
                return False, f"Model yüklenemedi: {path_obj.name} (node boyutu uyumsuz)"
        except Exception as exc:
            _model_ready = True
            return False, f"Model yükleme hatası: {exc}"


async def reload_model_from_bytes(
    weights: bytes,
    num_nodes: int = 34,
    lag: int = 1,
    horizon: int = 1,
    scaler_mean: float = 28.53,
    scaler_std: float = 38.72,
    node_map: Optional[Dict[int, Dict[str, int]]] = None,
) -> tuple:
    """
    Ham .pth bytes'ından modeli sıcak yeniden yükler (dosya sistemi gerekmez).
    Başarıyla yüklenirse gelecek başlatmalar için diske önbelleğe alınır.
    Returns (success: bool, message: str)
    """
    global _model, _model_ready, _model_lag, _history
    global NUM_NODES, LAG, HORIZON, SCALER_MEAN, SCALER_STD
    global _node_map

    async with _model_lock:
        NUM_NODES = num_nodes
        LAG = lag
        HORIZON = horizon
        SCALER_MEAN = scaler_mean
        SCALER_STD = scaler_std
        if node_map is not None:
            _node_map = node_map

        _model = None
        _model_ready = False
        _history = []

        try:
            result = _try_load_from_bytes(weights)
            if result is not None:
                _model, _model_lag = result
                _model_ready = True
                # Bir sonraki Python yeniden başlatmasında kullanmak için diske kaydet
                try:
                    _CACHED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                    _CACHED_MODEL_PATH.write_bytes(weights)
                    logger.info("Model önbelleğe alındı: %s", _CACHED_MODEL_PATH.name)
                except Exception as cache_err:
                    logger.warning("Model önbelleğe alınamadı: %s", cache_err)
                return True, f"Model bytes'dan yuklendi ({num_nodes} node, lag={_model_lag})"
            else:
                _model_ready = True
                return False, "Model bytes'dan yuklenemedi (node boyutu uyumsuz)"
        except Exception as exc:
            _model_ready = True
            return False, f"Model bytes yükleme hatası: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Normalizasyon
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(x: np.ndarray) -> np.ndarray:
    return (x - SCALER_MEAN) / (SCALER_STD + 1e-8)


def _denormalize(x: np.ndarray) -> np.ndarray:
    return x * SCALER_STD + SCALER_MEAN


# ─────────────────────────────────────────────────────────────────────────────
# Veri dönüşümü
# ─────────────────────────────────────────────────────────────────────────────

def belediye_to_node_vector(
    data_by_junction: Dict[int, List[dict]],
    minute_index: int,
) -> np.ndarray:
    """
    Belediye API cevabından belirli bir 10-dakika dilimi için 34-node vektörü üretir.
    """
    vec = np.zeros(NUM_NODES, dtype=np.float32)
    for jid, arms in data_by_junction.items():
        arm_map = _node_map.get(jid, {})
        for arm_data in arms:
            direction = str(arm_data.get("edge_direction", "")).strip().upper()
            node_idx = arm_map.get(direction)
            if node_idx is not None:
                vec[node_idx] = float(arm_data.get(str(minute_index), 0))
    return vec


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

_INFERENCE_LOCK = asyncio.Lock()


def _run_forward(net, x_norm: np.ndarray) -> np.ndarray:
    """Senkron AFDGCN forward pass. x_norm: (lag, NUM_NODES)"""
    import model.AFDGCN as afdgcn_mod

    x_t = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    saved = afdgcn_mod.ALGO
    try:
        afdgcn_mod.ALGO = "Garnoldi"
        with torch.no_grad():
            out = net(x_t)  # (1, horizon, 34, 1) veya (1, horizon, 34)
    finally:
        afdgcn_mod.ALGO = saved

    out_np = out.squeeze(0).squeeze(-1).detach().cpu().numpy()
    if out_np.ndim == 1:
        out_np = out_np[np.newaxis, :]
    return out_np  # (horizon, 34)


def _moving_average_predict(history: np.ndarray) -> np.ndarray:
    """Basit moving average + trend tahmini. history: (lag, NUM_NODES)"""
    if history.shape[0] >= 2:
        trend = (history[-1] - history[-2]) * 0.5
    else:
        trend = np.zeros(NUM_NODES, dtype=np.float32)
    pred = np.clip(history[-1] + trend, 0, None)
    return pred[np.newaxis, :]


# ─────────────────────────────────────────────────────────────────────────────
# Ana Tahmin Fonksiyonu
# ─────────────────────────────────────────────────────────────────────────────

async def predict_next_timestep(
    data_by_junction: Dict[int, List[dict]],
    current_minute_index: int,
) -> Dict[int, Dict[str, float]]:
    """
    Sonraki 10 dakika için kavşak kolu bazlı araç sayısı tahmini.

    Returns
    -------
    {junction_id: {arm_letter: predicted_count}}
    """
    global _history

    await ensure_model_loaded()

    lag_needed = _model_lag if _model is not None else LAG

    # Sunucu yeni başladıysa veya geçmiş yetersizse API verisinden önceki slotları doldur
    if len(_history) < lag_needed:
        _history = []
        start_idx = max(0, current_minute_index - lag_needed)
        for i in range(start_idx, current_minute_index):
            _history.append(belediye_to_node_vector(data_by_junction, i))

    current_vec = belediye_to_node_vector(data_by_junction, current_minute_index)
    _history.append(current_vec)
    if len(_history) > _MAX_HISTORY:
        _history.pop(0)

    if len(_history) < lag_needed:
        pad = [_history[0]] * (lag_needed - len(_history))
        history_arr = np.stack(pad + _history, axis=0)
    else:
        history_arr = np.stack(_history[-lag_needed:], axis=0)

    history_norm = _normalize(history_arr)

    if _model is not None:
        async with _INFERENCE_LOCK:
            loop = asyncio.get_event_loop()
            pred_norm = await loop.run_in_executor(
                None, _run_forward, _model, history_norm
            )
    else:
        pred_norm = _moving_average_predict(history_norm)

    pred = _denormalize(pred_norm)
    pred = np.clip(pred, 0, None)

    result: Dict[int, Dict[str, float]] = {}
    for jid, arm_map in ILDEM_NODE_MAP.items():
        result[jid] = {
            arm: float(pred[0, node_idx])
            for arm, node_idx in arm_map.items()
        }
    return result


def _predict_rolling_series_sync(
    data_by_junction: Dict[int, List[dict]],
    completed_idx: int,
) -> Dict[int, Dict[str, List[float]]]:
    """
    Gün başından completed_idx dahil her slot için gerçek bir-adım-ileride tahmin üretir.

    Slot i tahmini: model(actual[i-lag .. i-1]) → slot i araç sayısı
    Yani tahmin üretilirken slot i'nin gerçek değeri KULLANILMAZ (data leakage yok).

    Returns: {jid: {arm: [pred_slot_0, pred_slot_1, ..., pred_slot_completed_idx]}}
    """
    n_slots = completed_idx + 1
    lag = _model_lag if _model is not None else LAG

    # Tüm slotlar için gerçek vektörleri önceden hazırla
    actual = np.zeros((n_slots, NUM_NODES), dtype=np.float32)
    for s in range(n_slots):
        actual[s] = belediye_to_node_vector(data_by_junction, s)

    result_series = np.zeros((n_slots, NUM_NODES), dtype=np.float32)

    for i in range(n_slots):
        # Slot i'yi predict etmek için [i-lag .. i-1] penceresini kullan
        if i == 0:
            # Hiç geçmiş yok — sıfır vektörüyle tahmin
            window = np.zeros((lag, NUM_NODES), dtype=np.float32)
        elif i < lag:
            # Yeterli geçmiş yok — sola sıfır dolgusuy
            pad = np.zeros((lag - i, NUM_NODES), dtype=np.float32)
            window = np.vstack([pad, actual[:i]])
        else:
            window = actual[i - lag : i]

        window_norm = _normalize(window)

        if _model is not None:
            pred_norm = _run_forward(_model, window_norm)
        else:
            pred_norm = _moving_average_predict(window_norm)

        pred = _denormalize(pred_norm)
        pred = np.clip(pred, 0, None)
        result_series[i] = pred[0]  # (NUM_NODES,)

    # Node vektörünü kavşak/kol sözlüğüne çevir
    out: Dict[int, Dict[str, List[float]]] = {}
    for jid, arm_map in _node_map.items():
        out[jid] = {
            arm: [float(result_series[s, node_idx]) for s in range(n_slots)]
            for arm, node_idx in arm_map.items()
        }
    return out


async def predict_rolling_series(
    data_by_junction: Dict[int, List[dict]],
    completed_idx: int,
) -> Dict[int, Dict[str, List[float]]]:
    """
    Async wrapper — rolling tahmin serisini döndürür.
    Tüm forward pass'lar inference lock altında tek bir iş parçacığında çalışır.
    """
    await ensure_model_loaded()
    async with _INFERENCE_LOCK:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            _predict_rolling_series_sync,
            data_by_junction,
            completed_idx,
        )


def get_model_status() -> dict:
    """Wrapper durumunu döndürür (sağlık endpoint'i için)."""
    return {
        "model_loaded": _model is not None,
        "model_path": str(_requested_model_path) if _requested_model_path else None,
        "num_nodes": NUM_NODES,
        "lag": _model_lag,
        "horizon": HORIZON,
        "scaler_mean": SCALER_MEAN,
        "scaler_std": SCALER_STD,
        "fallback_active": _model is None,
        "history_length": len(_history),
    }


def clear_history() -> None:
    """Geçmiş tamponu temizler (test/yeniden başlatma için)."""
    global _history
    _history = []
