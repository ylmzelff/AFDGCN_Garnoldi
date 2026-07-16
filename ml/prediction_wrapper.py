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
LAG: int = 12
HORIZON: int = 144        # tam gun (144 slot) icin varsayilan ufuk
SCALER_MEAN: float = 28.53
SCALER_STD: float = 38.72
SLOTS_PER_DAY: int = 144  # 10-dakikalik araliklar

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
    PROJECT_ROOT / "saved_models" / "kayseri_ildem_v1.pth",
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

# True ise model zaten raw (denormalize) değer çıkarıyor, _denormalize() uygulanmaz.
_model_outputs_raw: bool = True

# Modelin beklediği input_dim (1=sadece flow, 3=flow+sin_tod+cos_tod)
_model_input_dim: int = 1
# True ise inference sirasinda sin/cos zaman-icin-gun ozellikleri eklenir
_model_tod_enabled: bool = False

_MAX_HISTORY = 24  # lag=12 icin en az 12, biraz pay birakiyoruz
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

def _detect_input_dim(state_dict: dict) -> int:
    """
    Checkpoint'ten input_dim'i okur. Kaydedildiyse 'input_dim' anahtarindan,
    yoksa 'feature_attention.weight' seklindan cikarir, bulamazsa 1 doner.
    """
    if 'input_dim' in state_dict:
        return int(state_dict['input_dim'])
    # feature_attention (1D conv): weight shape (out_ch, in_ch, kernel_size)
    fa_key = 'feature_attention.weight'
    if fa_key in state_dict:
        return int(state_dict[fa_key].shape[1])
    return 1


def _try_load(path: Path, input_dim: int = 1, tod_enabled: bool = False):
    """
    Verilen .pth dosyasından AFDGCN Model yüklemeyi dener.
    node_embedding boyutu NUM_NODES ile uyuşmuyorsa None döner.
    """
    import model.AFDGCN as afdgcn_mod

    state_dict = torch.load(path, map_location="cpu", weights_only=True)

    # Checkpoint dict formatı (scaler bilgisi içerebilir)
    _ckpt_scaler_mean = None
    _ckpt_scaler_std = None
    _ckpt_input_dim = None
    _ckpt_tod = None
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        _ckpt_scaler_mean = state_dict.get('scaler_mean')
        _ckpt_scaler_std = state_dict.get('scaler_std')
        _ckpt_input_dim = state_dict.get('input_dim')
        _ckpt_tod = state_dict.get('tod_enabled')
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

    # input_dim: checkpoint'ten > parametre > otomatik tespitle
    resolved_input_dim = _ckpt_input_dim or input_dim or _detect_input_dim(state_dict)
    resolved_tod = _ckpt_tod if _ckpt_tod is not None else tod_enabled

    A = _build_adj()
    saved_algo = afdgcn_mod.ALGO
    try:
        afdgcn_mod.ALGO = "Garnoldi"
        net = afdgcn_mod.Model(
            num_node   = NUM_NODES,
            input_dim  = resolved_input_dim,
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
    global SCALER_MEAN, SCALER_STD, _model_outputs_raw, _model_input_dim, _model_tod_enabled
    if _ckpt_scaler_mean is not None and _ckpt_scaler_std is not None:
        SCALER_MEAN = float(_ckpt_scaler_mean)
        SCALER_STD = float(_ckpt_scaler_std)
        logger.info("Scaler .pth'dan yüklendi: mean=%.4f std=%.4f", SCALER_MEAN, SCALER_STD)
    raw_flag = state_dict if isinstance(state_dict, dict) else {}
    _model_outputs_raw = bool(raw_flag.pop('real_value_output', True))
    _model_input_dim = resolved_input_dim
    _model_tod_enabled = bool(resolved_tod)
    logger.info("Model cikti modu: %s", 'RAW' if _model_outputs_raw else 'NORM')
    logger.info("Model yuklendi: %s (%d node, lag=%d, input_dim=%d, tod=%s)",
                path.name, NUM_NODES, lag, resolved_input_dim, resolved_tod)
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


def _try_load_from_bytes(data: bytes, input_dim: int = 1, tod_enabled: bool = False):
    """
    Ham .pth bytes'ından AFDGCN modelini yükler.
    _try_load() ile aynı doğrulama mantığını izler.
    """
    import io
    import model.AFDGCN as afdgcn_mod

    buf = io.BytesIO(data)
    state_dict = torch.load(buf, map_location="cpu", weights_only=True)

    _ckpt_scaler_mean = None
    _ckpt_scaler_std = None
    _ckpt_input_dim = None
    _ckpt_tod = None
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        _ckpt_scaler_mean = state_dict.get('scaler_mean')
        _ckpt_scaler_std = state_dict.get('scaler_std')
        _ckpt_input_dim = state_dict.get('input_dim')
        _ckpt_tod = state_dict.get('tod_enabled')
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

    resolved_input_dim = _ckpt_input_dim or input_dim or _detect_input_dim(state_dict)
    resolved_tod = _ckpt_tod if _ckpt_tod is not None else tod_enabled

    A = _build_adj()
    saved_algo = afdgcn_mod.ALGO
    try:
        afdgcn_mod.ALGO = "Garnoldi"
        net = afdgcn_mod.Model(
            num_node   = NUM_NODES,
            input_dim  = resolved_input_dim,
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
    global SCALER_MEAN, SCALER_STD, _model_outputs_raw, _model_input_dim, _model_tod_enabled
    if _ckpt_scaler_mean is not None and _ckpt_scaler_std is not None:
        SCALER_MEAN = float(_ckpt_scaler_mean)
        SCALER_STD = float(_ckpt_scaler_std)
        logger.info("Scaler bytes'dan yuklendi: mean=%.4f std=%.4f", SCALER_MEAN, SCALER_STD)
    raw_flag = state_dict if isinstance(state_dict, dict) else {}
    _model_outputs_raw = bool(raw_flag.pop('real_value_output', True))
    _model_input_dim = resolved_input_dim
    _model_tod_enabled = bool(resolved_tod)
    logger.info("Model bytes'dan yuklendi (%d node, lag=%d, input_dim=%d, tod=%s)",
                NUM_NODES, lag, resolved_input_dim, resolved_tod)
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


def _build_temporal_features(flow_window: np.ndarray, window_start_slot: int) -> np.ndarray:
    """
    flow_window: (lag, NUM_NODES) normalize edilmis arac sayisi
    window_start_slot: penceredeki ilk slotun gun-icindeki indeksi (0-143)
    Returns: (lag, NUM_NODES, 3) = [flow, sin_tod, cos_tod]
    """
    lag = flow_window.shape[0]
    slots = np.array([(window_start_slot + k) % SLOTS_PER_DAY for k in range(lag)], dtype=np.float32)
    sin_tod = np.sin(2 * np.pi * slots / SLOTS_PER_DAY)
    cos_tod = np.cos(2 * np.pi * slots / SLOTS_PER_DAY)

    flow_3d = flow_window[:, :, np.newaxis]                               # (lag, N, 1)
    sin_3d  = np.tile(sin_tod[:, np.newaxis, np.newaxis], (1, NUM_NODES, 1))  # (lag, N, 1)
    cos_3d  = np.tile(cos_tod[:, np.newaxis, np.newaxis], (1, NUM_NODES, 1))  # (lag, N, 1)
    return np.concatenate([flow_3d, sin_3d, cos_3d], axis=-1)             # (lag, N, 3)


def _run_forward(net, x_norm: np.ndarray) -> np.ndarray:
    """
    Senkron AFDGCN forward pass.
    x_norm: (lag, NUM_NODES)          → D=1 olarak islenir
            (lag, NUM_NODES, D)       → D boyutu korunur
    """
    import model.AFDGCN as afdgcn_mod

    if x_norm.ndim == 2:
        # (lag, N) → (1, lag, N, 1)
        x_t = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    else:
        # (lag, N, D) → (1, lag, N, D)
        x_t = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)

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


def _autoregress(
    net,
    window_norm: np.ndarray,   # (lag, N) normalized
    window_start_slot: int,
    n_steps: int,
) -> np.ndarray:
    """
    Model horizon'u n_steps'ten kücükse otoregresif tahmin yapar.
    Her adimda modelin ciktisini bir sonraki adimin girisi olarak kullanir.
    Döner: (n_steps, N) ham (raw) araç sayisi.
    """
    if n_steps == 0:
        return np.zeros((0, NUM_NODES), dtype=np.float32)

    collected: list[np.ndarray] = []
    cur_window = window_norm.copy()
    cur_slot = window_start_slot

    while len(collected) < n_steps:
        if _model_tod_enabled:
            inp = _build_temporal_features(cur_window, cur_slot)
        else:
            inp = cur_window

        raw_out = _run_forward(net, inp)           # (horizon_actual, N)
        clipped = (
            np.clip(raw_out, 0, None) if _model_outputs_raw
            else np.clip(_denormalize(raw_out), 0, None)
        )
        needed = n_steps - len(collected)
        take = min(raw_out.shape[0], needed)
        for k in range(take):
            collected.append(clipped[k])

        # Pencereyi kaydır: tahmin edilen adımları normalize ederek ekle
        for k in range(take):
            norm_step = _normalize(clipped[k][np.newaxis])    # (1, N)
            if cur_window.shape[0] > 1:
                cur_window = np.vstack([cur_window[1:], norm_step])
            else:
                cur_window = norm_step
            cur_slot = (cur_slot + 1) % SLOTS_PER_DAY

    return np.vstack(collected)   # (n_steps, N)


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
        if _model_tod_enabled:
            window_start_slot = max(0, current_minute_index - lag_needed + 1) % SLOTS_PER_DAY
            model_input = _build_temporal_features(history_norm, window_start_slot)
        else:
            model_input = history_norm
        async with _INFERENCE_LOCK:
            loop = asyncio.get_event_loop()
            model_out = await loop.run_in_executor(
                None, _run_forward, _model, model_input
            )
        pred = np.clip(model_out, 0, None) if _model_outputs_raw else np.clip(_denormalize(model_out), 0, None)
    else:
        pred_norm = _moving_average_predict(history_norm)
        pred = np.clip(_denormalize(pred_norm), 0, None)

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
    Tüm gün (144 slot) tahmin serisi.

    Geçmiş slotlar gerçek veriyle, gelecek slotlar _autoregress() ile üretilir
    (model horizon=1 ise her adımda pencere kaydırılır, horizon>1 ise tek
    forward pass birden fazla adım döner — karar _autoregress() içinde otomatik).

    Tohum: günün başındaki ilk `lag` gerçek slot.
    completed_idx: gün içinde artar (0-143).
    """
    lag = _model_lag if _model is not None else LAG

    # Tüm mevcut gercek vektorler
    n_actual = completed_idx + 1
    actual = np.zeros((n_actual, NUM_NODES), dtype=np.float32)
    for s in range(n_actual):
        actual[s] = belediye_to_node_vector(data_by_junction, s)

    # Encoder penceresi: son lag gercek slot
    if n_actual >= lag:
        window = actual[-lag:]
        window_start_slot = (completed_idx - lag + 1) % SLOTS_PER_DAY
    else:
        pad = np.zeros((lag - n_actual, NUM_NODES), dtype=np.float32)
        window = np.vstack([pad, actual])
        window_start_slot = 0

    window_norm = _normalize(window)

    if _model is None:
        pred_norm = _moving_average_predict(window_norm)
        base = np.clip(_denormalize(pred_norm), 0, None)[0]
        result_series = np.tile(base, (SLOTS_PER_DAY, 1))  # (144, N)

    else:
        # Geçmiş slotlar gerçek veriyle, gelecek slotlar autoregress ile üretilir
        result_series = np.zeros((SLOTS_PER_DAY, NUM_NODES), dtype=np.float32)

        # Geçmiş: her slot için o ana kadarki gerçek veri penceresini kullan
        def _clip_pred(raw_out: np.ndarray) -> np.ndarray:
            return (np.clip(raw_out, 0, None) if _model_outputs_raw
                    else np.clip(_denormalize(raw_out), 0, None))

        for s in range(n_actual):
            if s < lag:
                pad = np.zeros((lag - s, NUM_NODES), dtype=np.float32)
                win = np.vstack([pad, actual[:s]]) if s > 0 else np.zeros((lag, NUM_NODES), dtype=np.float32)
            else:
                win = actual[s - lag:s]
            raw_out = _run_forward(_model, _normalize(win))
            result_series[s] = _clip_pred(raw_out)[0]

        # Gelecek: son gerçek pencereden autoregress
        n_future = SLOTS_PER_DAY - n_actual
        if n_future > 0:
            future_preds = _autoregress(_model, window_norm, window_start_slot, n_future)
            result_series[n_actual:n_actual + n_future] = future_preds

    out: Dict[int, Dict[str, List[float]]] = {}
    for jid, arm_map in _node_map.items():
        out[jid] = {
            arm: [float(result_series[s, node_idx]) for s in range(result_series.shape[0])]
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


def _predict_full_day_sync(
    seed_data_by_junction: Dict[int, List[dict]],
    seed_completed_idx: int = 143,
) -> Dict[int, Dict[str, List[float]]]:
    """
    Gün-öncesi tahmin — bir önceki günün verisiyle yarının 144 slotunu tahmin eder.
    144 ayrı forward pass, her seferinde pencere kaydırılır (_autoregress()).

    seed_data_by_junction: Bir onceki gunun gercek verisi (belediye API formati)
    seed_completed_idx:    Tohum verideki son gecerli slot (genellikle 143 = 23:50)
    """
    lag = _model_lag if _model is not None else LAG

    # Tohum: onceki gunun son 'lag' slotunu pencere olarak al
    seed_len = seed_completed_idx + 1
    seed_actual = np.zeros((seed_len, NUM_NODES), dtype=np.float32)
    for s in range(seed_len):
        seed_actual[s] = belediye_to_node_vector(seed_data_by_junction, s)

    # Son lag slotu: onceki gunun sonu (00:00'dan hemen oncesi)
    if seed_len >= lag:
        window = seed_actual[-lag:]                         # (lag, N)
    else:
        pad = np.zeros((lag - seed_len, NUM_NODES), dtype=np.float32)
        window = np.vstack([pad, seed_actual])              # (lag, N)

    window_norm = _normalize(window)

    # Temporal feature: pencere onceki gunun son lag slotuna ait
    window_start_slot = (seed_completed_idx - lag + 1) % SLOTS_PER_DAY
    if _model_tod_enabled:
        model_input = _build_temporal_features(window_norm, window_start_slot)
    else:
        model_input = window_norm

    if _model is not None:
        predicted = _autoregress(_model, window_norm, window_start_slot, SLOTS_PER_DAY)
    else:
        # Fallback: hareketli ortalama tabanlı gun pattern'i
        base = np.clip(_denormalize(_moving_average_predict(window_norm)), 0, None)[0]
        peak = np.array([
            1.8 if 42 <= s <= 54 else   # 07:00-09:00
            1.5 if 102 <= s <= 114 else  # 17:00-19:00
            1.2 if 72 <= s <= 78 else    # 12:00-13:00
            0.3 if s < 30 or s >= 138 else 0.8
            for s in range(SLOTS_PER_DAY)
        ], dtype=np.float32)
        predicted = np.outer(peak, base)                    # (144, N)

    out: Dict[int, Dict[str, List[float]]] = {}
    for jid, arm_map in _node_map.items():
        out[jid] = {
            arm: [float(predicted[s, node_idx]) for s in range(predicted.shape[0])]
            for arm, node_idx in arm_map.items()
        }
    return out


async def predict_full_day(
    seed_data_by_junction: Dict[int, List[dict]],
    seed_completed_idx: int = 143,
) -> Dict[int, Dict[str, List[float]]]:
    """
    Async wrapper — bir sonraki gun icin tam-gun (144 slot) otoregresif tahmin.
    seed_data_by_junction: onceki gunun gercek verisi (belediye API formati).
    """
    await ensure_model_loaded()
    async with _INFERENCE_LOCK:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            _predict_full_day_sync,
            seed_data_by_junction,
            seed_completed_idx,
        )


def get_model_status() -> dict:
    """Wrapper durumunu döndürür (sağlık endpoint'i için)."""
    return {
        "model_loaded": _model is not None,
        "model_path": str(_requested_model_path) if _requested_model_path else None,
        "num_nodes": NUM_NODES,
        "lag": _model_lag,
        "horizon": HORIZON,
        "input_dim": _model_input_dim,
        "tod_enabled": _model_tod_enabled,
        "scaler_mean": SCALER_MEAN,
        "scaler_std": SCALER_STD,
        "fallback_active": _model is None,
        "history_length": len(_history),
    }


def clear_history() -> None:
    """Geçmiş tamponu temizler (test/yeniden başlatma için)."""
    global _history
    _history = []
