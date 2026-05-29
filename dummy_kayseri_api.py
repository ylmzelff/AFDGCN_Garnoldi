"""
Dummy Kayseri AusTKM API — HuggingFace Spaces Deploy
=====================================================
Gerçek belediye sunucusuna erişim olmadan sistemi test etmek için.
Gerçek AusTKM API formatını birebir taklit eder.

Endpoint: GET /api/SensorVerileri?bolgeAdi=İLDEM&tarih=YYYY-MM-DD
Auth:     Bearer <herhangi_bir_token>
"""

import random
import time
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Dummy Kayseri AusTKM API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Bölge–Kavşak tanımları ────────────────────────────────────────────────────

BOLGE_MAP = {
    "İLDEM": {
        89:  ["A", "B", "C", "D"],
        187: ["A", "B", "C", "D"],
        95:  ["A", "B", "C", "D"],
        121: ["A", "B", "C", "D"],
        184: ["A", "B", "D"],
        188: ["A", "B", "C", "D"],
        117: ["A", "C", "D"],
        192: ["A", "B", "C", "D"],
        194: ["A", "B", "C", "D"],
    },
    "TUNA": {
        5:  ["A", "B", "C", "D"],
        3:  ["A", "B", "C", "D"],
        87: ["A", "B", "C", "D"],
        25: ["A", "B", "C", "D"],
        26: ["A", "B", "C", "D"],
        27: ["A", "B", "C", "D"],
        7:  ["A", "B", "C", "D"],
    },
    "KIZILIRMAK": {
        130: ["A", "B", "C", "D"],
        38:  ["A", "B", "C", "D"],
        176: ["A", "B", "C", "D"],
    },
}

# Alternatif yazımları normalize et
BOLGE_ALIASES = {
    "ILDEM": "İLDEM",
    "İLDEM": "İLDEM",
    "TUNA": "TUNA",
    "KIZILIRMAK": "KIZILIRMAK",
}


def _peak_factor(hour: int) -> float:
    if 7 <= hour <= 9 or 17 <= hour <= 19:
        return 2.2
    elif 12 <= hour <= 14:
        return 1.5
    return 1.0


def _date_seed(tarih: str) -> int:
    """YYYY-MM-DD → sayısal seed (farklı günler farklı veri üretir)."""
    try:
        y, m, d = tarih.split("-")
        return int(y) * 10000 + int(m) * 100 + int(d)
    except Exception:
        return 20260101


def _make_saatlik_veriler(junction_id: int, arm: str, date_seed: int) -> dict:
    """144 slot (00:00–23:50) için araç sayısı üret — tarihe göre deterministik."""
    result = {}
    for slot in range(144):
        hour = slot // 6
        minute = (slot % 6) * 10
        rng = random.Random(date_seed * 1_000_000 + junction_id * 10000 + ord(arm) * 100 + slot)
        base = rng.uniform(5, 60) * _peak_factor(hour)
        key = f"{hour:02d}:{minute:02d}"
        result[key] = round(base)
    return result


@app.get("/api/SensorVerileri")
def sensor_verileri(
    bolgeAdi: str = Query(..., description="Bölge adı: İLDEM, TUNA, KIZILIRMAK"),
    tarih: str = Query(..., description="Tarih: YYYY-MM-DD"),
):
    bolge_key = BOLGE_ALIASES.get(bolgeAdi.upper().strip(), bolgeAdi.upper().strip())
    junctions = BOLGE_MAP.get(bolge_key, {})

    ds = _date_seed(tarih)
    data = []
    for jid, arms in junctions.items():
        for arm in arms:
            data.append({
                "intersectionId": jid,
                "edgeDirection": arm,
                "edgeName": f"Kol {arm}",
                "saatlikVeriler": _make_saatlik_veriler(jid, arm, ds),
            })

    return {
        "success": True,
        "bolgeAdi": bolge_key,
        "tarih": tarih,
        "toplamKayit": len(data),
        "data": data,
    }


@app.get("/health")
def health():
    return {"status": "ok", "mode": "dummy", "regions": list(BOLGE_MAP.keys())}


@app.get("/")
def root():
    return {
        "service": "Dummy Kayseri AusTKM API",
        "note": "Gerçek belediye API'si yerine test verisi üretir.",
        "endpoints": ["GET /api/SensorVerileri?bolgeAdi=İLDEM&tarih=2026-01-01", "GET /health"],
    }
