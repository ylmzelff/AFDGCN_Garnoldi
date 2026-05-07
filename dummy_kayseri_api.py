"""
Dummy Kayseri API — Test / Geliştirme Ortamı
=============================================
Gerçek belediye sunucusuna erişim olmadan sistemi test etmek için.

Gerçek API formatını taklit eder:
  POST /auth/login        → JWT token
  GET  /{city}/{region}   → junction + time_slot verisi

Başlatmak:
    .venv\\Scripts\\python.exe -m uvicorn dummy_kayseri_api:app --port 9000 --reload
"""

import random
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Dummy Kayseri API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Sabit bölge–kavşak tanımları ──────────────────────────────────────────────

REGION_JUNCTIONS: dict = {
    "ildem": {
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
    "tuna": {
        5:  ["A", "B", "C", "D"],
        3:  ["A", "B", "C", "D"],
        87: ["A", "B", "C", "D"],
        25: ["A", "B", "C", "D"],
        26: ["A", "B", "C", "D"],
        27: ["A", "B", "C", "D"],
        7:  ["A", "B", "C", "D"],
    },
    "kizilirmak": {
        130: ["A", "B", "C", "D"],
        38:  ["A", "B", "C", "D"],
        176: ["A", "B", "C", "D"],
    },
}

# ── Auth ──────────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/auth/login")
def login(req: LoginRequest):
    # Her kullanıcı/şifreyi kabul et — sadece dummy token dön
    return {
        "access_token": "dummy_token_for_testing",
        "token_type": "bearer",
        "expires_in": 86400,
    }

# ── Veri Uç Noktaları ─────────────────────────────────────────────────────────

def _make_junction_data(junction_id: int, arms: list[str], n_slots: int = 6) -> dict:
    """
    n_slots adet time_slot üretir (her slot = 10 dakika).
    Değerler, aynı 10 dakikalık pencerede tutarlı olması için
    junction_id + slot_index + zaman dilimi ile seed'lenir.
    """
    hour = time.localtime().tm_hour
    # Güncel 10-dakika penceresi (0-5 arası her saatte 6 dilim)
    current_slot = time.localtime().tm_min // 10

    peak = 1.0
    if 7 <= hour <= 9 or 17 <= hour <= 19:
        peak = 2.2
    elif 12 <= hour <= 14:
        peak = 1.4

    time_slots = []
    for slot_idx in range(n_slots):
        edges = []
        # Geçmiş slotlar: current_slot'tan geriye doğru
        abs_slot = (hour * 6 + current_slot - (n_slots - 1 - slot_idx))
        for arm in arms:
            # Her (kavşak, kol, zaman_dilimi) için deterministik seed
            rng = random.Random(junction_id * 1000 + ord(arm) + abs_slot * 17)
            base = rng.uniform(10, 85) * peak
            edges.append({
                "edge_direction": arm,
                "edge_name": f"Kol {arm}",
                "traffic_count": round(base),
            })
        time_slots.append({"slot_index": slot_idx, "edges": edges})

    return {"junction_id": junction_id, "time_slots": time_slots}


@app.get("/{city}/{region}")
def get_region_data(city: str, region: str):
    region_lower = region.lower()
    if region_lower not in REGION_JUNCTIONS:
        raise HTTPException(status_code=404, detail=f"Bölge bulunamadı: {region}")

    junctions = []
    for jid, arms in REGION_JUNCTIONS[region_lower].items():
        junctions.append(_make_junction_data(jid, arms))

    return {
        "city": city,
        "region": region_lower,
        "junction_count": len(junctions),
        "junctions": junctions,
    }


@app.get("/health")
def health():
    return {"status": "ok", "mode": "dummy"}


@app.get("/")
def root():
    return {
        "service": "Dummy Kayseri API",
        "note": "Bu sunucu gerçek belediye API'si yerine test verisi üretir.",
        "endpoints": [
            "POST /auth/login",
            "GET  /{city}/{region}  (örn: /kayseri/ildem)",
            "GET  /health",
        ],
    }
