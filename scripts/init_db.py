"""
Veritabanını başlatır: tabloları oluşturur.

Kullanım (proje kökünden çalıştır):
    python scripts/init_db.py

Ortam değişkeni:
    DATABASE_URL=postgresql+asyncpg://postgres:postgres123@localhost:5432/afdgcn
    (varsayılan .env'den okunur)
"""

from __future__ import annotations

import asyncio
import os
import sys

# Proje kökünü sys.path'e ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# .env yükle (python-dotenv varsa)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from backend.app.db.session import create_all_tables, engine
from backend.app.core.config import settings
from backend.app.db import models  # noqa: F401 — tablolar Base.metadata'ya kayıt olsun

DATABASE_URL = settings.database_url


async def init() -> None:
    print(f"[→] Veritabanına bağlanılıyor: {DATABASE_URL}")

    # Tabloları oluştur (idempotent — zaten varsa atlanır)
    await create_all_tables()
    print("[OK] Tablolar oluşturuldu / mevcutsa atlandı")
    print()
    print("Tablolar:")
    print("  ✓ phase_predictions  — faz öneri logları")
    print("  ✓ model_events       — model yükleme / fallback olayları")
    print()
    print("[OK] Veritabanı hazır. Phase API'yi başlatabilirsiniz:")
    print("     python phase_api.py")


if __name__ == "__main__":
    asyncio.run(init())
