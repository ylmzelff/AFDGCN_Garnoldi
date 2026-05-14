"""
Kayseri Client Manuel Test Script
====================================

kayseri_api.py (port 9000) ayaktayken çalıştır:
    python tests/test_belediye_api.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.kayseri_client import KayseriAPIClient


async def main() -> None:
    print("=" * 60)
    print("🔧 KAYSERİ API İSTEMCİ TESTİ")
    print("=" * 60)

    client = KayseriAPIClient()

    # 1. Sağlık kontrolü
    ok = await client.health_check()
    print(f"\n[{'OK' if ok else 'HATA'}] Kayseri API sağlık: {'çevrimiçi' if ok else 'çevrimdışı'}")

    if not ok:
        print("\n⚠️  kayseri_api.py port 9000'de çalışmıyor.")
        print("    Önce: python kayseri_api.py")
        await client.close()
        return

    # 2. Login
    print("\n[→] Login deneniyor...")
    await client.ensure_authenticated()
    st = client.get_status()
    print(f"[{'OK' if st['authenticated'] else 'HATA'}] Token alındı: {st['authenticated']}")
    print(f"    Token süresi: {st['token_expires_in']}s")

    # 3. İldem bölgesi verisi
    print("\n[→] İldem bölgesi çekiliyor...")
    try:
        data = await client.fetch_region("ildem")
        print(f"[OK] {len(data)} kavşak alındı")
        for jid, arms in list(data.items())[:2]:
            print(f"    Kavşak {jid}: {len(arms)} kol")
            for arm in arms[:2]:
                direction = arm.get("edge_direction", "?")
                slot0 = arm.get("0", 0)
                print(f"        Kol {direction}: slot[0]={slot0}")
    except Exception as exc:
        print(f"[HATA] {exc}")

    await client.close()
    print("\n[OK] Test tamamlandı.")


if __name__ == "__main__":
    asyncio.run(main())

