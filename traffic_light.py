# traffic_light.py
# Kullanım:
#   python traffic_light.py <intersection> <arm_letter> <hour> <excel_path>
#
# Örnek:
#   python traffic_light.py Beyazşehir A 14 ocak_tahmin.xlsx

import sys
import pandas as pd
import numpy as np

# -----------------------------
# 1) PARAMETRELER
# -----------------------------
LANE_CONFIG = {
    "Gesi":       {"A": 4, "B": 3, "C": 3, "D": 1},
    "Serkent":    {"A": 2, "B": 2, "C": 2, "D": 2},
    "Beyazşehir": {"A": 3, "B": 3, "C": 3, "D": 2},
    "Toki":       {"A": 2, "B": 2, "C": 2, "D": 2},
    "İldem 1":    {"A": 2, "B": 2, "C": 2, "D": 3},
    "İldem 2":    {"A": 3, "B": 3, "C": 2, "D": 1},
    "İldem 3":    {"A": 3, "C": 3, "D": 2},
    "İldem 4":    {"A": 2, "C": 2, "D": 2},
    "İldem 5":    {"A": 3, "B": 2, "C": 3, "D": 2}
}

FIXED_PROTECTION = 6
MIN_GREEN = 10
FIXED_YELLOW = 3   # Sarı sabit

# -----------------------------
# 2) EXCEL HEADER BULMA
# -----------------------------
def find_header_row(df_raw: pd.DataFrame) -> int:
    """
    Excel'de 'Tarih' ve 'Saat' kelimelerinin olduğu satırı header olarak bulur.
    Bulamazsa fallback verir.
    """
    for i in range(min(20, len(df_raw))):
        row = df_raw.iloc[i].astype(str).str.lower()
        if row.str.contains("tarih").any() and row.str.contains("saat").any():
            return i
    return 2

# -----------------------------
# 3) O SAATTEKİ ARAÇ SAYILARINI OKU + SÜRE HESAPLA
# -----------------------------
def compute_timings_from_hourly_counts(intersection: str, hour: int, excel_path: str):
    """
    excel_path içindeki intersection sheet'inden ilgili hour satırlarını alır,
    kolların toplam araç sayılarını çıkarır, lane sayısıyla ağırlıklandırır
    ve cycle + green sürelerini dağıtır.

    Çıktı:
      results: { 'A': (green, yellow, red, protection, cycle_time, veh_count), ... }
      cycle_time
    """
    xls = pd.ExcelFile(excel_path)
    if intersection not in xls.sheet_names:
        raise ValueError(f"Sheet bulunamadı: {intersection} | Dosya: {excel_path}")

    df_raw = pd.read_excel(excel_path, sheet_name=intersection, header=None, dtype=object)
    header_idx = find_header_row(df_raw)

    df = pd.read_excel(excel_path, sheet_name=intersection, header=header_idx).copy()
    df.columns = [str(c).strip() for c in df.columns]

    if len(df.columns) < 3:
        raise ValueError("Excel formatı beklenen gibi değil (kol bazlı kolon yok).")

    # Saat filtresi (örn: 14 -> "14:")
    h_prefix = f"{int(hour):02d}:"
    time_col = df.columns[1]
    df[time_col] = df[time_col].astype(str)

    df_h = df[df[time_col].str.startswith(h_prefix)].copy()
    if df_h.empty:
        return {}, 0

    # Kol kolonlarını seç (Tarih, Saat, Toplam, Unnamed dışı)
    candidate_cols = [
        c for c in df.columns
        if c.lower() not in ["tarih", "saat", "toplam"]
        and "unnamed" not in c.lower()
    ]

    # Sadece A/B/C/D ile başlayan kolonlar
    arm_cols = []
    for c in candidate_cols:
        c0 = str(c).strip()
        if len(c0) > 0 and c0[0] in ["A", "B", "C", "D"]:
            arm_cols.append(c)

    # Numeriğe çevir
    for c in arm_cols:
        df_h[c] = pd.to_numeric(df_h[c], errors="coerce").fillna(0)

    # O saat için toplam araç sayıları
    counts = {c: float(df_h[c].sum()) for c in arm_cols}

    # Faz listesi (lane ile ağırlık)
    phases = []
    for full_name, count in counts.items():
        arm_letter = str(full_name).strip()[0]
        if arm_letter not in LANE_CONFIG.get(intersection, {}):
            continue
        lanes = LANE_CONFIG[intersection][arm_letter]
        load = count / max(lanes, 1)
        phases.append({
            "name": full_name,
            "arm": arm_letter,
            "count": count,
            "lanes": lanes,
            "load": load
        })

    if not phases:
        return {}, 0

    total_weighted_load = sum(p["load"] for p in phases)
    num_phases = len(phases)

    # Cycle time: 60-120 arası ölçekle (senin mantıkla aynı)
    load_factor = np.clip((total_weighted_load - 100) / (2500 - 100), 0, 1)
    cycle_time = int(60 + load_factor * 60)

    # Loss: her faz için (yellow + protection)
    total_loss = num_phases * (FIXED_YELLOW + FIXED_PROTECTION)
    net_green_pool = max(cycle_time - total_loss, num_phases * MIN_GREEN)

    # Green dağıtımı
    for p in phases:
        share = (p["load"] / total_weighted_load) if total_weighted_load > 0 else (1.0 / num_phases)
        p["green"] = int(max(MIN_GREEN, round(share * net_green_pool)))

    # Yuvarlama düzeltmesi (toplam green net_green_pool olmalı)
    green_sum = sum(p["green"] for p in phases)
    diff = net_green_pool - green_sum
    if diff != 0:
        max(phases, key=lambda x: x["load"])["green"] += diff

    # Sonuçları kol harfine göre yaz
    results = {}
    for p in phases:
        green = max(0, int(p["green"]))
        yellow = FIXED_YELLOW
        red = max(0, int(cycle_time - green - yellow))  # görsel amaçlı
        protection = FIXED_PROTECTION
        results[p["arm"]] = (green, yellow, red, protection, cycle_time, int(p["count"]))

    return results, cycle_time

# -----------------------------
# 4) MAIN
# -----------------------------
def main():
    if len(sys.argv) < 5:
        print("KULLANIM: python traffic_light.py <intersection> <arm_letter> <hour> <excel_path>")
        sys.exit(1)

    intersection = sys.argv[1]
    arm_letter = sys.argv[2].strip()[0]  # A/B/C/D
    hour = int(sys.argv[3])
    excel_path = sys.argv[4]

    results, cycle = compute_timings_from_hourly_counts(intersection, hour, excel_path)

    if arm_letter not in results:
        print(f"⚠️ Kol bulunamadı: {arm_letter} | intersection={intersection} hour={hour}")
        print("📤 SONUÇ: 0,0,0,0")
        return

    green, yellow, red, protection, cycle_time, veh_count = results[arm_letter]

    # Arayüz bu satırı parse ediyor
    print(f"📤 SONUÇ: {green},{yellow},{red},{protection}")

if __name__ == "__main__":
    main()
