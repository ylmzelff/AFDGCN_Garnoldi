import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Komut satırından argümanları al
if len(sys.argv) < 4:
    print("Usage: python traffic_light.py <intersection> <arm> <hour> [excel_file_path]")
    print("  hour: 7, 14, veya 17")
    sys.exit(1)

intersection_name = sys.argv[1]
arm_name = sys.argv[2]
hour_param = int(sys.argv[3])

# Excel dosya yolu (opsiyonel parametre veya varsayılan)
if len(sys.argv) >= 5:
    EXCEL_FILE = Path(sys.argv[4])
else:
    # Varsayılan Excel dosya yolu (Google Colab)
    BASE_DIR = Path("/content/AFDGCN_Garnoldi")
    EXCEL_FILE = BASE_DIR / "gesi_kavsak_raporlari.xlsx"

# Kavşak bazlı koruma süreleri (min–max)
PROTECTION_RULES = {
    "Gesi":       {"min": 7, "max": 13},
    "Serkent":    {"min": 7, "max": 9},
    "Beyazşehir": {"min": 7, "max": 12},
    "Toki":       {"min": 7, "max": 13},
    "İldem 1":    {"min": 7, "max": 13},
    "İldem 2":    {"min": 7, "max": 13},
    "İldem 3":    {"min": 7, "max": 12},
    "İldem 4":    {"min": 7, "max": 12},
    "İldem 5":    {"min": 7, "max": 12},
}

def calculate_traffic_light_notebook_style(intersection, arm, hour, excel_file):
    """
    Notebook'taki mantıkla trafik ışığı sürelerini hesaplar.
    Excel dosyasından direkt veri okur ve TÜM kavşak kollarını dikkate alarak orantılı dağıtım yapar.
    """
    try:
        # Excel dosyasını oku
        if not Path(excel_file).exists():
            raise FileNotFoundError(f"Excel dosyası bulunamadı: {excel_file}")
        
        # İlgili kavşağın sayfasını oku (Notebook mantığı)
        df_raw = pd.read_excel(excel_file, sheet_name=intersection, header=None)
        
        # Başlıkları ve veriyi düzenle (Notebook'taki gibi)
        df_ham = df_raw.iloc[1:].copy()  # İlk satırdaki gereksiz başlığı atla
        
        # Sütun isimlerini belirle (date, time, A, B, C, D...) - Notebook mantığı
        yeni_sutunlar = {df_ham.columns[1]: 'date', df_ham.columns[2]: 'time'}
        faz_sutunlari = []
        
        # A, B, C, D kollarını bul (Excel'deki sütun sırasına göre)
        for i in range(3, len(df_ham.columns)):
            if i >= 7:  # Maksimum 4 kol (D) alıyoruz
                break
            kol_adi = chr(65 + (i-3))  # A, B, C, D...
            yeni_sutunlar[df_ham.columns[i]] = kol_adi
            faz_sutunlari.append(kol_adi)
        
        df = df_ham.rename(columns=yeni_sutunlar)
        
        # Saat filtreleme (07:00, 14:00, 17:00) - Notebook mantığı
        df['hour'] = pd.to_datetime(df['time'].astype(str), errors='coerce').dt.hour
        df_filtered = df[df['hour'].isin([7, 14, 17])].copy()
        
        if len(df_filtered) == 0:
            df_filtered = df
        
        # Saatlik toplamları al (Notebook mantığı)
        hourly_totals = df_filtered.groupby('hour')[faz_sutunlari].sum().reset_index()
        
        # Belirtilen saatin verisini al
        hour_data = hourly_totals[hourly_totals['hour'] == hour]
        
        if len(hour_data) == 0:
            raise ValueError(f"Saat {hour} için veri bulunamadı! Mevcut saatler: {hourly_totals['hour'].tolist()}")
        
        row = hour_data.iloc[0]
        
        # Tüm kollar için faz listesi oluştur
        phases = []
        for kol_adi in faz_sutunlari:
            count = row[kol_adi] if kol_adi in row.index and not pd.isna(row[kol_adi]) else 0
            
            if count > 0:
                phases.append({
                    'name': kol_adi,
                    'count': count
                })
        
        if not phases:
            raise ValueError("Hiç aktif kol bulunamadı (tüm kollar 0 araç)")
        
        # Toplam araç sayısı
        total_v = sum(p['count'] for p in phases)
        num_phases = len(phases)
        
        # Koruma parametrelerini al
        rules = PROTECTION_RULES.get(intersection, {"min": 7, "max": 13})
        prot_min = rules["min"]
        prot_max = rules["max"]
        
        # 🚦 1. Dinamik Döngü ve Koruma
        ratio = np.clip((total_v - 500) / (8000 - 500), 0, 1)
        base_prot = int(prot_max - (ratio * (prot_max - prot_min)))
        
        # 🛡️ Döngü Güvencesi
        min_cycle_for_10s = num_phases * (10 + base_prot)
        cycle_time = max(min_cycle_for_10s, int(60 + (ratio * 60)))
        
        net_green_pool = cycle_time - (num_phases * base_prot)
        
        # 🌱 2. Yeşil Dağıtımı
        for p in phases:
            share = p['count'] / total_v
            g_calc = round(share * net_green_pool)
            
            p['f_green'] = max(10, g_calc)
            p['f_prot'] = base_prot
            
            # Yeşil çok uzarsa korumadan çalma
            if p['f_green'] > 30:
                can_steal = p['f_prot'] - prot_min
                stolen = min(p['f_green'] - 30, can_steal)
                p['f_green'] = 30 + stolen
                p['f_prot'] -= stolen
        
        # ⏱ 3. Dengeleme
        diff = cycle_time - sum(p['f_green'] + p['f_prot'] for p in phases)
        heaviest = max(phases, key=lambda x: x['count'])
        heaviest['f_green'] += diff
        
        # İstenen kol için sonuçları al
        target_phase = next((p for p in phases if p['name'] == arm), None)
        if not target_phase:
            raise ValueError(f"Kol bulunamadı: {arm}")
        
        green = target_phase['f_green']
        yellow = 2
        red = target_phase['f_prot'] - yellow
        protection = target_phase['f_prot']
        
        return green, yellow, red, protection
        
    except Exception as e:
        import traceback
        print(f"\n❌ HATA: {e}")
        print(traceback.format_exc())
        return 30, 3, 60, 13

# Ana fonksiyon
green, yellow, red, protection = calculate_traffic_light_notebook_style(intersection_name, arm_name, hour_param, EXCEL_FILE)

# Sonucu virgülle ayrılmış formatta yazdır
print(f"📤 SONUÇ: {green},{yellow},{red},{protection}")

