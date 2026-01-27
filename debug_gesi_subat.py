import pandas as pd

# Test get_ildem_counts logic for Gesi ŞUBAT
kavsak = "Gesi"
kol = "A - SİVAS BULVARI-SİVAS YÖNÜ"
tarih = "23.02.2025"
saat = 14
data_type = "karlı gün verisi(şubat)"

ILDEM_REAL = "şubat_gerçek.xlsx"
h_prefix = f"{int(saat):02d}:"

try:
    df_real = pd.read_excel(ILDEM_REAL, sheet_name=kavsak, header=2).copy()
    df_real.columns = [str(c).strip() for c in df_real.columns]
    df_real.iloc[:, 0] = pd.to_datetime(df_real.iloc[:, 0], errors='coerce').dt.strftime('%d.%m.%Y')
    
    print(f"Tarih sütunu son 5 değer: {df_real.iloc[:, 0].tail().tolist()}")
    print(f"\nAranan tarih: {tarih}")
    print(f"Aranan saat prefix: {h_prefix}")
    print(f"Aranan kol: {kol}")
    
    if kol in df_real.columns:
        print(f"✅ Kol '{kol}' bulundu")
    else:
        print(f"❌ Kol '{kol}' bulunamadı")
    
    r_mask = (df_real.iloc[:, 0] == tarih) & (df_real.iloc[:, 1].astype(str).str.startswith(h_prefix))
    print(f"\nMask ile eşleşen satır sayısı: {r_mask.sum()}")
    
    if r_mask.sum() > 0 and kol in df_real.columns:
        v_real = int(pd.to_numeric(df_real[r_mask][kol], errors='coerce').fillna(0).sum())
        print(f"✅ Gerçek değer: {v_real}")
    else:
        print("❌ Veri bulunamadı")
        
except Exception as e:
    import traceback
    print(f"❌ Hata: {traceback.format_exc()}")
