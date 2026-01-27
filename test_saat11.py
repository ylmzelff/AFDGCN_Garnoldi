import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"c:\Users\lenovo\Desktop\Yeni_klasör\projects\AFDGCN_Garnoldi")

# Test with SAAT 11 (like in the UI screenshots)
kavsak = "Gesi"
kol = "A - SİVAS BULVARI-SİVAS YÖNÜ"
tarih_ocak = "17.01.2026"
tarih_subat = "23.02.2025"
saat = 11

print("=" * 60)
print("OCAK TEST (Saat 11)")
print("=" * 60)

ILDEM_REAL = str(BASE_DIR / "ocak_gerçek.xlsx")
h_prefix = f"{int(saat):02d}:"

try:
    df_real = pd.read_excel(ILDEM_REAL, sheet_name=kavsak, header=2).copy()
    df_real.columns = [str(c).strip() for c in df_real.columns]
    df_real.iloc[:, 0] = pd.to_datetime(df_real.iloc[:, 0], errors='coerce').dt.strftime('%d.%m.%Y')
    
    r_mask = (df_real.iloc[:, 0] == tarih_ocak) & (df_real.iloc[:, 1].astype(str).str.startswith(h_prefix))
    print(f"Eşleşen satır sayısı: {r_mask.sum()}")
    
    if r_mask.sum() > 0:
        v_real = int(pd.to_numeric(df_real[r_mask][kol], errors='coerce').fillna(0).sum())
        print(f"✅ Gerçek değer: {v_real}")
    else:
        print("❌ Hiç eşleşen satır yok")
except Exception as e:
    print(f"❌ Hata: {e}")

print("\n" + "=" * 60)
print("ŞUBAT TEST (Saat 11)")
print("=" * 60)

ILDEM_REAL = str(BASE_DIR / "şubat_gerçek.xlsx")

try:
    df_real = pd.read_excel(ILDEM_REAL, sheet_name=kavsak, header=2).copy()
    df_real.columns = [str(c).strip() for c in df_real.columns]
    df_real.iloc[:, 0] = pd.to_datetime(df_real.iloc[:, 0], errors='coerce').dt.strftime('%d.%m.%Y')
    
    r_mask = (df_real.iloc[:, 0] == tarih_subat) & (df_real.iloc[:, 1].astype(str).str.startswith(h_prefix))
    print(f"Eşleşen satır sayısı: {r_mask.sum()}")
    
    if r_mask.sum() > 0:
        v_real = int(pd.to_numeric(df_real[r_mask][kol], errors='coerce').fillna(0).sum())
        print(f"✅ Gerçek değer: {v_real}")
    else:
        print("❌ Hiç eşleşen satır yok")
except Exception as e:
    print(f"❌ Hata: {e}")
