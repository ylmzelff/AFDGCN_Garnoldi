import gradio as gr
import pandas as pd
import numpy as np
import os, shutil, subprocess, json
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO
import tempfile
import plotly.graph_objects as go
import folium
import warnings
import time

warnings.filterwarnings('ignore', category=FutureWarning)

# === Paths ===
BASE_DIR = Path().absolute()
DATA_DIR = BASE_DIR / "data" / "Kayseri"
CONFIG_PY = BASE_DIR / "config.py"
LOAD_DATASET = BASE_DIR / "lib" / "load_dataset.py"
GRAPH_CSV_DEST = DATA_DIR / "directed_graph_edges.csv"
NPZ_DEST = DATA_DIR / "kol_bazli_2.npz"

REAL_FLOW_CSV = BASE_DIR / "real_flow.csv"
TEST_RESULTS_CSV = BASE_DIR / "test_results.csv"
REAL_FLOW_GAR = BASE_DIR / "real_flow_gar.csv"
TEST_GAR = BASE_DIR / "test_gar.csv"
PLOT_PY = BASE_DIR / "plot.py"
PRED_CSV = BASE_DIR / "pred_csv.py"
PLOT_IMG = BASE_DIR / "traffic_flow_zoomed.png"
TRAFFIC_LIGHT_PY = str(BASE_DIR / "traffic_light.py")
EXCEL_FILE = BASE_DIR / "gesi_kavşak_raporları.xlsx"

KAYSERI_ULASIM_LOGO_PATH = BASE_DIR / "kayseri_ulaşım.png"
SMARTTECH_LOGO_PATH = BASE_DIR / "smarttecl_logo.png"

# ==========================================
# 1) ANALYSIS.PY MANTIĞI VE VERİLERİ
# ==========================================
ANALYSIS_INTERSECTIONS = {
    "Emrah": ["A - EMRAH CAD. KUZEY", "B - KIZILIRMAK CAD DOĞU", "C - EMRAH CAD GÜNEY", "D - KIZILIRMAK CAD BATI"],
    "Farabi": ["A - FARABİ CAD KUZEY", "B - KIZILIRMAK CAD DOĞU", "C - FARABİ CAD GÜNEY", "D - KIZILIRMAK CAD BATI"],
    "Yahya Kemal": ["A - YAHYA KEMAL CAD. KUZEY", "B - KIZILIRMAK CAD DOĞU", "C - YAHYA KEMAL CAD. GÜNEY", "D - KIZILIRMAK CAD BATI"],
    "Gesi": ["A - SİVAS BULVARI-SİVAS YÖNÜ", "B - GESİ CAD.", "C - SİVAS BULV - ŞEHİR MERKEZİ", "D - 381. SOKAK"],
    "İldem 1": ["A - DİNÇER SOKAK", "B - ALPARSLANTÜRKEŞ BUL.", "D - GESİ CAD"],
    "İldem 2": ["A - DİNÇER SOKAK KUZEY", "B - HANEDAN SOKAK", "C - DİNÇER SOKAK GÜNEY", "D - FETİH SOKAK"],
    "İldem 3": ["A - DÜNDAR TAŞER CAD. KUZEY", "C - DÜNDAR TAŞER CAD. GÜNEY", "D - HANEDAN SOKAK"],
    "İldem 4": ["A - YAVUZSULTAN SELİM CAD. KUZEY", "B - VATAN SOKAK", "C - YAVUZSULTAN SELİM CAD. GÜNEY", "D - ORKUN SOKAK"],
    "İldem 5": ["A - S.A BEDUK CAD. KUZEY", "B - VATAN SOKAK BATI", "C - S.A BEDUK CAD. GÜNEY", "D - VATAN SOKAK DOĞU"],
    "Serkent": ["A - 822. SK", "B - GESİ CAD. DOĞU", "C - KOCASİNAN CAD.", "D - GESİ CAD. BATI"],
    "Beyazşehir": ["A - OSMAN ÖZCAN CAD.", "B - GESİ CAD DOĞU", "C - MARKETLER ÇIKIŞ", "D - GESİ CAD BATI"],
    "Toki": ["A - 832.CD", "B - GESİ CAD. DOĞU", "C - KADİR HAS BUL.", "D - GESİ CAD BATI"]
}

DATES_KIZILIRMAK = ["28.10.2025", "29.10.2025", "30.10.2025", "31.10.2025"]
DATES_ILDEM = {
    "karlı gün verisi(şubat)": ["23.02.2025", "24.02.2025", "25.02.2025", "26.02.2025", "27.02.2025", "28.02.2025"],
    "karlı gün verisi(ocak)": ["17.01.2026", "18.01.2026", "19.01.2026", "20.01.2026", "21.01.2026", "22.01.2026", "23.01.2026"],
    "normal veri(kasım)": ["26.11.2025", "27.11.2025", "28.11.2025"]
}
KIZILIRMAK_LIST = ["Emrah", "Farabi", "Yahya Kemal"]

ANALYSIS_COORDS = {
    "Emrah": [38.725, 35.485], "Farabi": [38.728, 35.490], "Yahya Kemal": [38.730, 35.495],
    "Gesi": [38.777174902567886, 35.57388774400878],
    "İldem 1": [38.756, 35.590], "İldem 2": [38.758, 35.595], "İldem 3": [38.760, 35.600],
    "İldem 4": [38.762, 35.605], "İldem 5": [38.764, 35.610], "Serkent": [38.768, 35.570],
    "Beyazşehir": [38.760, 35.560], "Toki": [38.775, 35.580]
}

INTERSECTIONS = {
    "Gesi": {"A - SİVAS BULVARI-SİVAS YÖNÜ": 0, "B - GESİ CAD.": 1, "C - SİVAS BULV - ŞEHİR MERKEZİ": 2, "D - 381. SOKAK": 3},
    "Serkent": {"A - 822. SK": 4, "B - GESİ CAD. DOĞU": 5, "C - KOCASİNAN CAD.": 6, "D - GESİ CAD. BATI": 7},
    "Beyazşehir": {"A - OSMAN ÖZCAN CAD.": 8, "B - GESİ CAD DOĞU": 9, "C - MARKETLER ÇIKIŞ": 10, "D - GESİ CAD BATI": 11},
    "Toki": {"A - 832.CD": 12, "B - GESİ CAD. DOĞU": 13, "C - KADİR HAS BUL.": 14, "D - GESİ CAD BATI": 15},
    "İldem 1": {"A - DİNÇER SOKAK": 16, "B - ALPARSLANTÜRKEŞ BUL.": 17, "D - GESİ CAD": 18},
    "İldem 2": {"A - DİNÇER SOKAK KUZEY": 19, "B - HANEDAN SOKAK": 20, "C - DİNÇER SOKAK GÜNEY": 21, "D - FETİH SOKAK": 22},
    "İldem 3": {"A - DÜNDAR TAŞER CAD. KUZEY": 23, "C - DÜNDAR TAŞER CAD. GÜNEY": 24, "D - HANEDAN SOKAK": 25},
    "İldem 4": {"A - YAVUZSULTAN SELİM CAD. KUZEY": 26, "B - VATAN SOKAK": 27, "C - YAVUZSULTAN SELİM CAD. GÜNEY": 28, "D - ORKUN SOKAK": 29},
    "İldem 5": {"A - S.A BEDUK CAD. KUZEY": 30, "B - VATAN SOKAK BATI": 31, "C - S.A BEDUK CAD. GÜNEY": 32, "D - VATAN SOKAK DOĞU": 33}
}

def get_base64_image_tag(image_path: Path, max_height: int = 90) -> str:
    if not image_path.exists():
        return ""
    with Image.open(image_path) as img:
        if img.height > max_height:
            new_width = int(img.width * max_height / img.height)
            img = img.resize((new_width, max_height), Image.LANCZOS)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_b64}"

# ==========================================
# 2) VERİ HESAPLAMA - DÜZELTİLMİŞ
# ==========================================
def get_kizilirmak_counts(kavsak, kol_full, tarih, saat):
    """Kızılırmak veri okuma"""
    KIZILIRMAK_REAL = str(BASE_DIR / "kizilirmak_gerçek.xlsx")
    KIZILIRMAK_PRED = str(BASE_DIR / "kizilirmak_tahmin.xlsx")
    try:
        harf = kol_full.split(" - ")[0]
        col_idx = {"A": 3, "B": 4, "C": 5, "D": 6}.get(harf, 3)
        d, m, y = tarih.split(".")
        h_prefix = f"{int(saat):02d}:"

        xl_real = pd.ExcelFile(KIZILIRMAK_REAL)
        target_sheet = next((s for s in xl_real.sheet_names if kavsak.strip().lower() in s.strip().lower()), None)
        df_real = pd.read_excel(KIZILIRMAK_REAL, sheet_name=target_sheet, header=None, dtype=object)
        mask_r = df_real.apply(
            lambda r: (str(y) in str(r.iloc[1]) and str(m) in str(r.iloc[1]) and str(d) in str(r.iloc[1]))
                      and (h_prefix in str(r.iloc[2])),
            axis=1
        )
        v_real = int(pd.to_numeric(df_real[mask_r].iloc[:, col_idx], errors="coerce").fillna(0).sum())

        xl_pred = pd.ExcelFile(KIZILIRMAK_PRED)
        df_pred = pd.read_excel(KIZILIRMAK_PRED, sheet_name=target_sheet, header=None, dtype=object)
        mask_p = df_pred.apply(
            lambda r: (str(y) in str(r.iloc[1]) and str(m) in str(r.iloc[1]) and str(d) in str(r.iloc[1]))
                      and (h_prefix in str(r.iloc[2])),
            axis=1
        )
        v_pred = int(pd.to_numeric(df_pred[mask_p].iloc[:, col_idx], errors="coerce").fillna(0).sum())
        return v_real, v_pred
    except:
        return 0, 0

def find_header_row(file_path, sheet_name):
    """Excel'de 'Tarih' içeren header satırını dinamik bul"""
    df_temp = pd.read_excel(file_path, sheet_name=sheet_name, header=None, nrows=10)
    for idx, row in df_temp.iterrows():
        if any('Tarih' in str(cell) for cell in row):
            return idx
    return 2  # Default

def get_ildem_counts(kavsak, kol, tarih, saat, data_type="karlı gün verisi(şubat)"):
    """Geliştirilmiş İldem veri okuma - Dinamik header + güçlü tarih parse"""
    file_mapping = {
        "karlı gün verisi(şubat)": (str(BASE_DIR / "şubat_gerçek.xlsx"), str(BASE_DIR / "şubat_tahmin.xlsx")),
        "karlı gün verisi(ocak)": (str(BASE_DIR / "ocak_gerçek.xlsx"), str(BASE_DIR / "ocak_tahmin.xlsx")),
        "normal veri(kasım)": (str(BASE_DIR / "kasım_gerçek.xlsx"), str(BASE_DIR / "kasım_tahmin.xlsx"))
    }

    ILDEM_REAL, ILDEM_PRED = file_mapping.get(data_type, file_mapping["karlı gün verisi(şubat)"])
    h_prefix = f"{int(saat):02d}:"

    try:
        # GERÇEK VERİ - Dinamik header
        header_row_real = find_header_row(ILDEM_REAL, kavsak)
        df_real = pd.read_excel(ILDEM_REAL, sheet_name=kavsak, header=header_row_real).copy()
        df_real.columns = [str(c).strip() for c in df_real.columns]

        # Kolon kontrolü
        if kol not in df_real.columns:
            similar = [c for c in df_real.columns if kol.split("-")[0].strip() in c]
            if similar:
                kol = similar[0]
            else:
                return 0, 0

        # Tarih parse
        if pd.api.types.is_datetime64_any_dtype(df_real.iloc[:, 0]):
            df_real['date_only'] = df_real.iloc[:, 0].dt.normalize()
        else:
            try:
                df_real['date_parsed'] = pd.to_datetime(df_real.iloc[:, 0], format='%d.%m.%Y', errors='coerce')
            except:
                df_real['date_parsed'] = pd.NaT
            if df_real['date_parsed'].isna().all():
                df_real['date_parsed'] = pd.to_datetime(df_real.iloc[:, 0], dayfirst=True, errors='coerce')
            df_real['date_only'] = df_real['date_parsed'].dt.normalize()

        # Target tarih
        d, m, y = tarih.split(".")
        target_date = pd.Timestamp(year=int(y), month=int(m), day=int(d))

        r_mask = (df_real['date_only'] == target_date.normalize()) & \
                 (df_real.iloc[:, 1].astype(str).str.startswith(h_prefix))
        v_real = int(pd.to_numeric(df_real[r_mask][kol], errors="coerce").fillna(0).sum())

        # TAHMİN VERİ - Dinamik header
        header_row_pred = find_header_row(ILDEM_PRED, kavsak)
        df_pred = pd.read_excel(ILDEM_PRED, sheet_name=kavsak, header=header_row_pred).copy()
        df_pred.columns = [str(c).strip() for c in df_pred.columns]

        if kol not in df_pred.columns:
            similar = [c for c in df_pred.columns if kol.split("-")[0].strip() in c]
            if similar:
                kol = similar[0]

        # Tarih parse
        if pd.api.types.is_datetime64_any_dtype(df_pred.iloc[:, 0]):
            df_pred['date_only'] = df_pred.iloc[:, 0].dt.normalize()
        else:
            try:
                df_pred['date_parsed'] = pd.to_datetime(df_pred.iloc[:, 0], format='%d.%m.%Y', errors='coerce')
            except:
                df_pred['date_parsed'] = pd.NaT
            if df_pred['date_parsed'].isna().all():
                df_pred['date_parsed'] = pd.to_datetime(df_pred.iloc[:, 0], dayfirst=True, errors='coerce')
            df_pred['date_only'] = df_pred['date_parsed'].dt.normalize()

        p_mask = (df_pred['date_only'] == target_date.normalize()) & \
                 (df_pred.iloc[:, 1].astype(str).str.startswith(h_prefix))
        v_pred = int(pd.to_numeric(df_pred[p_mask][kol], errors="coerce").fillna(0).sum())

        return v_real, v_pred
    except Exception as e:
        print(f"Error in get_ildem_counts: {e}")
        return 0, 0

# ==========================================
# 3) TRAFFIC ANALYSIS UI HELPERS
# ==========================================
def create_map():
    m = folium.Map(location=[38.745, 35.54], zoom_start=12, tiles="cartodbpositron")
    for name, coords in ANALYSIS_COORDS.items():
        color = "#16a085" if name in KIZILIRMAK_LIST else "#3498db"
        html = f"""<div style="font-family:sans-serif; min-width:140px;"><b style="color:{color};">{name} Kavşağı</b><br>
        <button onclick="const trigger = window.parent.document.querySelector('#map_trigger textarea'); trigger.value = '{name}'; trigger.dispatchEvent(new Event('input', {{ bubbles: true }}));"
        style="background:{color}; color:white; border:none; padding:8px; border-radius:4px; cursor:pointer; width:100%; margin-top:8px; font-weight:bold;">SEÇ</button></div>"""
        folium.CircleMarker(
            location=coords, radius=10, color=color, fill=True,
            popup=folium.Popup(html, max_width=200)
        ).add_to(m)
    return m._repr_html_()

def update_dashboard(kavsak, tarih, saat, kol, data_type="karlı gün verisi(şubat)"):
    if not kavsak:
        kavsak = "Farabi"
    v_arms = ANALYSIS_INTERSECTIONS.get(kavsak, [])

    is_kizilirmak = kavsak in KIZILIRMAK_LIST
    current_dates = DATES_KIZILIRMAK if is_kizilirmak else DATES_ILDEM.get(data_type, DATES_ILDEM["karlı gün verisi(şubat)"])

    if tarih not in current_dates:
        tarih = current_dates[0]
    if v_arms and (kol not in v_arms):
        kol = v_arms[0]

    if is_kizilirmak:
        real_v, pred_v = get_kizilirmak_counts(kavsak, kol, tarih, saat)
    else:
        real_v, pred_v = get_ildem_counts(kavsak, kol, tarih, saat, data_type)

    diff = abs(real_v - pred_v)
    acc = (1 - (diff / real_v)) * 100 if real_v > 0 else 0

    if acc >= 85:
        confidence_color = "#16a085"
    elif acc >= 70:
        confidence_color = "#3498db"
    else:
        confidence_color = "#f39c12"

    confidence_display = f"{max(0, min(acc, 100)):.1f}%"

    arm_labels = [a.split(" - ")[0] for a in v_arms]
    rc, pc = [], []
    for a in v_arms:
        rv, pv = get_kizilirmak_counts(kavsak, a, tarih, saat) if is_kizilirmak else get_ildem_counts(kavsak, a, tarih, saat, data_type)
        rc.append(rv); pc.append(pv)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=arm_labels, y=rc, name="Gerçek", marker_color="#334155"))
    fig.add_trace(go.Bar(x=arm_labels, y=pc, name="Tahmin", marker_color="#94a3b8"))
    fig.update_layout(
        barmode="group",
        height=350,
        template="plotly_white",
        margin=dict(l=20, r=20, t=10, b=10),
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center")
    )

    def card(label, val, color="#0f172a"):
        return f"""
        <div style='background:#fff; border-radius:4px; border:1px solid #e2e8f0; display:flex; flex-direction:column; justify-content:center; align-items:center; height:100px; width:100%;'>
            <div style='font-size:11px; color:#64748b; font-weight:700; text-transform:uppercase; letter-spacing:1px;'>{label}</div>
            <div style='font-size:32px; font-weight:700; color:{color}; margin-top:5px;'>{val}</div>
        </div>"""

    def confidence_card(main_label, color):
        return f"""
        <div style='background:#fff; border-radius:4px; border:1px solid #e2e8f0; display:flex; flex-direction:column; justify-content:center; align-items:center; height:100px; width:100%;'>
            <div style='font-size:11px; color:#64748b; font-weight:700; text-transform:uppercase; letter-spacing:1px;'>DOĞRULUK</div>
            <div style='font-size:28px; font-weight:700; color:{color}; margin-top:5px;'>{main_label}</div>
        </div>"""

    return (
        card("GERÇEK ARAÇ", real_v, "#1e293b"),
        card("AI TAHMİN", pred_v, "#1e293b"),
        confidence_card(confidence_display, confidence_color),
        card("VARYANS", diff, "#1e293b"),
        fig,
        gr.update(choices=v_arms, value=kol)
    )

def handle_map_selection_v2(kavsak, current_data_type):
    choices = ANALYSIS_INTERSECTIONS.get(kavsak, [])
    if kavsak in KIZILIRMAK_LIST:
        dates = DATES_KIZILIRMAK
        return (
            gr.update(choices=choices, value=choices[0] if choices else None),
            gr.update(choices=dates, value=dates[0] if choices else None),
            gr.update(visible=False)
        )
    elif kavsak == "Gesi":
        dt = current_data_type if current_data_type in DATES_ILDEM else "normal veri(kasım)"
        dates = DATES_ILDEM.get(dt, DATES_ILDEM["normal veri(kasım)"])
        return (
            gr.update(choices=choices, value=choices[0] if choices else None),
            gr.update(choices=dates, value=dates[0] if dates else None),
            gr.update(visible=True, value=dt)
        )
    else:
        dt = current_data_type if current_data_type in DATES_ILDEM else "karlı gün verisi(şubat)"
        dates = DATES_ILDEM.get(dt, DATES_ILDEM["karlı gün verisi(şubat)"])
        return (
            gr.update(choices=choices, value=choices[0] if choices else None),
            gr.update(choices=dates, value=dates[0] if dates else None),
            gr.update(visible=True, value=dt)
        )

def update_dates_by_data_type(data_type, kavsak):
    if kavsak not in KIZILIRMAK_LIST:
        dates = DATES_ILDEM.get(data_type, DATES_ILDEM["karlı gün verisi(şubat)"])
        return gr.update(choices=dates, value=dates[0] if dates else None)
    return gr.update()

# ==========================================
# 4) TRAFFIC LIGHT HTML
# ==========================================
def generate_traffic_light_html(green_sec, yellow_sec, red_sec, protection_sec):
    try:
        green = float(green_sec) if green_sec else 0
        yellow = float(yellow_sec) if yellow_sec else 0
        red = float(red_sec) if red_sec else 0
        protection = float(protection_sec) if protection_sec else 0

        total_time = green + yellow + red
        green_pct = (green / total_time * 100) if total_time > 0 else 0
        yellow_pct = (yellow / total_time * 100) if total_time > 0 else 0
        red_pct = (red / total_time * 100) if total_time > 0 else 0

        html = f"""
        <div style="background: #ffffff; padding: 20px; border-radius: 12px; box-shadow: none;">
            <div style="display: flex; justify-content: space-around; align-items: flex-start; flex-wrap: wrap; gap: 30px;">
                <div style="text-align: center;">
                    <div style="background: linear-gradient(145deg, #e3ede6 60%, #b7cfc0 100%); padding: 15px 12px; border-radius: 16px; border: 2px solid #c7dbcf; display: inline-block;">
                        <div id="red-light" style="width: 60px; height: 60px; border-radius: 50%; margin: 8px auto; background: #6c1a1a; box-shadow: inset 0 2px 5px rgba(0,0,0,0.2); transition: all 0.4s ease;"></div>
                        <div id="yellow-light" style="width: 60px; height: 60px; border-radius: 50%; margin: 8px auto; background: #bfae3a; box-shadow: inset 0 2px 5px rgba(0,0,0,0.2); transition: all 0.4s ease;"></div>
                        <div id="green-light" style="width: 60px; height: 60px; border-radius: 50%; margin: 8px auto; background: #2e7d32; box-shadow: inset 0 2px 5px rgba(0,0,0,0.2); transition: all 0.4s ease;"></div>
                    </div>
                </div>

                <div style="flex: 1; min-width: 300px;">
                    <div style="margin-bottom: 25px;">
                        <div style="font-size: 11px; color: #64748b; font-weight: bold; margin-bottom: 8px; text-transform: uppercase;">
                            <i class="fas fa-clock"></i> Cycle Timeline ({total_time:.0f}s)
                        </div>
                        <div style="background: #f1f5f9; border-radius: 8px; overflow: hidden; height: 30px; display: flex; border: 1px solid #e2e8f0;">
                            <div style="background: #4caf50; width: {green_pct}%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 12px;">{green:.0f}s</div>
                            <div style="background: #ff9800; width: {yellow_pct}%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 12px;">{yellow:.0f}s</div>
                            <div style="background: #f44336; width: {red_pct}%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 12px;">{red:.0f}s</div>
                        </div>
                    </div>

                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 15px;">
                        <div style="background: #ffffff; border-radius: 10px; padding: 15px; border: 1px solid #e2e8f0; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                            <div style="font-size: 11px; color: #64748b; font-weight: bold; text-transform: uppercase;">GREEN</div>
                            <div style="font-size: 28px; font-weight: 800; color: #2e7d32; margin: 5px 0;">{green:.0f}s</div>
                        </div>
                        <div style="background: #ffffff; border-radius: 10px; padding: 15px; border: 1px solid #e2e8f0; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                            <div style="font-size: 11px; color: #64748b; font-weight: bold; text-transform: uppercase;">YELLOW</div>
                            <div style="font-size: 28px; font-weight: 800; color: #f57c00; margin: 5px 0;">{yellow:.0f}s</div>
                        </div>
                        <div style="background: #ffffff; border-radius: 10px; padding: 15px; border: 1px solid #e2e8f0; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                            <div style="font-size: 11px; color: #64748b; font-weight: bold; text-transform: uppercase;">RED</div>
                            <div style="font-size: 28px; font-weight: 800; color: #c62828; margin: 5px 0;">{red:.0f}s</div>
                        </div>
                        <div style="background: #ffffff; border-radius: 10px; padding: 15px; border: 1px solid #e2e8f0; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                            <div style="font-size: 11px; color: #64748b; font-weight: bold; text-transform: uppercase;">PROTECTION</div>
                            <div style="font-size: 28px; font-weight: 800; color: #2c3e50; margin: 5px 0;">{protection:.0f}s</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            (function() {{
                const greenLight = document.getElementById('green-light');
                const yellowLight = document.getElementById('yellow-light');
                const redLight = document.getElementById('red-light');
                const greenTime = {green * 1000};
                const yellowTime = {yellow * 1000};
                const redTime = {red * 1000};
                let currentPhase = 0;
                function resetLights() {{
                    greenLight.style.background = '#1b5e20'; greenLight.style.boxShadow = 'inset 0 2px 5px rgba(0,0,0,0.2)';
                    yellowLight.style.background = '#827717'; yellowLight.style.boxShadow = 'inset 0 2px 5px rgba(0,0,0,0.2)';
                    redLight.style.background = '#7f0000'; redLight.style.boxShadow = 'inset 0 2px 5px rgba(0,0,0,0.2)';
                }}
                function animateLights() {{
                    resetLights();
                    if (currentPhase === 0) {{
                        greenLight.style.background = '#4caf50'; greenLight.style.boxShadow = '0 0 20px #4caf50';
                        setTimeout(() => {{ currentPhase = 1; animateLights(); }}, greenTime);
                    }} else if (currentPhase === 1) {{
                        yellowLight.style.background = '#ffeb3b'; yellowLight.style.boxShadow = '0 0 20px #ffeb3b';
                        setTimeout(() => {{ currentPhase = 2; animateLights(); }}, yellowTime);
                    }} else {{
                        redLight.style.background = '#f44336'; redLight.style.boxShadow = '0 0 20px #f44336';
                        setTimeout(() => {{ currentPhase = 0; animateLights(); }}, redTime);
                    }}
                }}
                animateLights();
            }})();
        </script>
        """
        return html
    except Exception as e:
        return f"<div style='color: red; padding: 20px;'>Error: {e}</div>"

# =========================
# 5) TRAIN / PLOT / PREPROCESS
# =========================
def upload_and_train(npz_file, graph_file, algorithm):
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if npz_file:
            shutil.copy(npz_file.name, NPZ_DEST)
        else:
            return "Please upload an NPZ file."

        if graph_file:
            shutil.copy(graph_file.name, GRAPH_CSV_DEST)
        else:
            return "Please upload a Graph CSV file."

        cp = CONFIG_PY.read_text()
        cp = pd.Series(cp.splitlines()).replace(r'^GRAPH = ".*"$', f'GRAPH = "{GRAPH_CSV_DEST.as_posix()}"', regex=True).str.cat(sep="\n")
        cp = pd.Series(cp.splitlines()).replace(r'^DATA_PATH = ".*"$', f'DATA_PATH = "{NPZ_DEST.as_posix()}"', regex=True).str.cat(sep="\n")
        cp = pd.Series(cp.splitlines()).replace(r'^MODEL = ".*"$', f'MODEL = "{algorithm}"', regex=True).str.cat(sep="\n")
        CONFIG_PY.write_text(cp)

        proc = subprocess.run(
            ["python", str(BASE_DIR / "train.py")],
            capture_output=True,
            text=True,
            check=True,
            cwd=BASE_DIR
        )
        logs = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
        return f"Training completed successfully!\n\nLogs:\n{logs[-2000:]}"
    except subprocess.CalledProcessError as e:
        return f"❌ Error during training:\n{e.stderr}\n{e.stdout}"
    except Exception as e:
        import traceback
        return f"❌ An unexpected error occurred:\n{traceback.format_exc()}"

def generate_plot(intersection, arm, zoom_start, zoom_end):
    try:
        if intersection is None or arm is None:
            return "Please select an Intersection and an Arm.", None

        loc_id = INTERSECTIONS[intersection][arm]

        subprocess.run(["python", str(PRED_CSV), str(REAL_FLOW_CSV), str(REAL_FLOW_GAR)], check=True, cwd=BASE_DIR)
        subprocess.run(["python", str(PRED_CSV), str(TEST_RESULTS_CSV), str(TEST_GAR)], check=True, cwd=BASE_DIR)

        lines = PLOT_PY.read_text().splitlines()
        patched = []
        for line in lines:
            if "location_id =" in line:
                patched.append(f"location_id = {loc_id}")
            elif "zoom_start_date = pd.Timestamp(" in line:
                patched.append(f'zoom_start_date = pd.Timestamp("{zoom_start}")')
            elif "zoom_end_date = pd.Timestamp(" in line:
                patched.append(f'zoom_end_date = pd.Timestamp("{zoom_end}")')
            elif "intersection_name =" in line:
                patched.append(f'intersection_name = "{intersection}"')
            elif "arm_name =" in line:
                patched.append(f'arm_name = "{arm}"')
            else:
                patched.append(line)
        PLOT_PY.write_text("\n".join(patched))

        subprocess.run(["python", str(PLOT_PY)], check=True, cwd=BASE_DIR)

        if PLOT_IMG.exists():
            return "Plot generated successfully!", np.array(Image.open(PLOT_IMG).convert("RGB"))
        return "❌ Plot image not found after generation.", None
    except subprocess.CalledProcessError as e:
        return f"❌ Error creating plot:\n{e.stderr}\n{e.stdout}", None
    except Exception as e:
        import traceback
        return f"❌ An unexpected error occurred:\n{traceback.format_exc()}", None

def select_algorithm_action(algorithm_name):
    return f"Algorithm selected: {algorithm_name}"

def datetime_to_minutes(dt):
    if pd.isna(dt):
        return None
    # Tam tarih-saat sıralaması için timestamp kullan
    return dt.timestamp()

def preprocess_excel_and_generate_npz_ui(excel_file):
    try:
        if excel_file is None:
            return "Please upload an Excel file.", None, None, gr.update(visible=False), gr.update(visible=False)

        xls = pd.ExcelFile(excel_file.name)
        sheet_names = xls.sheet_names

        all_datetimes = []
        for sheet in sheet_names:
            df_temp = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=10)
            mask = df_temp.apply(lambda row: row.astype(str).str.contains("Tarih").any(), axis=1)
            header_idx = mask.idxmax() if mask.any() else 0

            df = pd.read_excel(xls, sheet_name=sheet, header=header_idx)
            df["TarihSaat"] = pd.to_datetime(df["Tarih"].astype(str) + " " + df["Saat"].astype(str), errors="coerce")
            all_datetimes.extend(df["TarihSaat"].dropna().unique())

        unique_datetimes = sorted(set(all_datetimes), key=lambda x: datetime_to_minutes(x))
        datetime_to_timestep = {dt: i + 1 for i, dt in enumerate(unique_datetimes)}

        all_dfs = []
        location_offset = 0

        for sheet in sheet_names:
            df_temp = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=10)
            mask = df_temp.apply(lambda row: row.astype(str).str.contains("Tarih").any(), axis=1)
            header_idx = mask.idxmax() if mask.any() else 0

            df = pd.read_excel(xls, sheet_name=sheet, header=header_idx)

            candidate_cols = [
                col for col in df.columns
                if col not in ["Tarih", "Saat", "Toplam"]
                and "Unnamed" not in str(col)
            ]

            for col in candidate_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df_numeric = df[candidate_cols].select_dtypes(include="number")
            if df_numeric.empty:
                continue

            df["TarihSaat"] = pd.to_datetime(df["Tarih"].astype(str) + " " + df["Saat"].astype(str), errors="coerce")

            rows = []
            for _, row in df.iterrows():
                dt = row["TarihSaat"]
                if pd.isna(dt):
                    continue
                ts = datetime_to_timestep.get(dt)
                if ts is None:
                    continue

                for loc_idx, col_name in enumerate(df_numeric.columns):
                    flow_val = row[col_name] if not pd.isna(row[col_name]) else 0
                    rows.append({
                        "timestep": ts,
                        "location": location_offset + loc_idx,
                        "flow": flow_val,
                        "occupy": 1,
                        "speed": 1
                    })

            if rows:
                all_dfs.append(pd.DataFrame(rows))
                location_offset += len(df_numeric.columns)

        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.sort_values(by=["timestep", "location"], inplace=True)

        temp_dir = tempfile.mkdtemp()
        csv_path = os.path.join(temp_dir, "combined_fullcols_formatted.csv")
        npz_path = os.path.join(temp_dir, "kol_bazli_2.npz")

        combined_df.to_csv(csv_path, index=False)

        timesteps = combined_df["timestep"].max()
        locations = location_offset
        data = np.zeros((timesteps, locations, 3))

        for _, row in combined_df.iterrows():
            ts_idx = int(row["timestep"]) - 1
            loc_idx = int(row["location"])
            if ts_idx < timesteps and loc_idx < locations:
                data[ts_idx, loc_idx] = [row["flow"], row["occupy"], row["speed"]]

        np.savez(npz_path, data=data)

        return (
            f"✅ Preprocessing completed!\nNodes: {locations}, Timesteps: {timesteps}",
            gr.File(value=csv_path, visible=True),
            gr.File(value=npz_path, visible=True),
            gr.update(visible=True),
            gr.update(visible=True)
        )

    except Exception as e:
        import traceback
        return f"❌ Hata: {traceback.format_exc()}", None, None, gr.update(visible=False), gr.update(visible=False)

def clear_excel_inputs():
    return "Excel file cleared.", None, None, gr.update(visible=False), gr.update(visible=False)

# =========================
# HEADER / FOOTER + CSS
# =========================
try:
    main_logo_data_uri = get_base64_image_tag(SMARTTECH_LOGO_PATH, max_height=90)
except:
    main_logo_data_uri = ""

try:
    kayseri_logo_data_uri = get_base64_image_tag(KAYSERI_ULASIM_LOGO_PATH, max_height=80)
except:
    kayseri_logo_data_uri = ""

header_html = f"""
<div class="smarttech-header-container">
    <div style="display: flex; align-items: center;">
        <img src="{main_logo_data_uri}" alt="Logo" class="smarttech-logo"/>
        <div class="smarttech-text-group">
            <span class="smarttech-title">AI Solutions for Smart Cities:</span>
            <span class="smarttech-desc">Monitor the pulse of cities with real-time traffic analysis and sustainable solutions.</span>
        </div>
    </div>
    <button class="smarttech-button" onclick="window.open('https://smarttechforlife.org', '_blank')"> EXPLORE PLATFORM &rarr;</button>
</div>
<hr class="header-divider">
"""

footer_html = f"""
<div class="smarttech-footer-container">
    <hr class="footer-divider">
    <div class="collaboration-section">
        <p><strong>Thank You for Your Support and Partnership!</strong></p>
        <div class="collaboration-logos">
            <img src="{kayseri_logo_data_uri}" alt="Kayseri Ulaşım Logo" class="kayseri-logo"/>
        </div>
    </div>
    <div class="smarttech-contact-info">
        <p><strong>Contact Information</strong></p>
        <p>Yıldırım Beyazıt Neighborhood, Aşık Veysel Boulevard, Tekno 1 Building, No: 61, Interior Door No: 75, Melikgazi, Kayseri, Türkiye</p>
        <p>E-mail: <a href="mailto:smart.tech.arge@gmail.com">smart.tech.arge@gmail.com</a></p>
    </div>
</div>
"""

css_styles = """
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&family=Open+Sans:wght@400;600&display=swap');
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css');

html, body, #gradio-app { height:100% !important; width:100% !important; margin:0 !important; padding:0 !important; font-family:'Open Sans', sans-serif; }
.gradio-container { max-width:100% !important; margin:0 !important; padding:20px; height:100% !important; overflow-y:auto !important; display:flex; flex-direction:column; background-color:#ffffff; }
.gradio-tabs { background-color:#ffffff; border-radius:10px; display:flex; flex-direction:column; flex-grow:1; }

.smarttech-header-container { display:flex; align-items:center; justify-content:space-between; padding:5px 15px; margin:0 0 5px 0; background:linear-gradient(to right,#f8f8f8,#ffffff); border-radius:8px; flex-shrink:0; flex-wrap:wrap; gap:10px; }
.smarttech-logo { height:40px; margin-right:12px; }
.smarttech-text-group { display:flex; flex-direction:row; align-items:baseline; gap:8px; flex-wrap:wrap; }
.smarttech-title { font-family:'Montserrat', sans-serif; font-size:15px; font-weight:700; color:#2c3e50; }
.smarttech-desc { font-size:13px; color:#555555; }
.smarttech-button { background-color:#6C9F69; color:white; padding:6px 14px; border:none; border-radius:6px; cursor:pointer; font-size:12px; font-weight:600; transition:background-color 0.3s ease; box-shadow:0 2px 5px rgba(0, 128, 0, 0.2); white-space:nowrap; }
.smarttech-button:hover { background-color:#5A8757; }
.header-divider { border:0; height:1px; background-image:linear-gradient(to right, rgba(0,0,0,0), rgba(0,0,0,0.1), rgba(0,0,0,0)); margin-bottom:10px; flex-shrink:0; }

.smarttech-footer-container { margin-top:20px; padding-top:15px; background-color:#f8f8f8; border-top:1px solid #e0e0e0; text-align:center; font-size:12px; color:#555555; flex-shrink:0; border-radius:8px; padding-bottom:10px; }
.smarttech-contact-info a { color:#6C9F69; text-decoration:none; }
.collaboration-logos { display:flex; justify-content:center; align-items:center; }
.kayseri-logo { height:80px; object-fit:contain; margin:0 10px; }

.content-block { background-color:#ffffff; border:1px solid #e2e8f0; border-radius:4px; padding:20px; box-shadow:none; }
.train-model-button { background-color:#1e293b !important; color:#ffffff !important; border-radius:4px !important; font-weight:600 !important; text-transform:uppercase !important; letter-spacing:1px !important; height:48px !important; border:none !important; }
.select-button { background-color:#475569 !important; color:#ffffff !important; border-radius:4px !important; font-weight:600 !important; text-transform:uppercase !important; letter-spacing:1px !important; height:48px !important; border:none !important; }
.compact-row { margin-bottom:10px !important; gap:10px !important; }
.training-log-area { background-color:#f5f5f5 !important; border:1px solid #cccccc !important; padding:12px !important; min-height:300px !important; overflow-y:auto !important; font-family:'Consolas','monospace' !important; font-size:12px !important; color:#333333 !important; width:100% !important; }
.training-log-area textarea { background-color:#f5f5f5 !important; border:none !important; resize:none !important; }
"""

# =========================
# GRADIO APP
# =========================
with gr.Blocks(title="SmartTech AI Platform", css=css_styles) as demo:
    gr.HTML(header_html)

    current_intersection_state = gr.State(value="Beyazşehir")
    current_date_state = gr.State(value=DATES_ILDEM["normal veri(kasım)"][0])
    current_hour_state = gr.State(value=14)
    current_arm_state = gr.State(value=ANALYSIS_INTERSECTIONS.get("Beyazşehir", [None])[0])
    current_data_type_state = gr.State(value="normal veri(kasım)")

    with gr.Tabs() as tabs:

        with gr.TabItem("Data Preprocessing"):
            with gr.Column(elem_classes="content-block"):
                gr.Markdown("<p style='font-weight:700; color:#0f172a; text-align:center;'>UPLOAD EXCEL & PREPROCESS</p>")
                excel_input = gr.File(label="Upload Excel File", file_types=[".xlsx", ".xls"], file_count="single")
                preprocessing_status = gr.Textbox(label="Status", interactive=False, value="Please upload an Excel file to begin.")

                with gr.Row():
                    process_btn = gr.Button("PREPROCESS DATA", elem_classes="train-model-button")
                    clear_btn = gr.Button("CLEAR", elem_classes="train-model-button")

                with gr.Row():
                    csv_output_file = gr.File(label="Download Generated CSV", visible=False)
                    npz_output_file = gr.File(label="Download Generated NPZ", visible=False)

                process_btn.click(preprocess_excel_and_generate_npz_ui, inputs=[excel_input],
                                  outputs=[preprocessing_status, csv_output_file, npz_output_file, csv_output_file, npz_output_file])
                clear_btn.click(clear_excel_inputs, inputs=[],
                                outputs=[preprocessing_status, excel_input, csv_output_file, csv_output_file, npz_output_file])

        with gr.TabItem("Upload + Train"):
            with gr.Row(variant="panel"):
                with gr.Column(elem_classes="content-block"):
                    gr.Markdown("<p style='font-weight:700; color:#0f172a; text-align:center;'>UPLOAD NPZ</p>")
                    npz_input = gr.File(label="Upload NPZ File", file_types=[".npz"], file_count="single")
                    npz_output = gr.Markdown("No file selected yet.")

                with gr.Column(elem_classes="content-block"):
                    gr.Markdown("<p style='font-weight:700; color:#0f172a; text-align:center;'>UPLOAD CSV</p>")
                    csv_input = gr.File(label="Upload Graph CSV", file_types=[".csv"], file_count="single")
                    csv_output = gr.Markdown("No file selected yet.")

                with gr.Column(elem_classes="content-block"):
                    gr.Markdown("<p style='font-weight:700; color:#0f172a; text-align:center;'>SELECT ALGORITHM</p>")
                    algo_input = gr.Dropdown(["default", "Garnoldi", "APPNP", "GPRGNN"], label="Select Algorithm", value="default")
                    algorithm_status = gr.Markdown("Algorithm selection pending.")
                    select_algo_btn = gr.Button("SELECT", elem_classes="select-button")
                    select_algo_btn.click(select_algorithm_action, inputs=algo_input, outputs=algorithm_status)

            with gr.Row(variant="panel"):
                with gr.Column(scale=1, elem_classes="content-block"):
                    gr.Markdown("<p style='font-weight:700; color:#0f172a; text-align:center;'>TRAINING LOG</p>")
                    training_log_display = gr.Textbox(label="Log Output", lines=10, interactive=False,
                                                     value="Training logs will appear here when the process starts...",
                                                     elem_classes="training-log-area")

            with gr.Row():
                train_model_button = gr.Button("TRAIN THE MODEL", elem_classes="train-model-button")

            train_model_button.click(upload_and_train, inputs=[npz_input, csv_input, algo_input], outputs=training_log_display)

        with gr.TabItem("Plot"):
            with gr.Column(elem_classes="content-block", scale=1):
                gr.Markdown("<p style='font-weight:700; color:#0f172a; text-align:center;'>GENERATE PLOT</p>")
                plot_status_output = gr.Textbox(label="Plot Status", interactive=False, value="Select options and click Generate Plot.")
                with gr.Row():
                    intersection_dd = gr.Dropdown(choices=list(INTERSECTIONS.keys()), label="Select Intersection")
                    arm_dd = gr.Dropdown(label="Select Arm", choices=[], interactive=True)
                with gr.Row():
                    zoom_start = gr.Textbox(label="Start Date (YYYY-MM-DD HH:MM)", value="2025-06-01 02:00")
                    zoom_end = gr.Textbox(label="End Date (YYYY-MM-DD HH:MM)", value="2025-06-01 12:00")

                plot_btn = gr.Button("GENERATE PLOT", elem_classes="train-model-button")
                plot_out = gr.Image(type="numpy", label="Generated Plot", show_label=True, interactive=False)

                def update_arms(intersection):
                    if intersection:
                        return gr.update(choices=list(INTERSECTIONS[intersection].keys()), value=None, interactive=True)
                    return gr.update(choices=[], value=None, interactive=False)

                intersection_dd.change(update_arms, inputs=intersection_dd, outputs=arm_dd)
                plot_btn.click(generate_plot, [intersection_dd, arm_dd, zoom_start, zoom_end], [plot_status_output, plot_out])

        with gr.TabItem("Traffic Analysis"):
            with gr.Column(elem_classes="content-block"):
                gr.HTML("<h3 style='text-align:center; padding:10px; color:#0f172a; font-weight:700;'>TRAFİK YÖNETİM VE ANALİZ SİSTEMİ</h3>")

                with gr.Row():
                    with gr.Column(scale=1):
                        map_trigger = gr.Textbox(label="Lokasyon", elem_id="map_trigger", value="Beyazşehir")
                        gr.HTML(value=create_map())

                        with gr.Row(elem_classes="compact-row"):
                            data_type_in = gr.Dropdown(
                                choices=["karlı gün verisi(şubat)", "karlı gün verisi(ocak)", "normal veri(kasım)"],
                                label="Veri Tipi",
                                value="normal veri(kasım)",
                                visible=True
                            )
                            c_in = gr.Dropdown(
                                choices=ANALYSIS_INTERSECTIONS.get("Beyazşehir", []),
                                label="İncelenen Kol",
                                value=(ANALYSIS_INTERSECTIONS.get("Beyazşehir", [None])[0])
                            )

                        with gr.Row(elem_classes="compact-row"):
                            t_in = gr.Dropdown(
                                choices=DATES_ILDEM.get("normal veri(kasım)", DATES_KIZILIRMAK),
                                label="Tarih",
                                value=(DATES_ILDEM.get("normal veri(kasım)", DATES_KIZILIRMAK)[0])
                            )
                            s_in = gr.Dropdown(choices=list(range(24)), label="Saat", value=14)

                    with gr.Column(scale=1):
                        with gr.Row(elem_classes="compact-row"):
                            res1 = gr.HTML()
                            res2 = gr.HTML()
                        with gr.Row(elem_classes="compact-row"):
                            res3 = gr.HTML()
                            res4 = gr.HTML()
                        plot_out_analysis = gr.Plot(show_label=False)

                with gr.Row(elem_classes="compact-row"):
                    analyze_btn = gr.Button("ANALİZİ BAŞLAT", elem_classes="train-model-button")
                    traffic_light_btn = gr.Button("SİNYALİZASYON ÖNERİSİ", elem_classes="select-button")

                map_trigger.change(
                    fn=lambda kavsak, dt: (
                        handle_map_selection_v2(kavsak, dt)[0],
                        handle_map_selection_v2(kavsak, dt)[1],
                        handle_map_selection_v2(kavsak, dt)[2],
                        kavsak
                    ),
                    inputs=[map_trigger, current_data_type_state],
                    outputs=[c_in, t_in, data_type_in, current_intersection_state]
                )

                data_type_in.change(
                    fn=lambda dt, kavsak: (update_dates_by_data_type(dt, kavsak), dt),
                    inputs=[data_type_in, map_trigger],
                    outputs=[t_in, current_data_type_state]
                )

                map_trigger.change(lambda x: x, inputs=[map_trigger], outputs=[current_intersection_state])
                s_in.change(lambda x: x, inputs=[s_in], outputs=[current_hour_state])
                c_in.change(lambda x: x, inputs=[c_in], outputs=[current_arm_state])
                data_type_in.change(lambda x: x, inputs=[data_type_in], outputs=[current_data_type_state])

                analyze_btn.click(
                    fn=update_dashboard,
                    inputs=[map_trigger, t_in, s_in, c_in, data_type_in],
                    outputs=[res1, res2, res3, res4, plot_out_analysis, c_in]
                )

        with gr.TabItem("Traffic Light Suggestion"):
            with gr.Column(elem_classes="content-block"):
                gr.Markdown("<p style='font-weight:700; color:#0f172a; text-align:center;'>TRAFFIC LIGHT TIMING SUGGESTION</p>")
                suggestion_status = gr.Textbox(label="Status", interactive=False, value="Select intersection, arm, and hour to get traffic light suggestions.")

                with gr.Row():
                    tl_intersection_dd = gr.Dropdown(choices=list(INTERSECTIONS.keys()), label="Select Intersection")
                    tl_arm_dd = gr.Dropdown(label="Select Arm", choices=[], interactive=True)
                    tl_hour_dd = gr.Dropdown(choices=[7, 14, 17], label="Select Hour", interactive=True)

                calculate_btn = gr.Button("CALCULATE SUGGESTION", elem_classes="train-model-button")
                traffic_light_visual = gr.HTML(label="Traffic Light Visualization", value="")

                def update_tl_arms(intersection):
                    if intersection:
                        return gr.update(choices=list(INTERSECTIONS[intersection].keys()), value=None, interactive=True)
                    return gr.update(choices=[], value=None, interactive=False)

                def update_tl_arms_with_value(intersection, target_arm=None):
                    if intersection:
                        available_arms = list(INTERSECTIONS[intersection].keys())
                        value_to_set = target_arm if target_arm in available_arms else None
                        return gr.update(choices=available_arms, value=value_to_set, interactive=True)
                    return gr.update(choices=[], value=None, interactive=False)

                def calculate_traffic_light_suggestion(intersection, arm, hour, data_type="normal veri(kasım)"):
                    if not intersection or not arm or not hour:
                        return "Please select intersection, arm, and hour.", ""
                    try:
                        excel_mapping = {
                            "karlı gün verisi(şubat)": str(BASE_DIR / "şubat_tahmin.xlsx"),
                            "karlı gün verisi(ocak)": str(BASE_DIR / "ocak_tahmin.xlsx"),
                            "normal veri(kasım)": str(BASE_DIR / "kasım_tahmin.xlsx")
                        }
                        excel_file = excel_mapping.get(data_type, str(BASE_DIR / "kasım_tahmin.xlsx"))

                        arm_letter = arm.split(" - ")[0] if " - " in arm else arm
                        result = subprocess.run(
                            ["python", str(TRAFFIC_LIGHT_PY), intersection, arm_letter, str(hour), str(excel_file)],
                            capture_output=True, text=True, check=False, cwd=str(BASE_DIR)
                        )

                        if result.returncode != 0:
                            error_msg = f"❌ traffic_light.py failed with exit code {result.returncode}\n\n"
                            error_msg += f"STDOUT:\n{result.stdout}\n\n"
                            error_msg += f"STDERR:\n{result.stderr}"
                            return error_msg, ""

                        output_lines = result.stdout.strip().split("\n")
                        for line in reversed(output_lines):
                            if line.startswith("📤 SONUÇ:"):
                                values = line.replace("📤 SONUÇ:", "").strip().split(",")
                                if len(values) == 4:
                                    green, yellow, red, protection = values
                                    visual_html = generate_traffic_light_html(green, yellow, red, protection)
                                    return (f"✅ Suggestions calculated for {intersection} - {arm} at {hour}:00 ({data_type})", visual_html)
                        return f"⚠️ Unexpected output from traffic_light.py:\n{result.stdout}", ""
                    except Exception as e:
                        import traceback
                        return f"❌ Error:\n{traceback.format_exc()}", ""

                tl_data_type_state = gr.State(value="normal veri(kasım)")

                tl_intersection_dd.change(update_tl_arms, inputs=tl_intersection_dd, outputs=tl_arm_dd)
                calculate_btn.click(calculate_traffic_light_suggestion, inputs=[tl_intersection_dd, tl_arm_dd, tl_hour_dd, tl_data_type_state],
                                    outputs=[suggestion_status, traffic_light_visual])

                temp_intersection = gr.State()
                temp_arm = gr.State()

                def transfer_to_traffic_light_state_based(intersection_state, hour_state, arm_state, data_type_state):
                    kavsak = intersection_state if intersection_state else "Farabi"
                    kol = arm_state if arm_state else ANALYSIS_INTERSECTIONS.get(kavsak, ["A - OSMAN ÖZCAN CAD."])[0]
                    saat = hour_state if hour_state is not None else 14
                    data_type = data_type_state if data_type_state else "normal veri(kasım)"

                    intersection_mapping = {
                        "Emrah": "Gesi", "Farabi": "Gesi", "Yahya Kemal": "Gesi", "Gesi": "Gesi",
                        "İldem 1": "İldem 1", "İldem 2": "İldem 2", "İldem 3": "İldem 3",
                        "İldem 4": "İldem 4", "İldem 5": "İldem 5",
                        "Serkent": "Serkent", "Beyazşehir": "Beyazşehir", "Toki": "Toki"
                    }
                    mapped_intersection = intersection_mapping.get(kavsak, "Beyazşehir")
                    arm_letter = str(kol).split(" - ")[0] if " - " in str(kol) else "A"

                    available_arms = list(INTERSECTIONS.get(mapped_intersection, {}).keys())
                    matched_arm = next((a for a in available_arms if a.startswith(arm_letter)), available_arms[0] if available_arms else None)

                    valid_hours = [7, 14, 17]
                    hour_val = int(saat) if str(saat).isdigit() else 14
                    if hour_val not in valid_hours:
                        hour_val = min(valid_hours, key=lambda x: abs(x - hour_val))

                    msg = f"🎉 {kavsak} kavşağı {matched_arm} kolu saat {hour_val}:00 ({data_type}) parametreleri aktarıldı!"
                    return (
                        gr.update(value=mapped_intersection),
                        gr.update(value=hour_val),
                        msg,
                        mapped_intersection,
                        matched_arm,
                        data_type
                    )

                def update_arm_after_transfer(intersection, target_arm):
                    return update_tl_arms_with_value(intersection, target_arm)

                traffic_light_btn.click(
                    fn=transfer_to_traffic_light_state_based,
                    inputs=[current_intersection_state, current_hour_state, current_arm_state, current_data_type_state],
                    outputs=[tl_intersection_dd, tl_hour_dd, suggestion_status, temp_intersection, temp_arm, tl_data_type_state],
                    js="() => { const tabs = document.querySelectorAll('button[role=\"tab\"]'); if (tabs[4]) { tabs[4].click(); } }"
                ).then(
                    fn=update_arm_after_transfer,
                    inputs=[temp_intersection, temp_arm],
                    outputs=[tl_arm_dd]
                )

    gr.HTML(footer_html)

demo.launch(allowed_paths=[str(BASE_DIR), str(BASE_DIR.parent)], share=True, debug=True)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Uygulama durduruldu.")
except Exception as e:
    print(f"Hata: {e}")
    time.sleep(5)
