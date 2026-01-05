import gradio as gr
import pandas as pd
import numpy as np
import os, shutil, subprocess, json
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO
import tempfile

# === Paths ===
BASE_DIR = Path("/content/AFDGCN_Garnoldi")
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
TRAFFIC_LIGHT_PY = BASE_DIR / "traffic_light.py"
EXCEL_FILE = BASE_DIR / "gesi_kavşak_raporları.xlsx"

# Logonun doğru yolları: İki farklı logo için iki farklı yol tanımlandı.
KAYSERI_ULASIM_LOGO_PATH = BASE_DIR / "kayseri_ulaşım.png"
SMARTTECH_LOGO_PATH = BASE_DIR / "smarttecl_logo.png"


INTERSECTIONS = {
    "Gesi": {
        "A - SİVAS BULVARI-SİVAS YÖNÜ": 0,
        "B - GESİ CAD.": 1,
        "C - SİVAS BULV - ŞEHİR MERKEZİ": 2,
        "D - 381. SOKAK": 3
    },
    "Serkent": {
        "A - 822. SK": 4,
        "B - GESİ CAD. DOĞU": 5,
        "C - KOCASİNAN CAD.": 6,
        "D - GESİ CAD. BATI": 7
    },
    "Beyazşehir": {
        "A - OSMAN ÖZCAN CAD.": 8,
        "B - GESİ CAD DOĞU": 9,
        "C - MARKETLER ÇIKIŞ": 10,
        "D - GESİ CAD BATI": 11
    },
    "Toki": {
        "A - 832.CD": 12,
        "B - GESİ CAD. DOĞU": 13,
        "C - KADİR HAS BUL.": 14,
        "D - GESİ CAD BATI": 15
    },
    "İldem 1": {
        "A - DİNÇER SOKAK": 16,
        "B - ALPARSLANTÜRKEŞ BUL.": 17,
        "D - GESİ CAD": 18
    },
    "İldem 2": {
        "A - DİNÇER SOKAK KUZEY": 19,
        "B - HANEDAN SOKAK": 20,
        "C - DİNÇER SOKAK GÜNEY": 21,
        "D - FETİH SOKAK": 22
    },
    "İldem 3": {
        "A - DÜNDAR TAŞER CAD. KUZEY": 23,
        "C - DÜNDAR TAŞER CAD. GÜNEY": 24,
        "D - HANEDAN SOKAK": 25
    },
    "İldem 4": {
        "A - YAVUZSULTAN SELİM CAD. KUZEY": 26,
        "B - VATAN SOKAK": 27,
        "C - YAVUZSULTAN SELİM CAD. GÜNEY": 28,
        "D - ORKUN SOKAK": 29
    },
    "İldem 5": {
        "A - S.A BEDUK CAD. KUZEY": 30,
        "B - VATAN SOKAK BATI": 31,
        "C - S.A BEDUK CAD. GÜNEY": 32,
        "D - VATAN SOKAK DOĞU": 33
    }
}


# --- Logo Helper Function ---
def get_base64_image_tag(image_path: Path, max_height: int = 90) -> str:
    """Encodes an image to a base64 data URI."""
    if not image_path.exists():
        print(f"Warning: Image not found at {image_path}")
        return ""
    with Image.open(image_path) as img:
        if img.height > max_height:
            new_width = int(img.width * max_height / img.height)
            img = img.resize((new_width, max_height), Image.LANCZOS)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_b64}"


# === Helper Function: Generate Traffic Light Visualization HTML ===
def generate_traffic_light_html(green_sec, yellow_sec, red_sec, protection_sec):
    """
    Generates an animated traffic light visualization with timeline.
    
    Args:
        green_sec: Green light duration in seconds
        yellow_sec: Yellow light duration in seconds
        red_sec: Red light duration in seconds
        protection_sec: Protection time in seconds
    
    Returns:
        HTML string with embedded CSS and JavaScript
    """
    try:
        green = float(green_sec) if green_sec else 0
        yellow = float(yellow_sec) if yellow_sec else 0
        red = float(red_sec) if red_sec else 0
        protection = float(protection_sec) if protection_sec else 0
        
        total_time = green + yellow + red
        
        # Calculate percentages for timeline
        green_pct = (green / total_time * 100) if total_time > 0 else 0
        yellow_pct = (yellow / total_time * 100) if total_time > 0 else 0
        red_pct = (red / total_time * 100) if total_time > 0 else 0
        
        html = f"""
        <div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 30px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); border: 1px solid rgba(108, 159, 105, 0.2);">
            <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; gap: 30px;">
                <!-- Animated Traffic Light -->
                <div style="text-align: center;">
                    <h3 style="color: #2c5530; margin-bottom: 20px; font-family: 'Montserrat', sans-serif; font-weight: 600;">
                        <i class="fas fa-traffic-light" style="margin-right: 8px; color: #6C9F69;"></i>Traffic Light Simulation
                    </h3>
                    <div style="background: linear-gradient(145deg, #e3ede6 60%, #b7cfc0 100%); padding: 25px 20px; border-radius: 24px; box-shadow: 0 6px 24px rgba(108,159,105,0.13); display: inline-block; border: 2px solid #c7dbcf;">
                        <!-- Red Light -->
                        <div id="red-light" style="width: 80px; height: 80px; border-radius: 50%; margin: 10px auto; background: #6c1a1a; box-shadow: 0 2px 8px rgba(220,0,0,0.10) inset; transition: all 0.5s ease;">
                        </div>
                        <!-- Yellow Light -->
                        <div id="yellow-light" style="width: 80px; height: 80px; border-radius: 50%; margin: 10px auto; background: #bfae3a; box-shadow: 0 2px 8px rgba(200,200,0,0.10) inset; transition: all 0.5s ease;">
                        </div>
                        <!-- Green Light -->
                        <div id="green-light" style="width: 80px; height: 80px; border-radius: 50%; margin: 10px auto; background: #2e7d32; box-shadow: 0 2px 8px rgba(76,175,80,0.10) inset; transition: all 0.5s ease;">
                        </div>
                    </div>
                </div>
                <!-- Timeline Visualization -->
                <div style="flex: 1; min-width: 300px;">
                    <h3 style="color: #2c5530; margin-bottom: 15px; font-family: 'Montserrat', sans-serif; font-weight: 600;">
                        <i class="fas fa-clock" style="margin-right: 8px; color: #6C9F69;"></i>Timing Breakdown
                    </h3>
                    <!-- Timeline Bar -->
                    <div style="background: rgba(255,255,255,0.15); border-radius: 10px; overflow: hidden; height: 50px; display: flex; box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin-bottom: 20px; border: 1px solid rgba(255,255,255,0.2);">
                        <div style="background: linear-gradient(180deg, #7cb77f, #6C9F69); width: {green_pct}%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 14px; transition: all 0.3s ease; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">
                            {green:.0f}s
                        </div>
                        <div style="background: linear-gradient(180deg, #ffa726, #ff8f00); width: {yellow_pct}%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 14px; transition: all 0.3s ease; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">
                            {yellow:.0f}s
                        </div>
                        <div style="background: linear-gradient(180deg, #e57373, #d32f2f); width: {red_pct}%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 14px; transition: all 0.3s ease; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">
                            {red:.0f}s
                        </div>
                    </div>
                    <!-- Stats Cards -->
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 12px;">
                        <div style="background: linear-gradient(135deg, rgba(108, 159, 105, 0.15), rgba(90, 135, 87, 0.15)); border: 2px solid #6C9F69; border-radius: 10px; padding: 12px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                            <div style="color: #2c5530; font-size: 24px; font-weight: bold;">{green:.0f}s</div>
                            <div style="color: #6C9F69; font-size: 12px; margin-top: 4px; font-weight: 500;">Green</div>
                        </div>
                        <div style="background: linear-gradient(135deg, rgba(255, 179, 0, 0.15), rgba(255, 160, 0, 0.15)); border: 2px solid #ff8f00; border-radius: 10px; padding: 12px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                            <div style="color: #e65100; font-size: 24px; font-weight: bold;">{yellow:.0f}s</div>
                            <div style="color: #f57c00; font-size: 12px; margin-top: 4px; font-weight: 500;">Yellow</div>
                        </div>
                        <div style="background: linear-gradient(135deg, rgba(211, 47, 47, 0.15), rgba(198, 40, 40, 0.15)); border: 2px solid #d32f2f; border-radius: 10px; padding: 12px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                            <div style="color: #b71c1c; font-size: 24px; font-weight: bold;">{red:.0f}s</div>
                            <div style="color: #c62828; font-size: 12px; margin-top: 4px; font-weight: 500;">Red</div>
                        </div>
                        <div style="background: linear-gradient(135deg, rgba(108, 159, 105, 0.15), rgba(90, 135, 87, 0.15)); border: 2px solid #6C9F69; border-radius: 10px; padding: 12px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                            <div style="color: #2c5530; font-size: 24px; font-weight: bold;">{protection:.0f}s</div>
                            <div style="color: #6C9F69; font-size: 12px; margin-top: 4px; font-weight: 500;">Protection</div>
                        </div>
                    </div>
                    
                    <!-- Total Cycle Time -->
                    <div style="margin-top: 15px; text-align: center; background: linear-gradient(180deg, #7cb77f, #6C9F69); padding: 12px; border-radius: 8px; border: 1px solid #6C9F69; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                        <span style="color: #fff; font-size: 14px; font-weight: 500;">Total Cycle Time: </span>
                        <span style="color: #fff; font-size: 18px; font-weight: bold;">{total_time:.0f}s</span>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Animate traffic light
            (function() {{
                const greenLight = document.getElementById('green-light');
                const yellowLight = document.getElementById('yellow-light');
                const redLight = document.getElementById('red-light');
                
                const greenTime = {green * 1000};
                const yellowTime = {yellow * 1000};
                const redTime = {red * 1000};
                
                let currentPhase = 0;
                
                function resetLights() {{
                    greenLight.style.background = '#004a00';
                    greenLight.style.boxShadow = 'inset 0 4px 8px rgba(0,0,0,0.3)';
                    yellowLight.style.background = '#4a4400';
                    yellowLight.style.boxShadow = 'inset 0 4px 8px rgba(0,0,0,0.3)';
                    redLight.style.background = '#4a0000';
                    redLight.style.boxShadow = 'inset 0 4px 8px rgba(0,0,0,0.3)';
                }}
                
                function animateLights() {{
                    resetLights();
                    
                    if (currentPhase === 0) {{
                        // Green phase
                        greenLight.style.background = 'radial-gradient(circle, #4ade80, #22c55e)';
                        greenLight.style.boxShadow = '0 0 30px #22c55e, inset 0 4px 8px rgba(0,0,0,0.2)';
                        setTimeout(() => {{ currentPhase = 1; animateLights(); }}, greenTime);
                    }} else if (currentPhase === 1) {{
                        // Yellow phase
                        yellowLight.style.background = 'radial-gradient(circle, #fbbf24, #f59e0b)';
                        yellowLight.style.boxShadow = '0 0 30px #f59e0b, inset 0 4px 8px rgba(0,0,0,0.2)';
                        setTimeout(() => {{ currentPhase = 2; animateLights(); }}, yellowTime);
                    }} else {{
                        // Red phase
                        redLight.style.background = 'radial-gradient(circle, #f87171, #ef4444)';
                        redLight.style.boxShadow = '0 0 30px #ef4444, inset 0 4px 8px rgba(0,0,0,0.2)';
                        setTimeout(() => {{ currentPhase = 0; animateLights(); }}, redTime);
                    }}
                }}
                
                // Start animation
                animateLights();
            }})();
        </script>
        """
        
        return html
        
    except Exception as e:
        return f"<div style='color: red; padding: 20px;'>Error generating visualization: {e}</div>"


# === Upload + Train Function ===
def upload_and_train(npz_file, graph_file, algorithm):
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if npz_file:
            shutil.copy(npz_file.name, NPZ_DEST)
            print(f"Copied NPZ from {npz_file.name} to {NPZ_DEST}")
        else:
            return "Please upload an NPZ file.", None

        if graph_file:
            shutil.copy(graph_file.name, GRAPH_CSV_DEST)
            print(f"Copied CSV from {graph_file.name} to {GRAPH_CSV_DEST}")
        else:
            return "Please upload a Graph CSV file.", None

        # Update config.py
        cp = CONFIG_PY.read_text()
        cp = pd.Series(cp.splitlines()).replace(r'^GRAPH = ".*"$', f'GRAPH = "{GRAPH_CSV_DEST.as_posix()}"', regex=True).str.cat(sep='\n')
        cp = pd.Series(cp.splitlines()).replace(r'^DATA_PATH = ".*"$', f'DATA_PATH = "{NPZ_DEST.as_posix()}"', regex=True).str.cat(sep='\n')
        cp = pd.Series(cp.splitlines()).replace(r'^MODEL = ".*"$', f'MODEL = "{algorithm}"', regex=True).str.cat(sep='\n')
        CONFIG_PY.write_text(cp)
        print("config.py updated successfully.")

        # Train model
        print(f"Starting training with algorithm: {algorithm}")
        proc = subprocess.run(
            ["python", str(BASE_DIR / "train.py")],
            capture_output=True,
            text=True,
            check=True,
            cwd=BASE_DIR
        )
        logs = proc.stdout
        if proc.stderr:
            logs += "\n" + proc.stderr

        return f"Training completed successfully!\n\nLogs:\n{logs[-2000:]}"
    except subprocess.CalledProcessError as e:
        print(f"Subprocess Error:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
        return f"❌ Error during training:\n{e.stderr}\n{e.stdout}"
    except FileNotFoundError as e:
        return f"❌ Required file not found: {e}. Please ensure all scripts and data are in their correct paths."
    except Exception as e:
        import traceback
        print(f"General Error:\n{traceback.format_exc()}")
        return f"❌ An unexpected error occurred:\n{traceback.format_exc()}"

# === Plot Generation Function ===
def generate_plot(intersection, arm, zoom_start, zoom_end):
    try:
        if intersection is None or arm is None:
            return "Please select an Intersection and an Arm.", None

        loc_id = INTERSECTIONS[intersection][arm]

        # Step 1: Run pred_csv.py to generate GAR files
        print("Running pred_csv.py for real_flow.csv...")
        subprocess.run(["python", str(PRED_CSV), str(REAL_FLOW_CSV), str(REAL_FLOW_GAR)], check=True, cwd=BASE_DIR)
        print("Running pred_csv.py for test_results.csv...")
        subprocess.run(["python", str(PRED_CSV), str(TEST_RESULTS_CSV), str(TEST_GAR)], check=True, cwd=BASE_DIR)
        print("GAR files generated.")

        # Step 2: Patch plot.py
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
        print("plot.py patched successfully.")

        # Step 3: Run plot.py
        print("Running plot.py to generate image...")
        subprocess.run(["python", str(PLOT_PY)], check=True, cwd=BASE_DIR)
        print("plot.py execution complete.")

        if PLOT_IMG.exists():
            print(f"Plot image found at {PLOT_IMG}. Returning image.")
            return "Plot generated successfully!", np.array(Image.open(PLOT_IMG).convert("RGB"))
        else:
            print(f"Plot image not found at {PLOT_IMG}.")
            return "❌ Plot image not found after generation. Check plot.py output.", None

    except subprocess.CalledProcessError as e:
        print(f"Subprocess Error during plotting:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
        return f"❌ Error creating plot:\n{e.stderr}\n{e.stdout}", None
    except FileNotFoundError as e:
        print(f"File not found during plotting: {e}")
        return f"❌ Required file for plotting not found: {e}. Ensure pred_csv.py, plot.py, real_flow.csv, and test_results.csv exist.", None
    except Exception as e:
        import traceback
        print(f"General Error during plotting:\n{traceback.format_exc()}")
        return f"❌ An unexpected error occurred during plot generation:\n{traceback.format_exc()}", None

# --- Placeholder Function for "Select Algorithm" (as it's just a button in the UI) ---
def select_algorithm_action(algorithm_name):
    """Updates status based on algorithm selection."""
    return f"Algorithm selected: {algorithm_name}"

# === Preprocessing Functions (from the second code block) ===
def datetime_to_minutes(dt):
    if pd.isna(dt):
        return None
    return dt.day * 1440 + dt.hour * 60 + dt.minute

def preprocess_excel_and_generate_npz_ui(excel_file):
    try:
        if excel_file is None:
            return "Please upload an Excel file.", None, None, gr.update(visible=False), gr.update(visible=False)

        xls = pd.ExcelFile(excel_file.name)
        sheet_names = xls.sheet_names

        all_datetimes = []
        for sheet in sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet, skiprows=2)
            df['TarihSaat'] = pd.to_datetime(df['Tarih'].astype(str) + ' ' + df['Saat'].astype(str), errors='coerce')
            all_datetimes.extend(df['TarihSaat'].dropna().unique())

        unique_datetimes = sorted(set(all_datetimes), key=lambda x: datetime_to_minutes(x))
        datetime_to_timestep = {dt: i+1 for i, dt in enumerate(unique_datetimes)}

        all_dfs = []
        location_offset = 0
        sheet_info = []  # Debug bilgisi için
        
        for sheet in sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet, skiprows=2)
            candidate_cols = [col for col in df.columns if col not in ["Tarih", "Saat", "Toplam"]]
            
            # Tüm candidate sütunları numerik'e çevirmeyi dene
            for col in candidate_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df_numeric = df[candidate_cols].select_dtypes(include='number')

            if df_numeric.empty:
                sheet_info.append(f"❌ {sheet}: No numeric columns")
                continue

            num_cols = len(df_numeric.columns)
            col_names = ', '.join([f'"{col}"' for col in df_numeric.columns[:6]])  # İlk 6 sütun adı
            if len(df_numeric.columns) > 6:
                col_names += f", ... ({len(df_numeric.columns)} total)"
            sheet_info.append(f"✅ {sheet}: {num_cols} columns (locations {location_offset} to {location_offset + num_cols - 1})")
            sheet_info.append(f"   Columns: {col_names}")
            
            df['TarihSaat'] = pd.to_datetime(df['Tarih'].astype(str) + ' ' + df['Saat'].astype(str), errors='coerce')
            rows = []
            for _, row in df.iterrows():
                dt = row['TarihSaat']
                if pd.isna(dt): continue
                timestep_val = datetime_to_timestep.get(dt)
                if timestep_val is None: continue
                for loc_idx, col_name in enumerate(df_numeric.columns):
                    flow_val = row[col_name] if not pd.isna(row[col_name]) else 0
                    rows.append({
                        "timestep": timestep_val,
                        "location": location_offset + loc_idx,
                        "flow": flow_val,
                        "occupy": 1,
                        "speed": 1
                    })
            if rows:
                all_dfs.append(pd.DataFrame(rows))
                location_offset += len(df_numeric.columns)

        if not all_dfs:
            return "No valid numeric data found.", None, None, gr.update(visible=False), gr.update(visible=False)

        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.sort_values(by=['timestep', 'location'], inplace=True)
        combined_df = combined_df[['timestep', 'location', 'flow', 'occupy', 'speed']]

        temp_dir = tempfile.mkdtemp()
        csv_path = os.path.join(temp_dir, "combined_fullcols_formatted_0.csv")
        npz_path = os.path.join(temp_dir, "kol_bazli_0.npz")

        combined_df.to_csv(csv_path, index=False)
        timesteps = combined_df['timestep'].max()
        locations = combined_df['location'].max() + 1
        data = np.zeros((timesteps, locations, 3))
        for _, row in combined_df.iterrows():
            ts, loc = int(row['timestep']) - 1, int(row['location'])
            data[ts, loc] = [row['flow'], row['occupy'], row['speed']]
        np.savez(npz_path, data=data)

        # Detaylı bilgi mesajı
        info_msg = f"✅ Preprocessing completed!\n\n📊 Sheet Details:\n" + "\n".join(sheet_info)
        info_msg += f"\n\n📈 Total Locations: {locations} (0 to {locations-1})"
        info_msg += f"\n⏱️ Total Timesteps: {timesteps}"
        info_msg += f"\n💾 Data shape: ({timesteps}, {locations}, 3)"
        
        return info_msg, gr.File(value=csv_path, visible=True), gr.File(value=npz_path, visible=True), gr.update(visible=True), gr.update(visible=True)
    except Exception as e:
        import traceback
        return f"❌ An error occurred during preprocessing:\n{traceback.format_exc()}", gr.File(value=None, visible=False), gr.File(value=None, visible=False), gr.update(visible=False), gr.update(visible=False)

def clear_excel_inputs():
    """Clears the Excel upload and output files."""
    return "Excel file cleared.", None, None, gr.update(visible=False), gr.update(visible=False)

# --- UI Element Definitions ---
try:
    main_logo_data_uri = get_base64_image_tag(SMARTTECH_LOGO_PATH, max_height=90)
except FileNotFoundError:
    print(f"Warning: {SMARTTECH_LOGO_PATH.name} not found. Header might not display correctly.")
    main_logo_data_uri = ""

try:
    kayseri_logo_data_uri = get_base64_image_tag(KAYSERI_ULASIM_LOGO_PATH, max_height=80)
except FileNotFoundError:
    print(f"Warning: {KAYSERI_ULASIM_LOGO_PATH.name} not found. It won't be displayed.")
    kayseri_logo_data_uri = ""

# UPDATED header_html for more eye-catching text
header_html = f"""
<div class="smarttech-header-container">
    <div class="smarttech-logo-section">
        <img src="{main_logo_data_uri}" alt="SmartTech Logo" class="smarttech-logo"/>
    </div>
    <div class="smarttech-header-text">
        <h1 class="smarttech-main-heading">
            <i class="fas fa-rocket icon-main"></i> AI Solutions for Smart Cities
        </h1>
        <p class="smarttech-description">
            <i class="fas fa-microchip icon-desc"></i> AI-Powered Smart Transportation: Monitor the pulse of cities with real-time traffic analysis and forecasting.
            <br><i class="fas fa-chart-line icon-desc"></i>Data-Driven Decision Making: We offer detailed reporting and modeling for sustainable and effective solutions.
            <br><i class="fas fa-leaf icon-desc"></i> 	Green Cities, Clean Future: With our sustainability-focused technologies, we help build eco-friendly cities.
        </p>
        <button class="smarttech-button" onclick="window.open('https://smarttechforlife.org', '_blank')"> EXPLORE PLATFORM &rarr;</button>
    </div>
</div>
<hr class="header-divider">
"""

# UPDATED footer_html for more eye-catching text
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
/* Import Google Fonts for a more modern look */
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&family=Open+Sans:wght@400;600&display=swap');
/* Import Font Awesome for icons */
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css');

html, body, #gradio-app {
    height: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    font-family: 'Open Sans', sans-serif;
}

.gradio-container {
    max-width: 100% !important;
    margin: 0 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    background-color: #ffffff;
    padding: 20px;
    height: 100% !important;
    overflow-y: auto !important;
    display: flex;
    flex-direction: column;
}

.smarttech-header-container {
    display: flex;
    align-items: center;
    padding: 10px 8px;
    margin: 0 0 8px 0;
    background: linear-gradient(to right, #f8f8f8, #ffffff);
    border-radius: 8px;
    flex-shrink: 0;
}

.smarttech-logo-section {
    display: flex;
    align-items: center;
    margin-right: 20px;
    padding-left: 10px;
}

.smarttech-logo {
    height: 90px;
    margin-right: 8px;
}

.smarttech-logo-text {
    font-family: 'Montserrat', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #333333;
    letter-spacing: -0.5px;
}

.smarttech-header-text {
    flex-grow: 1;
}

.smarttech-main-heading {
    font-family: 'Montserrat', sans-serif;
    font-size: 20px;
    font-weight: 600;
    color: #2c3e50;
    margin-bottom: 5px;
    text-shadow: 0 1px 2px rgba(0,0,0,0.05);
}

.smarttech-description {
    font-size: 13px;
    color: #555555;
    margin-bottom: 10px;
    line-height: 1.5;
}

.icon-desc {
    color: #6C9F69;
    margin-right: 5px;
    font-size: 0.9em;
}

.icon-main {
    color: #6C9F69;
    margin-right: 5px;
    font-size: 1em;
}


.smarttech-button {
    background-color: #6C9F69;
    color: white;
    padding: 8px 18px;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 600;
    transition: background-color 0.3s ease, transform 0.2s ease, box-shadow 0.3s ease;
    box-shadow: 0 4px 10px rgba(0, 128, 0, 0.2);
}

.smarttech-button:hover {
    background-color: #5A8757;
    transform: translateY(-2px);
    box-shadow: 0 6px 15px rgba(0, 128, 0, 0.3);
}

.header-divider {
    border: 0;
    height: 1px;
    background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.1), rgba(0, 0, 0, 0));
    margin-bottom: 15px;
    flex-shrink: 0;
}


.content-block {
    background-color: #fcfcfc;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: space-between;
    min-height: 180px;
    text-align: center;
    flex: 1;
}

.block-title {
    font-family: 'Montserrat', sans-serif;
    font-size: 16px;
    font-weight: bold;
    color: #333333;
    margin-bottom: 15px;
    text-transform: uppercase;
}


.file-upload-button {
    background-color: #6C9F69;
    color: white;
    padding: 8px 12px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-size: 13px;
    transition: background-color 0.3s ease;
    margin-top: auto;
    width: 80%;
}

.file-upload-button:hover {
    background-color: #5A8757;
}


.select-button {
    background-color: #6C9F69;
    color: white;
    padding: 8px 12px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-size: 13px;
    transition: background-color 0.3s ease;
    margin-top: auto;
    width: 80%;
}

.select-button:hover {
    background-color: #5A8757;
}


.training-log-area {
    background-color: #f5f5f5 !important;
    border: 1px solid #cccccc !important;
    border-radius: 5px !important;
    padding: 12px !important;
    min-height: 300px !important;
    overflow-y: auto !important;
    font-family: 'Consolas', 'monospace' !important;
    font-size: 12px !important;
    color: #333333 !important;
    width: 100% !important;
    box-sizing: border-box !important;
    text-align: left !important;
}

.training-log-area textarea {
    background-color: #f5f5f5 !important;
    color: #333333 !important;
    border: none !important;
    outline: none !important;
    resize: none !important;
}

/* Train Model Button Styling */
.train-model-btn-container {
    text-align: center;
    margin-top: 15px;
    padding-bottom: 8px;
    flex-shrink: 0;
}

.train-model-button {
    background-color: #6C9F69;
    color: white;
    padding: 12px 25px;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-size: 16px;
    font-weight: bold;
    transition: background-color 0.3s ease, transform 0.2s ease;
}

.train-model-button:hover {
    background-color: #5A8757;
    transform: translateY(-2px);
}


.gradio-tabs {
    background-color: #ffffff;
    border-radius: 10px;
    box-shadow: none;
    flex-grow: 1;
    display: flex;
    flex-direction: column;
}

.gradio-tabs > div:first-child {
    border-bottom: 1px solid #e0e0e0;
    margin-bottom: 15px;
    flex-shrink: 0;
}

.gradio-tabs > div:first-child button {
    font-weight: normal;
    font-size: 15px;
    color: #555555;
    padding: 8px 18px;
    border-radius: 5px 5px 0 0;
    background-color: transparent;
    border: none;
    transition: color 0.3s ease, border-bottom 0.3s ease;
}

/* Active tab style */
.gradio-tabs > div:first-child button.selected {
    color: #6C9F69 !important;
    border-bottom: 2px solid #6C9F69 !important;
    font-weight: bold;
}

.gradio-tabs > div:first-child button:hover {
    color: #6C9F69;
}

.gradio-tabs > div:last-child {
    flex-grow: 1;
    display: flex;
    flex-direction: column;
}


/* --- NEW / MODIFIED CSS FOR FILE UPLOAD AND DROPDOWN --- */

.content-block .gr-file,
.content-block .gr-dropdown {
    background-color: #ffffff !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 8px !important;
    padding: 12px !important;
    margin-top: 8px;
    box-sizing: border-box;
    width: 100%;
}

.content-block .gr-file .gr-upload-target {
    background-color: #ffffff !important;
    border: 2px dashed #cccccc !important;
    border-radius: 8px !important;
    min-height: 80px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    align-items: center !important;
    color: #555555 !important;
    font-size: 13px !important;
    cursor: pointer !important;
    padding: 8px !important;
}

.content-block .gr-file .gr-upload-text {
    color: #555555 !important;
}
.content-block .gr-file .gr-upload-icon {
    color: #6C9F69 !important;
}


.content-block .gr-markdown:not(.block-title) {
    color: #333333;
    font-size: 13px;
    margin-top: 8px;
    text-align: center;
    width: 100%;
}

.content-block .gr-dropdown .wrap-inner--block {
    border: 1px solid #ccc !important;
    border-radius: 5px !important;
    background-color: #ffffff !important;
}

.content-block .gr-dropdown .gr-dropdown-option {
    padding: 6px 10px;
    font-size: 13px;
}


.gr-file > label > span:first-child {
    display: none !important;
}

.gr-file > label {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding-top: 3px;
}
.gr-file .small {
    color: #888;
    font-size: 0.9em;
}
.gr-file .small span {
    font-weight: bold;
}

.content-block .gr-image {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    margin-top: 15px;
    width: 100%;
    box-sizing: border-box;
}


.content-block .gr-file button {
    background-color: #f0f0f0 !important;
    color: #333333 !important;
    border: 1px solid #ccc !important;
    padding: 4px 8px !important;
    border-radius: 5px !important;
    margin-top: 3px !important;
    font-size: 12px !important;
}

.content-block .gr-file button:hover {
    background-color: #e0e0e0 !important;
}


.gradio-row {
    gap: 15px;
    flex-wrap: wrap;
    flex-grow: 1;
}


.gr-dropdown-input {
    color: #333333 !important;
}
.gr-dropdown-input span {
    color: #333333 !important;
}
.gr-dropdown-input .gr-dropdown-arrow {
    color: #333333 !important;
}
.gr-dropdown-label {
    color: #555555 !important;
    font-size: 13px !important;
    font-weight: normal !important;
    margin-bottom: 3px;
}

/* Adjustments for the file component's internal display of file name */
.gr-file-display {
    color: #333333 !important;
    font-size: 13px !important;
    margin-top: 8px;
    text-align: center;
    width: 100%;
}

/* Style for the default text "Upload NPZ File" */
.gr-file-label {
    color: #555555 !important;
    font-size: 13px !important;
    font-weight: normal !important;
    margin-bottom: 3px;
}

/* Correct the prompt text for file upload */
.gr-file .small {
    color: #888 !important;
    font-size: 0.9em !important;
    line-height: 1.5 !important;
}
.gr-file .small span {
    font-weight: bold !important;
}


.content-block .gr-markdown {
    color: #333333;
    font-size: 13px;
    margin-top: 8px;
    text-align: center;
    width: 100%;
}

.gradio-tabitem {
    display: flex;
    flex-direction: column;
    flex-grow: 1;
}

.gradio-column:has(.content-block) {
    display: flex;
    flex-direction: column;
}
.gradio-row[variant="panel"] {
    flex-grow: 1;
    align-items: stretch;
}

#plot_tab_content_column {
    flex-grow: 1;
    display: flex;
    flex-direction: column;
}


.smarttech-footer-container {
    margin-top: 20px;
    padding-top: 15px;
    background-color: #f8f8f8;
    border-top: 1px solid #e0e0e0;
    text-align: center;
    font-size: 12px;
    color: #555555;
    flex-shrink: 0;
    border-radius: 8px;
    padding-bottom: 10px;
}

.smarttech-contact-info p {
    margin: 5px 0;
    line-height: 1.4;
}

.smarttech-contact-info a {
    color: #6C9F69;
    text-decoration: none;
}

.smarttech-contact-info a:hover {
    text-decoration: underline;
}

.footer-divider {
    border: 0;
    height: 1px;
    background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.1), rgba(0, 0, 0, 0));
    margin-bottom: 15px;
}

button[aria-selected="true"] {
    color: #6C9F69 !important;
    border-bottom: 2px solid #6C9F69 !important;
    box-shadow: none !important;
}


button[aria-selected="true"]::after {
    border-bottom-color: #6C9F69 !important;
    background-color: #6C9F69 !important;
}


button[aria-selected="true"]:hover {
    color: #6C9F69 !important;
    border-bottom: 2px solid #6C9F69 !important;
}

/* --- Kayseri Ulaşım CSS'i --- */
.collaboration-section {
    margin-top: 10px;
    text-align: center;
}

.collaboration-section p {
    font-style: italic;
    font-size: 13px;
    margin-bottom: 8px;
    color: #777;
}

.collaboration-logos {
    display: flex;
    justify-content: center;
    align-items: center;
}

.kayseri-logo {
    height: 80px;
    object-fit: contain;
    margin: 0 10px;
}
"""

# Gradio Interface Definition
with gr.Blocks(title="SmartTech AI Platform", css=css_styles) as demo:
    # 1. Header Section
    gr.HTML(header_html)

    # Tabs for different functionalities
    with gr.Tabs() as tabs:
        with gr.TabItem("Data Preprocessing"):
            with gr.Column(elem_classes="content-block"):
                gr.Markdown("<p class='block-title'>UPLOAD EXCEL & PREPROCESS</p>")
                excel_input = gr.File(
                    label="Upload Excel File",
                    file_types=[".xlsx", ".xls"],
                    file_count="single"
                )
                preprocessing_status = gr.Textbox(label="Status", interactive=False, value="Please upload an Excel file to begin.")

                with gr.Row():
                    process_btn = gr.Button("PREPROCESS DATA", elem_classes="train-model-button")
                    clear_btn = gr.Button("CLEAR", elem_classes="train-model-button")

                with gr.Row():
                    csv_output_file = gr.File(label="Download Generated CSV", visible=False)
                    npz_output_file = gr.File(label="Download Generated NPZ", visible=False)

                process_btn.click(
                    preprocess_excel_and_generate_npz_ui,
                    inputs=[excel_input],
                    outputs=[preprocessing_status, csv_output_file, npz_output_file, csv_output_file, npz_output_file]
                )
                clear_btn.click(
                    clear_excel_inputs,
                    inputs=[],
                    outputs=[preprocessing_status, excel_input, csv_output_file, csv_output_file, npz_output_file]
                )

        with gr.TabItem("Upload + Train"):
            # Row for the first three blocks (side-by-side)
            with gr.Row(variant="panel"):
                # Upload NPZ Block
                with gr.Column(elem_classes="content-block"):
                    gr.Markdown("<p class='block-title'>UPLOAD NPZ</p>")
                    npz_input = gr.File(
                        label="Upload NPZ File",
                        file_types=[".npz"],
                        file_count="single"
                    )
                    npz_output = gr.Markdown("No file selected yet.")

                # Upload CSV Block
                with gr.Column(elem_classes="content-block"):
                    gr.Markdown("<p class='block-title'>UPLOAD CSV</p>")
                    csv_input = gr.File(
                        label="Upload Graph CSV",
                        file_types=[".csv"],
                        file_count="single"
                    )
                    csv_output = gr.Markdown("No file selected yet.")

                # Select Algorithm Block
                with gr.Column(elem_classes="content-block"):
                    gr.Markdown("<p class='block-title'>SELECT ALGORITHM</p>")
                    algo_input = gr.Dropdown(
                        ["default", "Garnoldi", "APPNP", "GPRGNN"],
                        label="Select Algorithm",
                        value="default"
                    )
                    algorithm_status = gr.Markdown("Algorithm selection pending.")
                    select_algo_btn = gr.Button("SELECT", elem_classes="select-button")
                    select_algo_btn.click(select_algorithm_action, inputs=algo_input, outputs=algorithm_status)

            # New Row for the Training Log (below the first three, full width)
            with gr.Row(variant="panel"):
                with gr.Column(scale=1, elem_classes="content-block"):
                    gr.Markdown("<p class='block-title'>TRAINING LOG</p>")
                    training_log_display = gr.Textbox(
                        label="Log Output",
                        lines=10,
                        interactive=False,
                        value="Training logs will appear here when the process starts...",
                        elem_classes="training-log-area"
                    )

            # Train the Model Button (placed outside the main content grid but still within the tab)
            with gr.Row(elem_classes="train-model-btn-container"):
                train_model_button = gr.Button("TRAIN THE MODEL", elem_classes="train-model-button")

            # Link backend functions
            npz_input.upload(lambda f: gr.Markdown(f"**NPZ File Uploaded:** `{f.name.split('/')[-1]}`" if f else "No file selected yet."), npz_input, npz_output)
            csv_input.upload(lambda f: gr.Markdown(f"**CSV File Uploaded:** `{f.name.split('/')[-1]}`" if f else "No file selected yet."), csv_input, csv_output)
            algo_input.change(select_algorithm_action, inputs=algo_input, outputs=algorithm_status)
            train_model_button.click(
                upload_and_train,
                inputs=[npz_input, csv_input, algo_input],
                outputs=training_log_display
            )

        with gr.TabItem("Plot"):
            # Applying content-block styling to the whole plot section
            with gr.Column(elem_classes="content-block", scale=1, elem_id="plot_tab_content_column"):
                gr.Markdown("<p class='block-title'>GENERATE PLOT</p>")
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

        with gr.TabItem("Traffic Light Suggestion"):
            with gr.Column(elem_classes="content-block"):
                gr.Markdown("<p class='block-title'>TRAFFIC LIGHT TIMING SUGGESTION</p>")
                suggestion_status = gr.Textbox(label="Status", interactive=False, value="Select intersection, arm, and hour to get traffic light suggestions.")

                with gr.Row():
                    tl_intersection_dd = gr.Dropdown(choices=list(INTERSECTIONS.keys()), label="Select Intersection")
                    tl_arm_dd = gr.Dropdown(label="Select Arm", choices=[], interactive=True)
                    tl_hour_dd = gr.Dropdown(choices=[7, 14, 17], label="Select Hour", interactive=True)

                calculate_btn = gr.Button("CALCULATE SUGGESTION", elem_classes="train-model-button")

                # Görsel Trafik Işığı Bileşeni - Hesaplamadan ÖNCE eklendi
                traffic_light_visual = gr.HTML(label="Traffic Light Visualization", value="")

                def update_tl_arms(intersection):
                    if intersection:
                        return gr.update(choices=list(INTERSECTIONS[intersection].keys()), value=None, interactive=True)
                    return gr.update(choices=[], value=None, interactive=False)

                def calculate_traffic_light_suggestion(intersection, arm, hour):
                    if not intersection or not arm or not hour:
                        return "Please select intersection, arm, and hour.", ""

                    try:
                        # Kol adlarından sadece harf kısmını al (A, B, C, D)
                        arm_letter = arm.split(" - ")[0] if " - " in arm else arm

                        # Run traffic_light.py script with updated parameters
                        print(f"Running traffic_light.py for {intersection} - {arm_letter} at hour {hour}...")
                        result = subprocess.run(
                            ["python", str(TRAFFIC_LIGHT_PY), intersection, arm_letter, str(hour), str(EXCEL_FILE)],
                            capture_output=True,
                            text=True,
                            check=True,
                            cwd=str(BASE_DIR)
                        )

                        # Parse the output (expecting: green,yellow,red,protection)
                        output_lines = result.stdout.strip().split('\n')
                        for line in reversed(output_lines):
                            if line.startswith("📤 SONUÇ:"):
                                values = line.replace("📤 SONUÇ:", "").strip().split(',')
                                if len(values) == 4:
                                    green, yellow, red, protection = values
                                    # Generate HTML visualization
                                    visual_html = generate_traffic_light_html(green, yellow, red, protection)
                                    return (
                                        f"✅ Suggestions calculated for {intersection} - {arm} at {hour}:00",
                                        visual_html
                                    )

                        return f"⚠️ Unexpected output from traffic_light.py:\n{result.stdout}", ""

                    except subprocess.CalledProcessError as e:
                        print(f"Error running traffic_light.py:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
                        return f"❌ Error calculating suggestions:\n{e.stderr}\n{e.stdout}", ""
                    except FileNotFoundError:
                        return f"❌ traffic_light.py not found. Please ensure the file exists.", ""
                    except Exception as e:
                        import traceback
                        print(f"Error in traffic light calculation:\n{traceback.format_exc()}")
                        return f"❌ An error occurred:\n{traceback.format_exc()}", ""

                tl_intersection_dd.change(update_tl_arms, inputs=tl_intersection_dd, outputs=tl_arm_dd)
                calculate_btn.click(
                    calculate_traffic_light_suggestion,
                    inputs=[tl_intersection_dd, tl_arm_dd, tl_hour_dd],
                    outputs=[suggestion_status, traffic_light_visual]
                )

    # 2. Footer Section for contact information and collaboration
    gr.HTML(footer_html)

# Launch the independent Gradio interface
demo.launch(
    allowed_paths=[str(BASE_DIR), str(BASE_DIR.parent)]
)
