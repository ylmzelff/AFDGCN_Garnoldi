# plot.py (Updated for intersection & arm with safe zoom)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import sys

# Load real and predicted data
df_real = pd.read_csv("./real_flow_gar.csv")
df_pred = pd.read_csv("./test_gar.csv", skiprows=1, header=None,
                      names=["timestep", "location", "flow", "occupy", "speed"])

# Parameters
start_date = pd.Timestamp("2025-06-01 00:00")
time_interval = 60  # minutes

# Intersection and arm names for each location_id
intersection_arm_names = [
    ("Yahya Kemal", "YAHYA KEMAL CAD. KUZEY"),
    ("Yahya Kemal", "KIZILIRMAK CAD DOĞU"),
    ("Yahya Kemal", "YAHYA KEMAL CAD. GÜNEY"),
    ("Yahya Kemal", "KIZILIRMAK CAD BATI"),
    ("Farabi", "FARABİ CAD KUZEY"),
    ("Farabi", "KIZILIRMAK CAD DOĞU"),
    ("Farabi", "FARABİ CAD GÜNEY"),
    ("Farabi", "KIZILIRMAK CAD BATI"),
    ("Emrah", "EMRAH CAD. KUZEY"),
    ("Emrah", "KIZILIRMAK CAD DOĞU"),
    ("Emrah", "EMRAH CAD GÜNEY"),
    ("Emrah", "KIZILIRMAK CAD BATI"),
]

zoom_start_date = pd.Timestamp("2025-06-01 00:00:00")
zoom_end_date = pd.Timestamp("2025-06-01 12:00:00")

def custom_date_format(x, _):
    dt = mdates.num2date(x)
    return f"{dt.month}.{dt.day}-{dt.strftime('%H:%M')}"

for location_id in range(12):
    intersection_name, arm_name = intersection_arm_names[location_id]

    # Filter for selected location (intersection-arm ID)
    df_pred_location = df_pred[df_pred['location'] == location_id]
    df_real_period   = df_real[df_real['location'] == location_id]

    # Generate time steps
    time_steps_pred = [start_date + pd.Timedelta(minutes=time_interval * i) for i in range(len(df_pred_location))]
    time_steps_real = [start_date + pd.Timedelta(minutes=time_interval * i) for i in range(len(df_real_period))]

    if not time_steps_real:
        print(f"❌ No real data for location_id {location_id}")
        continue

    # Clamp zoom dates within available data
    zoom_start_date_clamped = max(zoom_start_date, time_steps_real[0])
    zoom_end_date_clamped   = min(zoom_end_date, time_steps_real[-1])

    # Zoom range
    try:
        zoom_start = next(i for i, t in enumerate(time_steps_real) if t >= zoom_start_date_clamped)
        zoom_end   = next(i for i, t in enumerate(time_steps_real) if t >= zoom_end_date_clamped)
    except StopIteration:
        print(f"❌ Zoom timestamps out of range for location_id {location_id}.")
        continue

    zoom_data = df_real_period['flow'][zoom_start:zoom_end].dropna()

    if zoom_data.empty:
        print(f"⚠️ Warning: Zoom region has no valid flow data for location_id {location_id}. Plotting full data instead.")
        zoom_start = 0
        zoom_end = len(df_real_period)
        zoom_data = df_real_period['flow'].iloc[zoom_start:zoom_end].fillna(0)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(time_steps_real, df_real_period['flow'], label="Real Traffic Flow", color="blue", alpha=0.6)
    ax.plot(time_steps_pred, df_pred_location['flow'], label="Predicted Traffic Flow", color="orange", linestyle="--", alpha=0.6)
    ax.fill_between(time_steps_real[zoom_start:zoom_end],
                    df_real_period['flow'].min(), df_real_period['flow'].max(),
                    color="yellow", alpha=0.3, label="Zoomed Region")
    ax.set_title(f"{intersection_name} - {arm_name}", fontsize=14)
    ax.set_xlabel("Time")
    ax.set_ylabel("Traffic Flow")
    ax.legend(loc="upper left")
    ax.grid()
    ax.xaxis.set_major_formatter(FuncFormatter(custom_date_format))

    # Inset
    axins = ax.inset_axes([0.5, 0.5, 0.65, 0.6])
    axins.plot(time_steps_real, df_real_period['flow'], color="blue")
    axins.plot(time_steps_pred, df_pred_location['flow'], color="orange", linestyle="--")
    axins.set_xlim(time_steps_real[zoom_start], time_steps_real[zoom_end])
    axins.set_ylim(zoom_data.min() - 5, zoom_data.max() + 5)
    axins.grid()
    axins.set_xticklabels([])

    ax.indicate_inset_zoom(axins)

    plt.tight_layout()
    plt.savefig(f"traffic_flow_zoomed_{location_id}.png")
    plt.close(fig)