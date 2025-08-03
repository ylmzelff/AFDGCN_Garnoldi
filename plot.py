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
time_interval = 10  # minutes

# PATCHED VALUES HERE (DYNAMICALLY INJECTED BY INTERFACE)
location_id = 0
zoom_start_date = pd.Timestamp("2025-06-01 00:00:00")
zoom_end_date = pd.Timestamp("2025-06-01 15:50:00")
intersection_name = "Düvenönü"
arm_name = "Atatürk Bulvarı"

# Filter for selected location (intersection-arm ID)
df_pred_location = df_pred[df_pred['location'] == location_id]
df_real_period   = df_real[df_real['location'] == location_id]

# Generate time steps
time_steps_pred = [start_date + pd.Timedelta(minutes=time_interval * i) for i in range(len(df_pred_location))]
time_steps_real = [start_date + pd.Timedelta(minutes=time_interval * i) for i in range(len(df_real_period))]

# Clamp zoom dates within available data
zoom_start_date = max(zoom_start_date, time_steps_real[0])
zoom_end_date   = min(zoom_end_date, time_steps_real[-1])

# Zoom range
try:
    print(f"📌 df_real_period shape: {df_real_period.shape}")
    print(f"📌 time_steps_real first & last: {time_steps_real[0]} → {time_steps_real[-1]}")
    print(f"📌 Requested zoom: {zoom_start_date} → {zoom_end_date}")

    zoom_start = next(i for i, t in enumerate(time_steps_real) if t >= zoom_start_date)
    zoom_end   = next(i for i, t in enumerate(time_steps_real) if t >= zoom_end_date)

    print(f"📌 Zoom indices: start={zoom_start}, end={zoom_end}")

except StopIteration:
    print("❌ Zoom timestamps out of range.")
    sys.exit(1)

zoom_data = df_real_period['flow'][zoom_start:zoom_end].dropna()

if zoom_data.empty:
    print("⚠️ Warning: Zoom region has no valid flow data. Plotting full data instead.")
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

def custom_date_format(x, _):
    dt = mdates.num2date(x)
    return f"{dt.month}.{dt.day}-{dt.strftime('%H:%M')}"
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
plt.savefig("traffic_flow_zoomed.png")
plt.show()