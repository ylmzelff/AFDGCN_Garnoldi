import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

# Load Real Data
real_data_path = r"./real_gar_garnoldi (4).csv"
df_real = pd.read_csv(real_data_path)

# Load Prediction Data
pred_data_path = r"./test_gar_garnoldi (4).csv"
df_pred = pd.read_csv(pred_data_path)

# Define Parameters
# Sivas verisi: gercek takvim tarihleri npz/csv'ye tasinmadi (boslu 184 gun),
# bu yuzden x ekseni GERCEK tarihleri degil, sadece test setindeki sirayi
# saatlik adimlarla gosteren SENTETIK bir zaman eksenidir.
start_date = pd.Timestamp("2026-01-01 00:00")  # sentetik referans baslangic


# Time interval details
time_interval = 60  # Time interval in minutes (Sivas verisi saatlik)
daily_time_steps = int((24 * 60) / time_interval)  # Number of timesteps per day
test_time_steps = 882  # Test seti boyu (yeni 4416 timestep'lik veri seti icin)

# Extract relevant test data
df_pred_location = df_pred[df_pred['location'] == 1].iloc[-test_time_steps:]
df_real_period = df_real[df_real['location'] == 1].iloc[-test_time_steps:]

# Generate Time Steps
time_steps_pred = [start_date + pd.Timedelta(minutes=time_interval * i) for i in range(len(df_pred_location))]
time_steps_real = [start_date + pd.Timedelta(minutes=time_interval * i) for i in range(len(df_real_period))]

# Define zoom range based on date and hour
zoom_start_date = start_date                        # zoom başlangıcı
zoom_end_date = start_date + pd.Timedelta(days=4)    # zoom bitişi (ilk 4 gün)

zoom_start = next((i for i, t in enumerate(time_steps_real) if t >= zoom_start_date), 0)
zoom_end = next((i for i, t in enumerate(time_steps_real) if t >= zoom_end_date), len(time_steps_real) - 1)

# Ensure zoom_end is greater than zoom_start
if zoom_end <= zoom_start:
    zoom_end = min(zoom_start + 60, len(time_steps_real) - 1)  # Default to 60 timesteps (60 hours)

# Plot - Optimized for 2x2 layout on A4
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(time_steps_real, df_real_period['flow'], label="Real Traffic Flow", color="blue", alpha=0.6, linewidth=1.5)
ax.plot(time_steps_pred, df_pred_location['flow'], label="Predicted Traffic Flow", color="orange", linestyle="--", alpha=0.6, linewidth=1.5)

# Highlight zoomed region
ax.fill_between(time_steps_real[zoom_start:zoom_end],
                df_real_period['flow'].min(),
                df_real_period['flow'].max(),
                color="yellow", alpha=0.3, label="Zoomed Region")

ax.set_xlabel("Time", fontsize=14, fontweight='bold')
ax.set_ylabel("Traffic Flow", fontsize=14, fontweight='bold')
# ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax.grid(alpha=0.3)
ax.tick_params(axis='both', which='major', labelsize=12, pad=2)
# Reduce margins
ax.margins(x=0.01, y=0.05)

# Define custom date formatter
def custom_date_format(x, _):
    dt = mdates.num2date(x)
    return f"{dt.month}.{dt.day}-{dt.strftime('%H:%M')}"

# Show fewer dates with larger font
ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))  # Every 4 days
ax.xaxis.set_major_formatter(FuncFormatter(custom_date_format))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center', fontweight='bold')
plt.setp(ax.yaxis.get_majorticklabels(), fontweight='bold')

# Zoomed Portion
axins = ax.inset_axes([0.52, 0.52, 0.63, 0.55])
axins.plot(time_steps_real, df_real_period['flow'], label="Real Traffic Flow", color="blue", linewidth=1.2)
axins.plot(time_steps_pred, df_pred_location['flow'], label="Predicted Traffic Flow", color="orange", linestyle="--", linewidth=1.2)
axins.set_xlim(time_steps_real[zoom_start], time_steps_real[zoom_end])

# Safe min/max calculation with fallback
zoom_data = df_real_period['flow'].iloc[zoom_start:zoom_end]
if len(zoom_data) > 0:
    axins.set_ylim(zoom_data.min() - 5, zoom_data.max() + 5)
else:
    axins.set_ylim(df_real_period['flow'].min() - 5, df_real_period['flow'].max() + 5)

axins.grid(alpha=0.3)
axins.tick_params(axis='both', which='major', labelsize=8, pad=1)
#axins.xaxis.set_major_formatter(FuncFormatter(custom_date_format))
axins.set_xticklabels([])

ax.indicate_inset_zoom(axins, edgecolor="gray", linewidth=1)

# Optimize layout for A4 printing
plt.tight_layout(pad=0.5)
plt.savefig("traffic_flow_zoomed.png", dpi=300, bbox_inches='tight')
plt.show()