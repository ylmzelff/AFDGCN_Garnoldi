import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

# Load Real Data
real_data_path = r"./real_flow_gar_yeni.csv"
df_real = pd.read_csv(real_data_path)

# Load Prediction Data
pred_data_path = r"./test_gar_yeni.csv"
#pred_data_path = r"./test_result_pems_def.csv"
df_pred = pd.read_csv(pred_data_path, skiprows=1, header=None, names=['timestep', 'location', 'flow', 'occupy', 'speed'])

# Define Parameters
start_date = pd.Timestamp("2025-02-01 00:00")  # Adjusted to reflect test period

# Time interval details
time_interval = 10  # Time interval in minutes
daily_time_steps = int((24 * 60) / time_interval)  # Number of timesteps per day
test_time_steps = 604  # Number of test timesteps (~48.42 hours)

# Extract relevant test data
df_pred_location = df_pred[df_pred['location'] == 0].iloc[-test_time_steps:]
df_real_period = df_real[df_real['location'] == 0].iloc[-test_time_steps:]

# Generate Time Steps
time_steps_pred = [start_date + pd.Timedelta(minutes=time_interval * i) for i in range(len(df_pred_location))]
time_steps_real = [start_date + pd.Timedelta(minutes=time_interval * i) for i in range(len(df_real_period))]

# Define zoom range based on date and hour
zoom_start_date = pd.Timestamp("2025-02-01 02:00")  # Modify this
zoom_end_date = pd.Timestamp("2025-02-01 12:00")    # Modify this

zoom_start = next(i for i, t in enumerate(time_steps_real) if t >= zoom_start_date)
zoom_end = next(i for i, t in enumerate(time_steps_real) if t >= zoom_end_date)

# Plot
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(time_steps_real, df_real_period['flow'], label="Real Traffic Flow", color="blue", alpha=0.6)
ax.plot(time_steps_pred, df_pred_location['flow'], label="Predicted Traffic Flow", color="orange", linestyle="--", alpha=0.6)

# Highlight zoomed region
ax.fill_between(time_steps_real[zoom_start:zoom_end],
                df_real_period['flow'].min(),
                df_real_period['flow'].max(),
                color="yellow", alpha=0.3, label="Zoomed Region")

ax.set_xlabel("Time")
ax.set_ylabel("Traffic Flow")
ax.legend(loc="upper left")
ax.grid()

# Define custom date formatter
def custom_date_format(x, _):
    dt = mdates.num2date(x)
    return f"{dt.month}.{dt.day}-{dt.strftime('%H:%M')}"

ax.xaxis.set_major_formatter(FuncFormatter(custom_date_format))

# Zoomed Portion
axins = ax.inset_axes([0.5, 0.5, 0.65, 0.6])
axins.plot(time_steps_real, df_real_period['flow'], label="Real Traffic Flow", color="blue")
axins.plot(time_steps_pred, df_pred_location['flow'], label="Predicted Traffic Flow", color="orange", linestyle="--")
axins.set_xlim(time_steps_real[zoom_start], time_steps_real[zoom_end])
axins.set_ylim(min(df_real_period['flow'][zoom_start:zoom_end]) - 5, max(df_real_period['flow'][zoom_start:zoom_end]) + 5)
axins.grid()
#axins.xaxis.set_major_formatter(FuncFormatter(custom_date_format))
axins.set_xticklabels([])

ax.indicate_inset_zoom(axins)

plt.tight_layout()
plt.savefig("traffic_flow_zoomed.png")
plt.show()
