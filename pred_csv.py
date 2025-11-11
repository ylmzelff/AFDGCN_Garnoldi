# pred_csv.py (Updated for arm-based data)
import sys
import pandas as pd

if len(sys.argv) != 3:
    print("Usage: python pred_csv.py <input_file> <output_file>")
    sys.exit(1)

input_file  = sys.argv[1]
output_file = sys.argv[2]

df_input = pd.read_csv(input_file, skiprows=1, names=["flow"])

if df_input.empty:
    print(f"❌ Error: Input file {input_file} is empty.")
    sys.exit(1)

df_input["flow"] = df_input["flow"].fillna(0).round().astype(int).abs()

# Adjust the number of arms to match your dataset
num_arms = 12  # Update this if the number of arms changes (based on your Excel file)

timesteps = []
locations = []
flows = []
occupies = []
speeds = []

for i in range(len(df_input)):
    timestep = i // num_arms + 1
    location = i % num_arms
    timesteps.append(timestep)
    locations.append(location)
    flows.append(df_input.loc[i, "flow"])
    occupies.append(1)
    speeds.append(1)

df_output = pd.DataFrame({
    "timestep": timesteps,
    "location": locations,
    "flow": flows,
    "occupy": occupies,
    "speed": speeds
})

df_output.to_csv(output_file, index=False)
print(f"✅ Output file saved as {output_file}, shape: {df_output.shape}")
