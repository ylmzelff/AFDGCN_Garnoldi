import pandas as pd
import matplotlib.pyplot as plt

# Dosya yolları (aynı klasörde çalıştırıyorsan direkt isimler yeterli)
files = {
    "AFDGCN": "afdgcn_mae.csv",
    "APPNP": "appnp_mae.csv",
    "CHEBYSHEV": "chebyshev_g0_mae.csv",
    "GPRGNN": "gprgnn_mae.csv"
}

# MAE verilerini oku
mae_data = {}
for model_name, file_path in files.items():
    df = pd.read_csv(file_path)
    if "Validation MAE" not in df.columns:
        raise ValueError(f"{file_path} içinde 'Validation MAE' sütunu bulunamadı.")
    mae_data[model_name] = df["Validation MAE"].dropna().values

labels = list(mae_data.keys())
values = [mae_data[label] for label in labels]
means = [v.mean() for v in values]

# Grafik ayarları
plt.figure(figsize=(11, 5.2))

bp = plt.boxplot(
    values,
    labels=labels,
    patch_artist=True,
    showmeans=True,
    meanprops=dict(marker='D', markerfacecolor='blue', markeredgecolor='blue', markersize=4),
    flierprops=dict(marker='o', markersize=3, markerfacecolor='gray', markeredgecolor='gray', alpha=0.7),
    medianprops=dict(color='red', linewidth=1.5),
    whiskerprops=dict(color='gray', linewidth=1),
    capprops=dict(color='gray', linewidth=1),
    boxprops=dict(facecolor='#d9d9d9', color='gray')
)

# Ortalama değerlerini kutuların altına yaz
for i, mean_val in enumerate(means, start=1):
    plt.text(i, min(values[i-1]) + 0.05, f"{mean_val:.2f}",
             ha='center', va='bottom', fontsize=20, fontweight='bold')

plt.title("Comparative MAE: APPNP vs GPRGNN vs AFDGCN vs Chebyshev G0", fontsize=10)
plt.xlabel("Model")
plt.ylabel("Validation MAE")
plt.grid(axis='y', linestyle='--', alpha=0.35)
plt.tight_layout()
plt.show()

# Konsola ortalamaları da yazdır
print("Model bazlı ortalama Validation MAE:")
for label, m in zip(labels, means):
    print(f"{label}: {m:.4f}")