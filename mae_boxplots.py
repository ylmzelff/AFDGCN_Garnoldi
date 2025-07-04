import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os  # used for directory creation

# Base function configuration: we select which basis functions are active.
# display_name determines how they are labeled in the plots.
bases_config = {
    "Monomial":      {"active": False, "display_name": "Monomial"},
    "Chebyshev":     {"active": True,  "display_name": "Chebyshev"},
    "Legendre":      {"active": True,  "display_name": "Legendre"},
    "Jacobi":        {"active": False, "display_name": "Jacobi"},
    "PPR":           {"active": False, "display_name": "PPR"},
    "SChebyshev":    {"active": False, "display_name": "Scaled Chebyshev"}
}

# Listing only the active bases, the three plots below will be generated separately for each.
active_bases = [name for name, cfg in bases_config.items() if cfg["active"]]

# Connecting to the drive and setting the filters for Garnoldi
drive_base = "/content/drive/MyDrive/mae_files"
filters = ["g0", "g1", "g2", "g3"]
# Sayısal etiketleri kısaltmak için fonksiyon (örneğin: 1250 → 1.25K)
def format_val(val):
    return f"{val/1000:.2f}K" if val >= 1000 else f"{val:.2f}"

# PLOT 1: GARNOLDI FILTER BOXPLOTS – Visualizes the 4 Garnoldi filters
for basis in active_bases:
    basis_display = bases_config[basis]["display_name"]
    folder_path = os.path.join(drive_base, basis.lower())
    df_garnoldi = pd.DataFrame()

    print(f"\n[BASIS: {basis_display}] Loading Garnoldi filter data...")
    for filt in filters:
        filename = f"mae_values_garnoldi_{basis.lower()}_{filt}.csv"
        full_path = os.path.join(folder_path, filename)
        if os.path.exists(full_path):
            print(f"   ✔ Loaded: {filename}")
            df = pd.read_csv(full_path)
            df["Filter"] = filt
            df["Basis"] = basis_display
            df_garnoldi = pd.concat([df_garnoldi, df], ignore_index=True)
        else:
            print(f"Missing file: {filename}")

    if not df_garnoldi.empty:
        print("Drawing Garnoldi filter boxplot...")
        plt.figure(figsize=(12, 6))
        ax = sns.boxplot(x="Filter", y="Validation MAE", hue="Basis", data=df_garnoldi, palette="Set2")
        plt.title(f"Garnoldi – MAE Distribution Across Filters ({basis_display})")
        plt.xlabel("Filter")
        plt.ylabel("Validation MAE")
        plt.grid(True, linestyle="--", alpha=0.5)
        xticks = ax.get_xticks()
        for i, filt in enumerate(df_garnoldi["Filter"].unique()):
            mean_val = df_garnoldi[df_garnoldi["Filter"] == filt]["Validation MAE"].mean()
            ax.annotate(f"{mean_val:.2f}", (xticks[i], mean_val), ha='center', va='bottom', fontsize=10, fontweight='bold')
        out_path = f"figures/{basis.lower()}/garnoldi_filter_boxplot_{basis.lower()}.png"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved: {out_path}")

# PLOT 2: FULL MODEL COMPARISON – Compares all Garnoldi filters with other models
for basis in active_bases:
    basis_display = bases_config[basis]["display_name"]
    folder_path = os.path.join(drive_base, basis.lower())
    df_all = pd.DataFrame()
    print(f"\n[BASIS: {basis_display}] Comparing all models...")

    for filt in filters:
        fname = f"mae_values_garnoldi_{basis.lower()}_{filt}.csv"
        fpath = os.path.join(folder_path, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            df["Model"] = f"Garnoldi-{filt}"
            df_all = pd.concat([df_all, df], ignore_index=True)

    for model in ["APPNP", "AFDGCN", "GPRGNN"]:
        fname = f"mae_values_{model.lower()}_{basis.lower()}.csv"
        fpath = os.path.join(folder_path, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            df["Model"] = model
            df_all = pd.concat([df_all, df], ignore_index=True)

    if not df_all.empty:
        print("Drawing full model comparison boxplot...")
        plt.figure(figsize=(14, 6))
        ax = sns.boxplot(x="Model", y="Validation MAE", data=df_all, palette="pastel")

        plt.title(f"Model Comparison – MAE Distribution ({basis_display})", fontsize=14, fontweight='bold')
        plt.xlabel("Model", fontsize=12)
        plt.ylabel("Validation MAE", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)

        xticks = ax.get_xticks()
        models = df_all["Model"].unique()

        for i, model in enumerate(models):
            mean_val = df_all[df_all["Model"] == model]["Validation MAE"].mean()
            offset = mean_val * 0.05  # Etiketi biraz yukarı taşı
            ax.annotate(format_val(mean_val),
                        (xticks[i], mean_val + offset),
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold')

        out_path = f"figures/{basis.lower()}/all_models_comparison_{basis.lower()}_cleaned.png"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved: {out_path}")

# PLOT 3: BEST GARNOLDI FILTER VS MODELS – Shows the best Garnoldi filter compared to other 3 models
for basis in active_bases:
    basis_display = bases_config[basis]["display_name"]
    folder_path = os.path.join(drive_base, basis.lower())
    df_all = pd.DataFrame()
    print(f"\n[BASIS: {basis_display}] Selecting best Garnoldi filter...")

    filter_mae_means = {}
    for filt in filters:
        fname = f"mae_values_garnoldi_{basis.lower()}_{filt}.csv"
        fpath = os.path.join(folder_path, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            filter_mae_means[filt] = df["Validation MAE"].mean()

    if not filter_mae_means:
        print("No valid Garnoldi files found.")
        continue

    # Determine the best filter based on lowest mean MAE
    best_filter = min(filter_mae_means, key=filter_mae_means.get)
    print(f"Best Garnoldi filter: {best_filter.upper()}")
    best_path = os.path.join(folder_path, f"mae_values_garnoldi_{basis.lower()}_{best_filter}.csv")
    df_best = pd.read_csv(best_path)
    df_best["Model"] = f"Garnoldi-{best_filter}"
    df_all = pd.concat([df_all, df_best], ignore_index=True)

    for model in ["APPNP", "AFDGCN", "GPRGNN"]:
        fname = f"mae_values_{model.lower()}_{basis.lower()}.csv"
        fpath = os.path.join(folder_path, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            df["Model"] = model
            df_all = pd.concat([df_all, df], ignore_index=True)

    if not df_all.empty:
        print("Drawing best Garnoldi filter vs models boxplot...")
        plt.figure(figsize=(12, 6))
        ax = sns.boxplot(x="Model", y="Validation MAE", data=df_all, palette="Set1")
        plt.title(f"Best Garnoldi Filter vs Other Models ({basis_display})")
        plt.xlabel("Model")
        plt.ylabel("Validation MAE")
        plt.grid(True, linestyle="--", alpha=0.5)
        xticks = ax.get_xticks()
        for i, model in enumerate(df_all["Model"].unique()):
            mean_val = df_all[df_all["Model"] == model]["Validation MAE"].mean()
            ax.annotate(f"{mean_val:.2f}", (xticks[i], mean_val), ha='center', va='bottom', fontsize=10, fontweight='bold')
        out_path = f"figures/{basis.lower()}/best_garnoldi_vs_models_{basis.lower()}.png"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved: {out_path}")
