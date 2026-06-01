"""
Seq2Seq AFDGCN Tahmin Degerlendirme Scripti
===========================================
Egitim sonrasi modelin ne kadar iyi tahmin yaptigini anlamak icin kullanilir.

Calistirma (Colab veya yerel):
    python evaluate.py

Uretilen dosyalar:
    evaluation_results.png  — gercek vs tahmin grafigi (her kavşak + kol)
    evaluation_metrics.csv  — MAE / RMSE / MAPE her kavşak ve saat dilimi icin
"""

import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import args
from model.AFDGCN import Model as Network, ALGO
from lib.dataloader import get_dataloader
from lib.load_graph import get_adjacency_matrix
import scipy.sparse as sp

# ─── Kavşak / Kol isimleri ───────────────────────────────────────────────────

JUNCTION_NAMES = {
    89: "Gesi", 187: "Serkent", 95: "Beyazsehir", 121: "Toki",
    184: "Ildem 1", 188: "Ildem 2", 117: "Ildem 3",
    192: "Ildem 4", 194: "Ildem 5",
}
NODE_MAP = {
    89:  {"A": 0,  "B": 1,  "C": 2,  "D": 3},
    187: {"A": 4,  "B": 5,  "C": 6,  "D": 7},
    95:  {"A": 8,  "B": 9,  "C": 10, "D": 11},
    121: {"A": 12, "B": 13, "C": 14, "D": 15},
    184: {"A": 16, "B": 17,           "D": 18},
    188: {"A": 19, "B": 20, "C": 21, "D": 22},
    117: {"A": 23,           "C": 24, "D": 25},
    192: {"A": 26, "B": 27, "C": 28, "D": 29},
    194: {"A": 30, "B": 31, "C": 32, "D": 33},
}

SLOT_LABELS = [
    f"{h:02d}:{m:02d}"
    for h in range(24) for m in range(0, 60, 10)
]

# ─── Model yukle ─────────────────────────────────────────────────────────────

def load_best_model(device):
    """En iyi kayitli modeli yukler."""
    # Olasi model yollari
    candidates = [
        PROJECT_ROOT / "Kayseri_AFDGCN_best_model.pth",
        PROJECT_ROOT / "saved_models" / "kayseri_ildem_v2.pth",
        PROJECT_ROOT / "saved_models" / "kayseri_ildem_v1.pth",
    ]
    # Ayrica log dizininde ara
    for f in PROJECT_ROOT.glob("**/*best_model*.pth"):
        candidates.insert(0, f)

    model_path = None
    for c in candidates:
        if c.exists():
            model_path = c
            break

    if model_path is None:
        raise FileNotFoundError(
            "Egitilmis model bulunamadi. Once 'python train.py' calistirin."
        )
    print(f"Model yukleniyor: {model_path}")

    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    input_dim  = ckpt.get("input_dim", 1 + (2 if args.tod else 0))
    horizon    = args.horizon

    Adj = get_adjacency_matrix(args.graph_path, args.num_nodes,
                               type="connectivity", id_filename=args.filename_id)
    def norm(adj):
        adj = sp.coo_matrix(adj)
        rs = np.array(adj.sum(1))
        d  = np.power(rs, -0.5).flatten()
        d[np.isinf(d)] = 0.
        D = sp.diags(d)
        return np.array(D.dot(adj).dot(D).toarray())
    A = torch.tensor(norm(Adj), dtype=torch.float32).to(device)

    import model.AFDGCN as afdgcn_mod
    saved_algo = afdgcn_mod.ALGO
    afdgcn_mod.ALGO = "Garnoldi"
    net = Network(
        num_node=args.num_nodes, input_dim=input_dim,
        hidden_dim=64, output_dim=1, embed_dim=34, cheb_k=2,
        horizon=horizon, num_layers=1, heads=4, timesteps=args.lag,
        A=A, kernel_size=5, use_seq2seq=True,
    )
    afdgcn_mod.ALGO = saved_algo
    net.load_state_dict(state_dict, strict=False)
    net.eval()
    return net.to(device), input_dim

# ─── Test verisi uzerinde calistir ───────────────────────────────────────────

def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    _, _, test_loader, scaler = get_dataloader(
        args, normalizer=args.normalizer,
        tod=args.tod, dow=False, single=False,
    )

    model, input_dim = load_best_model(device)

    all_pred = []
    all_true = []

    with torch.no_grad():
        for data, target in test_loader:
            data  = data.to(device)
            label = target[..., :1].to(device)           # (B, 144, N, 1)
            out   = model(data)                           # (B, 144, N)
            if args.real_value:
                out_raw   = scaler.inverse_transform(out)
                label_raw = scaler.inverse_transform(label)
            else:
                out_raw   = scaler.inverse_transform(out)
                label_raw = scaler.inverse_transform(label)
            all_pred.append(out_raw.cpu().numpy())
            all_true.append(label_raw.squeeze(-1).cpu().numpy())

    # (samples, 144, 34)
    pred = np.concatenate(all_pred, axis=0)
    true = np.concatenate(all_true, axis=0)

    print(f"\nTest ornekleri: {pred.shape[0]}")
    print(f"Tahmin sekli: {pred.shape}  (ornekler × 144 slot × 34 node)")
    return pred, true, scaler

# ─── Metrik hesapla ──────────────────────────────────────────────────────────

def compute_metrics(pred, true):
    """
    Dondurur:
      overall  — tum test seti icin MAE/RMSE/MAPE
      per_node — her node icin MAE/RMSE/MAPE
      per_hour — her saat dilimi icin MAE (0-23)
    """
    mask  = true > 0.5   # sifir/dusuk trafik noktalarini maskeliyoruz

    def mae(p, t):   return float(np.abs(p - t).mean())
    def rmse(p, t):  return float(np.sqrt(((p - t)**2).mean()))
    def mape(p, t, m): return float((np.abs(p[m] - t[m]) / (t[m] + 1e-6)).mean() * 100)

    overall = {
        "MAE":  mae(pred, true),
        "RMSE": rmse(pred, true),
        "MAPE": mape(pred, true, mask),
    }

    # Node bazinda (34 node)
    per_node = {}
    for node_idx in range(34):
        p = pred[:, :, node_idx]
        t = true[:, :, node_idx]
        m = t > 0.5
        per_node[node_idx] = {
            "MAE":  mae(p, t),
            "RMSE": rmse(p, t),
            "MAPE": mape(p, t, m),
        }

    # Saat dilimi bazinda (24 saat, her biri 6 slot)
    per_hour = {}
    for h in range(24):
        slots = list(range(h*6, h*6+6))
        p = pred[:, slots, :]
        t = true[:, slots, :]
        m = t > 0.5
        per_hour[h] = {
            "MAE":  mae(p, t),
            "RMSE": rmse(p, t),
            "MAPE": mape(p, t, m),
        }

    return overall, per_node, per_hour

# ─── Gorsellestir ────────────────────────────────────────────────────────────

def plot_results(pred, true, overall, per_hour):
    """
    3 grafik:
      1. Ornek bir gun gercek vs tahmin (Gesi kavşagi kol A)
      2. Saat bazinda MAE (hangi saatte hata yuksek?)
      3. Tum kavşaklar MAE karsilastirmasi
    """
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    hours = list(range(24))
    hour_labels = [f"{h:02d}:00" for h in hours]
    hour_maes = [per_hour[h]["MAE"] for h in hours]

    # ── 1. Ornek gun: Gesi A ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    sample_idx = 0
    node_idx   = NODE_MAP[89]["A"]        # Gesi kavşagi A kolu

    p_day = pred[sample_idx, :, node_idx]
    t_day = true[sample_idx, :, node_idx]
    x     = range(144)

    ax1.fill_between(x, t_day, alpha=0.15, color="#2563eb", label="_nolegend_")
    ax1.plot(x, t_day, color="#2563eb", linewidth=2.5, label="Gercek")
    ax1.plot(x, p_day, color="#f59e0b", linewidth=2.5,
             linestyle="--", label="Garnoldi Tahmini")
    ax1.axvspan(42, 54,  alpha=0.08, color="red",   label="Sabah Piki (07-09)")
    ax1.axvspan(102, 114, alpha=0.08, color="green", label="Aksam Piki (17-19)")
    ax1.set_xticks(range(0, 144, 12))
    ax1.set_xticklabels([SLOT_LABELS[i] for i in range(0, 144, 12)], rotation=30)
    ax1.set_title("Gesi Kavsagi — Kol A: Gercek vs Tahmin (ornek test gunu)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Arac Sayisi")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    sample_mae = float(np.abs(p_day - t_day).mean())
    ax1.text(0.02, 0.96, f"MAE = {sample_mae:.1f} arac", transform=ax1.transAxes,
             fontsize=10, color="#374151",
             verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#e5e7eb"))

    # ── 2. Saat bazinda MAE ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    bar_colors = [
        "#ef4444" if (7 <= h <= 9 or 17 <= h <= 19) else
        "#f59e0b" if (12 <= h <= 13) else "#60a5fa"
        for h in hours
    ]
    ax2.bar(hours, hour_maes, color=bar_colors, edgecolor="white", width=0.7)
    ax2.set_xticks(hours[::2])
    ax2.set_xticklabels(hour_labels[::2], rotation=45, fontsize=8)
    ax2.set_title("Saat Bazinda MAE\n(kirmizi = pik saatler)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("MAE (arac)")
    ax2.grid(axis="y", alpha=0.3)

    # ── 3. Kavsak bazinda MAE ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    junc_maes  = []
    junc_names = []
    for jid, arm_map in NODE_MAP.items():
        nodes = list(arm_map.values())
        p_j = pred[:, :, nodes]
        t_j = true[:, :, nodes]
        junc_maes.append(float(np.abs(p_j - t_j).mean()))
        junc_names.append(JUNCTION_NAMES.get(jid, str(jid)))

    colors = ["#ef4444" if m == max(junc_maes) else
              "#22c55e" if m == min(junc_maes) else "#60a5fa"
              for m in junc_maes]
    ax3.barh(junc_names, junc_maes, color=colors, edgecolor="white")
    ax3.set_title("Kavsak Bazinda MAE\n(kirmizi = en zor, yesil = en kolay)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("MAE (arac)")
    ax3.grid(axis="x", alpha=0.3)

    # ── Baslik ────────────────────────────────────────────────────────────────
    overall_text = (
        f"Genel Test Sonuclari:  "
        f"MAE = {overall['MAE']:.2f} arac  |  "
        f"RMSE = {overall['RMSE']:.2f}  |  "
        f"MAPE = {overall['MAPE']:.1f}%"
    )
    fig.suptitle(
        f"Garnoldi Seq2Seq — Tahmin Degerlendirmesi\n{overall_text}",
        fontsize=12, fontweight="bold", y=1.01,
    )

    out_path = PROJECT_ROOT / "evaluation_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nGrafik kaydedildi: {out_path}")
    plt.close()

# ─── CSV kaydet ──────────────────────────────────────────────────────────────

def save_metrics_csv(overall, per_node, per_hour):
    import csv

    path = PROJECT_ROOT / "evaluation_metrics.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        w.writerow(["=== GENEL ==="])
        w.writerow(["MAE", "RMSE", "MAPE (%)"])
        w.writerow([f"{overall['MAE']:.3f}", f"{overall['RMSE']:.3f}", f"{overall['MAPE']:.2f}"])

        w.writerow([])
        w.writerow(["=== SAAT BAZINDA MAE ==="])
        w.writerow(["Saat", "MAE"])
        for h in range(24):
            w.writerow([f"{h:02d}:00", f"{per_hour[h]['MAE']:.3f}"])

        w.writerow([])
        w.writerow(["=== NODE BAZINDA ==="])
        w.writerow(["Node", "Kavsak", "Kol", "MAE", "RMSE", "MAPE (%)"])
        for jid, arm_map in NODE_MAP.items():
            jname = JUNCTION_NAMES.get(jid, str(jid))
            for arm, nidx in arm_map.items():
                m = per_node[nidx]
                w.writerow([nidx, jname, arm,
                            f"{m['MAE']:.3f}", f"{m['RMSE']:.3f}", f"{m['MAPE']:.2f}"])

    print(f"Metrikler kaydedildi: {path}")

# ─── Ana akis ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Garnoldi Seq2Seq — Tahmin Degerlendirmesi")
    print("=" * 60)

    pred, true, scaler = run_evaluation()
    overall, per_node, per_hour = compute_metrics(pred, true)

    print(f"\n{'='*40}")
    print(f"  MAE  : {overall['MAE']:.2f} arac/10dk")
    print(f"  RMSE : {overall['RMSE']:.2f}")
    print(f"  MAPE : {overall['MAPE']:.1f}%")
    print(f"{'='*40}")
    print("\nSaat bazinda en zor zamanlar:")
    sorted_hours = sorted(per_hour, key=lambda h: per_hour[h]["MAE"], reverse=True)
    for h in sorted_hours[:5]:
        print(f"  {h:02d}:00  MAE = {per_hour[h]['MAE']:.2f}")
    print("\nEn zor kavsak:")
    worst_node = max(per_node, key=lambda n: per_node[n]["MAE"])
    for jid, arm_map in NODE_MAP.items():
        if worst_node in arm_map.values():
            arm = [a for a, i in arm_map.items() if i == worst_node][0]
            print(f"  {JUNCTION_NAMES[jid]} kol {arm} — MAE = {per_node[worst_node]['MAE']:.2f}")

    plot_results(pred, true, overall, per_hour)
    save_metrics_csv(overall, per_node, per_hour)

    print("\nBitirdi.")
