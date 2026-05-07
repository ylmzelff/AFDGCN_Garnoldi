"""
Eğitilmiş modeli saved_models/ dizinine kopyalar ve doğrular.

Kullanım:
    python scripts/upload_model.py \\
        --weights-path  saved_models/model_epoch_100.pth \\
        --output-name   kayseri_ildem_34.pth \\
        --scaler-mean   28.53 \\
        --scaler-std    38.72

    (--output-name verilmezse --weights-path yerinde bırakılır, sadece doğrulanır)

Yapılan işlemler:
    1. .pth dosyasını torch.load ile doğrular
    2. node_embedding boyutunu raporlar
    3. İstenen hedef isimle saved_models/ içine kopyalar
    4. prediction_wrapper.py'nin beklediği isimlendirmeye göre kontrol eder
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import sys
from pathlib import Path

# Proje kökü
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch


# prediction_wrapper'ın aradığı birincil model adı
PRIMARY_MODEL_NAME = "kayseri_ildem_34.pth"


def validate_and_copy(weights_path: Path, output_name: str | None) -> None:
    if not weights_path.exists():
        sys.exit(f"[HATA] Dosya bulunamadı: {weights_path}")

    # .pth doğrula
    with open(weights_path, "rb") as f:
        raw_bytes = f.read()

    try:
        state_dict = torch.load(io.BytesIO(raw_bytes), map_location="cpu", weights_only=True)
        print(f"[OK] Model doğrulandı — {len(state_dict)} katman tensörü")
    except Exception as exc:
        sys.exit(f"[HATA] .pth dosyası okunamadı: {exc}")

    # Boyut bilgisi
    if "node_embedding" in state_dict:
        n_nodes, e_dim = state_dict["node_embedding"].shape
        print(f"[OK] node_embedding: {n_nodes} node × {e_dim} embed_dim")
        if n_nodes != 34:
            print(f"[UYARI] prediction_wrapper.py 34 node bekliyor, bu model {n_nodes} node içeriyor.")
    else:
        print("[UYARI] node_embedding anahtarı bulunamadı.")

    # Kopyala
    if output_name:
        dest = ROOT / "saved_models" / output_name
        dest.parent.mkdir(exist_ok=True)
        if dest.resolve() != weights_path.resolve():
            shutil.copy2(weights_path, dest)
            print(f"[OK] Kopyalandı → {dest}")
        else:
            print(f"[OK] Kaynak ve hedef aynı dosya, kopyalama atlandı.")

        if output_name == PRIMARY_MODEL_NAME:
            print(f"\n[OK] prediction_wrapper.py bu modeli otomatik yükleyecek.")
        else:
            print(f"\n[BILGI] prediction_wrapper.py birincil olarak '{PRIMARY_MODEL_NAME}' arar.")
            print(f"        Birincil model olarak kullanmak için:")
            print(f"        python scripts/upload_model.py --weights-path {weights_path} --output-name {PRIMARY_MODEL_NAME}")
    else:
        print(f"\n[BILGI] --output-name verilmedi. Dosya yerinde doğrulandı.")
        print(f"        Sistem tarafından kullanılabilmesi için hedef isim:")
        print(f"        saved_models/{PRIMARY_MODEL_NAME}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AFDGCN Model Doğrulayıcı & Kopyalayıcı")
    p.add_argument("--weights-path", required=True, help=".pth dosyası yolu")
    p.add_argument(
        "--output-name",
        default=None,
        help=f"Hedef dosya adı (varsayılan: yok — sadece doğrular). "
             f"Sistem tarafından kullanılmak için '{PRIMARY_MODEL_NAME}' verin.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate_and_copy(Path(args.weights_path), args.output_name)

