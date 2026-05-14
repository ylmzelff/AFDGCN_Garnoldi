"""
Webster Faz Hesaplayıcı — Unit Testleri
=========================================

Çalıştırma (proje kökünden):
    pytest tests/test_phase_calculator.py -v

Test senaryoları:
    1.  Sıfır araç        → cycle_min, tüm kollar MIN_GREEN alır
    2.  Tek kol           → tüm net yeşil havuzu o kola gider
    3.  Eşit araç         → yeşil süreler eşit dağılır
    4.  Yüksek trafik     → cycle_time CYCLE_MAX'a yaklaşır
    5.  Alçak trafik      → cycle_time CYCLE_MIN'de kalır
    6.  Durum sınıfları   → low / medium / high eşikleri doğru
    7.  Toplam süre denkliği → green + yellow + red + protection == cycle_time
    8.  Min yeşil güvencesi → hiçbir kol MIN_GREEN altına düşmez
    9.  İldem şerit config  → bilinen kavşak (89) doğru şerit sayılarını kullanır
    10. compute_region_phases → tüm junction ID'ler döner
    11. Tuna region → varsayılan 2 şerit kullanır
    12. Boş arm_counts  → sadece meta key'ler döner, çökmez
"""

from __future__ import annotations

import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.services.phase_calculator import (
    CYCLE_MAX,
    CYCLE_MIN,
    FIXED_PROTECTION,
    FIXED_YELLOW,
    ILDEM_LANE_CONFIG,
    MIN_GREEN,
    THRESH_HIGH,
    THRESH_LOW,
    TUNA_DEFAULT_LANES,
    compute_phases,
    compute_region_phases,
)


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcı
# ─────────────────────────────────────────────────────────────────────────────

def arm_data(result: dict, arm: str) -> dict:
    """Sonuç dict'inden tek kolun verisini döndürür."""
    return result[arm]


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Sıfır Araç
# ─────────────────────────────────────────────────────────────────────────────

def test_zero_vehicles_cycle_min():
    """Tüm kollar 0 araçsa cycle_time CYCLE_MIN olmalı."""
    result = compute_phases(89, {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0})
    assert result["_cycle_time"] == CYCLE_MIN


def test_zero_vehicles_min_green():
    """Tüm kollar 0 araçsa her kol en az MIN_GREEN almalı."""
    result = compute_phases(89, {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0})
    for arm in ("A", "B", "C", "D"):
        assert result[arm]["green"] >= MIN_GREEN, f"{arm} kolu MIN_GREEN altında"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Tek Kol
# ─────────────────────────────────────────────────────────────────────────────

def test_single_arm_gets_max_green():
    """Tek kollu kavşakta o kol tüm net yeşil havuzunu almalı."""
    result = compute_phases(999, {"A": 30.0})
    cycle = result["_cycle_time"]
    green = result["A"]["green"]
    assert green >= MIN_GREEN
    # green + yellow + protection <= cycle
    assert green + FIXED_YELLOW + FIXED_PROTECTION <= cycle + 1  # +1 int rounding


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Eşit Araç Sayısı → Eşit Yeşil
# ─────────────────────────────────────────────────────────────────────────────

def test_equal_vehicles_equal_green():
    """
    Eşit araç sayısı ve eşit şerit sayısında tüm kollar aynı yeşil süresini almalı.
    junction_id=187: her kol 2 şerit (eşit).
    """
    result = compute_phases(187, {"A": 20.0, "B": 20.0, "C": 20.0, "D": 20.0})
    greens = [result[arm]["green"] for arm in ("A", "B", "C", "D")]
    # Maksimum sapma 1 saniye olabilir (int yuvarlama)
    assert max(greens) - min(greens) <= 1


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Yüksek Trafik → Cycle Uzar
# ─────────────────────────────────────────────────────────────────────────────

def test_high_traffic_increases_cycle():
    """Çok fazla araç varsa cycle_time CYCLE_MIN'den büyük olmalı."""
    result = compute_phases(89, {"A": 500.0, "B": 400.0, "C": 300.0, "D": 200.0})
    assert result["_cycle_time"] > CYCLE_MIN


def test_very_high_traffic_caps_at_cycle_max():
    """Aşırı araç sayısında cycle_time CYCLE_MAX'ı aşmamalı."""
    result = compute_phases(89, {"A": 9999.0, "B": 9999.0, "C": 9999.0, "D": 9999.0})
    assert result["_cycle_time"] <= CYCLE_MAX


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Düşük Trafik → Cycle Min'de Kalır
# ─────────────────────────────────────────────────────────────────────────────

def test_low_traffic_stays_at_cycle_min():
    """Az araçla cycle_time CYCLE_MIN olmalı."""
    result = compute_phases(187, {"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0})
    assert result["_cycle_time"] == CYCLE_MIN


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Durum Sınıfları
# ─────────────────────────────────────────────────────────────────────────────

def test_status_low():
    """Araç sayısı < THRESH_LOW → status='low'."""
    result = compute_phases(187, {"A": THRESH_LOW - 1.0, "B": 1.0, "C": 1.0, "D": 1.0})
    assert result["A"]["status"] == "low"


def test_status_medium():
    """THRESH_LOW ≤ araç < THRESH_HIGH → status='medium'."""
    count = (THRESH_LOW + THRESH_HIGH) / 2
    result = compute_phases(187, {"A": count, "B": 1.0, "C": 1.0, "D": 1.0})
    assert result["A"]["status"] == "medium"


def test_status_high():
    """Araç sayısı ≥ THRESH_HIGH → status='high'."""
    result = compute_phases(187, {"A": THRESH_HIGH + 1.0, "B": 1.0, "C": 1.0, "D": 1.0})
    assert result["A"]["status"] == "high"


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Toplam Süre Denkliği
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("junction_id,arms", [
    (89,  {"A": 15.0, "B": 30.0, "C": 20.0, "D": 5.0}),
    (187, {"A": 50.0, "B": 50.0, "C": 50.0, "D": 50.0}),
    (194, {"A": 100.0, "B": 5.0, "C": 80.0, "D": 40.0}),
])
def test_phase_sum_equals_cycle(junction_id, arms):
    """
    Her kol için: green + yellow + red + protection == cycle_time.
    Trafik ışığının tam bir döngü (cycle) içinde kapanması gerekir.
    """
    result = compute_phases(junction_id, arms)
    cycle = result["_cycle_time"]
    for arm in arms:
        d = result[arm]
        total = d["green"] + d["yellow"] + d["red"] + d["protection"]
        assert total == cycle, (
            f"Kavşak {junction_id} kol {arm}: "
            f"{d['green']}+{d['yellow']}+{d['red']}+{d['protection']} = {total} ≠ {cycle}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: Min Yeşil Güvencesi
# ─────────────────────────────────────────────────────────────────────────────

def test_min_green_always_respected():
    """Hiçbir kol MIN_GREEN saniyenin altında yeşil alamaz."""
    result = compute_phases(89, {"A": 1000.0, "B": 0.1, "C": 0.1, "D": 0.1})
    for arm in ("A", "B", "C", "D"):
        assert result[arm]["green"] >= MIN_GREEN, (
            f"Kol {arm}: green={result[arm]['green']} < MIN_GREEN={MIN_GREEN}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: İldem Şerit Konfigürasyonu
# ─────────────────────────────────────────────────────────────────────────────

def test_ildem_junction_89_correct_lanes():
    """
    Kavşak 89 (Gesi) şerit konfigürasyonu: A=4, B=3, C=3, D=1.
    Aynı araç sayısı verildiğinde D kolu (1 şerit) daha yüksek load'a sahip olmalı.
    """
    result = compute_phases(89, {"A": 12.0, "B": 12.0, "C": 12.0, "D": 12.0})
    assert result["D"]["lanes"] == ILDEM_LANE_CONFIG[89]["D"]  # 1
    assert result["A"]["lanes"] == ILDEM_LANE_CONFIG[89]["A"]  # 4
    # D kolu (1 şerit) yük, A kolundan (4 şerit) büyük olmalı
    assert result["D"]["load"] > result["A"]["load"]


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: compute_region_phases
# ─────────────────────────────────────────────────────────────────────────────

def test_region_phases_returns_all_junctions():
    """compute_region_phases verilen tüm kavşak ID'leri için sonuç döndürmeli."""
    predictions = {
        89:  {"A": 10.0, "B": 8.0, "C": 6.0, "D": 4.0},
        187: {"A": 20.0, "B": 20.0, "C": 20.0, "D": 20.0},
        95:  {"A": 5.0,  "B": 5.0,  "C": 5.0,  "D": 5.0},
    }
    result = compute_region_phases(predictions, region="ildem")
    assert set(result.keys()) == {89, 187, 95}


def test_region_phases_has_meta_keys():
    """Her kavşak sonucunda _cycle_time ve _total_vehicles olmalı."""
    predictions = {121: {"A": 15.0, "B": 10.0, "C": 8.0, "D": 3.0}}
    result = compute_region_phases(predictions, region="ildem")
    assert "_cycle_time" in result[121]
    assert "_total_vehicles" in result[121]


# ─────────────────────────────────────────────────────────────────────────────
# Test 11: Tuna Region → Varsayılan Şerit
# ─────────────────────────────────────────────────────────────────────────────

def test_tuna_region_uses_default_lanes():
    """Tuna bölgesinde her kol TUNA_DEFAULT_LANES şerit kullanmalı."""
    result = compute_phases(5, {"A": 20.0, "B": 20.0, "C": 20.0, "D": 20.0}, region="tuna")
    for arm in ("A", "B", "C", "D"):
        assert result[arm]["lanes"] == TUNA_DEFAULT_LANES


# ─────────────────────────────────────────────────────────────────────────────
# Test 12: Boş arm_counts
# ─────────────────────────────────────────────────────────────────────────────

def test_empty_arm_counts_does_not_crash():
    """Boş arm_counts verildiğinde fonksiyon çökmemeli, sadece meta key'ler dönmeli."""
    result = compute_phases(89, {})
    assert "_cycle_time" in result
    assert "_total_vehicles" in result
    assert result["_total_vehicles"] == 0
