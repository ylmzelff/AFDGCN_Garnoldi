/**
 * Trafik Işığı Faz Hesaplayıcı
 * ================================
 * Webster benzeri orantılı yeşil dağıtımı.
 * Python phase_calculator.py'nin TypeScript karşılığı.
 */

import type { ArmPhase, PhaseSeriesItem } from '../types/phases.types'

// ─────────────────────────────────────────────────────────────────────────────
// Sabitler
// ─────────────────────────────────────────────────────────────────────────────

const FIXED_YELLOW = 3
const FIXED_PROTECTION = 6
const MIN_GREEN = 10
const CYCLE_MIN = 60
const CYCLE_MAX = 120

const THRESH_LOW = 10.0
const THRESH_HIGH = 50.0

// ─────────────────────────────────────────────────────────────────────────────
// Şerit Konfigürasyonları
// ─────────────────────────────────────────────────────────────────────────────

export const ILDEM_LANE_CONFIG: Record<number, Record<string, number>> = {
  89:  { A: 4, B: 3, C: 3, D: 1 },
  187: { A: 2, B: 2, C: 2, D: 2 },
  95:  { A: 3, B: 3, C: 3, D: 2 },
  121: { A: 2, B: 2, C: 2, D: 2 },
  184: { A: 2, B: 2,        D: 3 },
  188: { A: 3, B: 3, C: 2, D: 1 },
  117: { A: 3,        C: 3, D: 2 },
  192: { A: 2, B: 1, C: 2, D: 2 },
  194: { A: 3, B: 2, C: 3, D: 2 },
}

// Bölge → kavşak → kol → şerit sayısı.  Yeni şehirler/bölgeler buraya eklenir.
export const REGION_LANE_CONFIGS: Record<string, Record<number, Record<string, number>>> = {
  ildem: ILDEM_LANE_CONFIG,
}

const DEFAULT_LANES = 2

export const JUNCTION_NAMES: Record<number, string> = {
  89: 'Gesi', 187: 'Serkent', 95: 'Beyazşehir', 121: 'Toki',
  184: 'İldem 1', 188: 'İldem 2', 117: 'İldem 3',
  192: 'İldem 4', 194: 'İldem 5',
  5: 'Tuna 5', 3: 'Tuna 3', 87: 'Tuna 87',
  25: 'Tuna 25', 26: 'Tuna 26', 27: 'Tuna 27', 7: 'Tuna 7',
}

/** Kavşak kol görüntüleme adları — bölge/şehir bağımsız ortak kayıt */
export const ARM_DISPLAY_NAMES: Record<number, Record<string, string>> = {
  89:  { A: 'SİVAS BULVARI-SİVAS YÖNÜ', B: 'GESİ CAD.', C: 'SİVAS BULV-ŞEHİR MERKEZİ', D: '381. SOKAK' },
  187: { A: '822. SK', B: 'GESİ CAD. DOĞU', C: 'KOCASİNAN CAD.', D: 'GESİ CAD. BATI' },
  95:  { A: 'OSMAN ÖZCAN CAD.', B: 'GESİ CAD DOĞU', C: 'MARKETLER ÇIKIŞ', D: 'GESİ CAD BATI' },
  121: { A: '832. CD', B: 'GESİ CAD. DOĞU', C: 'KADİR HAS BUL.', D: 'GESİ CAD BATI' },
  184: { A: 'DİNÇER SOKAK', B: 'ALPARSLAN TÜRKEŞ BUL.', D: 'GESİ CAD' },
  188: { A: 'DİNÇER SOKAK KUZEY', B: 'HANEDAN SOKAK', C: 'DİNÇER SOKAK GÜNEY', D: 'FETİH SOKAK' },
  117: { A: 'DÜNDAR TAŞER CAD. KUZEY', C: 'DÜNDAR TAŞER CAD. GÜNEY', D: 'HANEDAN SOKAK' },
  192: { A: 'YAVUZ SULTAN SELİM CAD. KUZEY', B: 'VATAN SOKAK', C: 'YAVUZ SULTAN SELİM CAD. GÜNEY', D: 'ORKUN SOKAK' },
  194: { A: 'S.A BEDÜK CAD. KUZEY', B: 'VATAN SOKAK BATI', C: 'S.A BEDÜK CAD. GÜNEY', D: 'VATAN SOKAK DOĞU' },
}

// ─────────────────────────────────────────────────────────────────────────────
// Tek Kavşak Faz Hesabı
// ─────────────────────────────────────────────────────────────────────────────

export interface ArmPhaseData {
  arm_name: string
  green: number
  yellow: number
  red: number
  protection: number
  cycle_time: number
  vehicle_count: number
  lanes: number
  load: number
  status: 'low' | 'medium' | 'high'
}

export interface JunctionPhaseData {
  _cycle_time: number
  _total_vehicles: number
  [arm: string]: ArmPhaseData | number
}

export class PhaseCalculatorService {
  computePhases(
    junctionId: number,
    armCounts: Record<string, number>,
    region: string = 'ildem',
  ): JunctionPhaseData {
    const baseCfg: Record<string, number> =
      (REGION_LANE_CONFIGS[region] ?? {})[junctionId] ?? {}
    const armDisplayNames = ARM_DISPLAY_NAMES[junctionId] ?? {}

    const laneCfg: Record<string, number> = {}
    for (const arm of Object.keys(armCounts)) {
      laneCfg[arm] = baseCfg[arm] ?? DEFAULT_LANES
    }

    const phases: Array<{
      arm: string
      count: number
      lanes: number
      load: number
      green: number
    }> = []

    for (const [arm, count] of Object.entries(armCounts)) {
      const lanes = laneCfg[arm] ?? DEFAULT_LANES
      const load = count / Math.max(lanes, 1)
      phases.push({ arm, count, lanes, load, green: 0 })
    }

    if (phases.length === 0) {
      return { _cycle_time: CYCLE_MIN, _total_vehicles: 0 }
    }

    const totalLoad = phases.reduce((s, p) => s + p.load, 0)
    const totalVehs = phases.reduce((s, p) => s + p.count, 0)
    const n = phases.length

    const loadFactor = Math.min(Math.max((totalLoad - 50.0) / (1500.0 - 50.0), 0), 1)
    const cycleTime = Math.round(CYCLE_MIN + loadFactor * (CYCLE_MAX - CYCLE_MIN))

    const totalLoss = n * (FIXED_YELLOW + FIXED_PROTECTION)
    const netGreenPool = Math.max(cycleTime - totalLoss, n * MIN_GREEN)

    for (const p of phases) {
      const share = totalLoad > 0 ? p.load / totalLoad : 1 / n
      p.green = Math.max(MIN_GREEN, Math.round(share * netGreenPool))
    }

    // Correction to ensure sum matches pool
    const greenSum = phases.reduce((s, p) => s + p.green, 0)
    const diff = netGreenPool - greenSum
    if (diff !== 0) {
      const maxPhase = phases.reduce((a, b) => (b.load > a.load ? b : a))
      maxPhase.green += diff
    }

    const result: JunctionPhaseData = {
      _cycle_time: cycleTime,
      _total_vehicles: Math.round(totalVehs),
    }

    for (const p of phases) {
      const green = Math.max(0, p.green)
      const yellow = FIXED_YELLOW
      const red = Math.max(0, cycleTime - green - yellow - FIXED_PROTECTION)
      const status: 'low' | 'medium' | 'high' =
        p.count < THRESH_LOW ? 'low' : p.count < THRESH_HIGH ? 'medium' : 'high'

      result[p.arm] = {
        arm_name: armDisplayNames[p.arm] ?? `Kol ${p.arm}`,
        green,
        yellow,
        red,
        protection: FIXED_PROTECTION,
        cycle_time: cycleTime,
        vehicle_count: Math.round(p.count),
        lanes: p.lanes,
        load: Math.round(p.load * 100) / 100,
        status,
      } satisfies ArmPhaseData
    }

    return result
  }

  computeRegionPhases(
    regionPredictions: Record<number, Record<string, number>>,
    region: string = 'ildem',
  ): Record<number, JunctionPhaseData> {
    const result: Record<number, JunctionPhaseData> = {}
    for (const [jidStr, armCounts] of Object.entries(regionPredictions)) {
      result[Number(jidStr)] = this.computePhases(Number(jidStr), armCounts, region)
    }
    return result
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Faz Serisi — tahmin edilen araç sayıları → her 10 dk'lık slot için faz
  // ─────────────────────────────────────────────────────────────────────────

  private slotToLabel(slot: number, slotMinutes: number = 10): string {
    const totalMinutes = slot * slotMinutes
    const hh = Math.floor(totalMinutes / 60)
    const mm = totalMinutes % 60
    return `${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')}`
  }

  /**
   * Her slot için tahmin edilen araç sayılarını Webster faz hesaplayıcısına
   * besler ve her slot × kavşak için faz önerisi döner.
   *
   * @param predictionSeries  junctionId → kol → [slot0_araç, slot1_araç, …]
   * @param region            Bölge adı (şerit konfigürasyonu için)
   * @param armNames          Opsiyonel kol görüntü adları (kavşak → kol → isim)
   * @param slotMinutes       Her slot'un dakika uzunluğu (Kayseri: 10, Sivas: 60)
   * @returns                 junctionId → [PhaseSeriesItem per slot]
   */
  computePhaseSeries(
    predictionSeries: Record<number, Record<string, number[]>>,
    region: string = 'ildem',
    armNames: Record<number, Record<string, string>> = {},
    slotMinutes: number = 10,
  ): Record<number, PhaseSeriesItem[]> {
    const result: Record<number, PhaseSeriesItem[]> = {}

    for (const [jidStr, armSeries] of Object.entries(predictionSeries)) {
      const jid = Number(jidStr)
      const armKeys = Object.keys(armSeries)
      const slotCount = armKeys.length > 0 ? (armSeries[armKeys[0]]?.length ?? 0) : 0
      const junctionArmNames = armNames[jid] ?? {}
      const slots: PhaseSeriesItem[] = []

      for (let i = 0; i < slotCount; i++) {
        // Bu slottaki her kolun tahmin edilen araç sayısı
        const armCounts: Record<string, number> = {}
        for (const arm of armKeys) {
          armCounts[arm] = armSeries[arm]?.[i] ?? 0
        }

        const phaseData = this.computePhases(jid, armCounts, region)
        const cycleTime = phaseData['_cycle_time'] as number
        const totalVehicles = phaseData['_total_vehicles'] as number

        const arms: ArmPhase[] = []
        for (const [key, value] of Object.entries(phaseData)) {
          if (key.startsWith('_') || typeof value !== 'object') continue
          const ad = value as ArmPhaseData
          arms.push({
            arm: key,
            armName: junctionArmNames[key] ?? `Kol ${key}`,
            vehicleCount: ad.vehicle_count,
            lanes: ad.lanes,
            load: ad.load,
            status: ad.status,
            green: ad.green,
            yellow: ad.yellow,
            red: ad.red,
            cycleTime: ad.cycle_time,
          })
        }
        arms.sort((a, b) => a.arm.localeCompare(b.arm))

        slots.push({
          slot_index: i,
          time_label: this.slotToLabel(i, slotMinutes),
          cycle_time: cycleTime,
          total_vehicles: totalVehicles,
          arms,
        })
      }

      result[jid] = slots
    }

    return result
  }
}
