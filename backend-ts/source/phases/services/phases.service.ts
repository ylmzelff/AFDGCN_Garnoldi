/**
 * Phases Service
 * ==============
 * İldem, Tuna ve diğer bölgeler için faz yanıtları oluşturur.
 * Python phases.py mantığının TypeScript karşılığı.
 */

import type { KayseriClientService } from '../../predict/services/kayseri-client.service'
import type { PythonModelService } from '../../predict/services/python-model.service'
import type { PhaseCalculatorService } from './phase-calculator.service'
import { JUNCTION_NAMES } from './phase-calculator.service'
import { REGION_CONFIG } from '../../predict/services/real-time-predictor.service'
import type { ArmPhase, JunctionPhase, RegionPhaseResponse } from '../types/phases.types'

// ─────────────────────────────────────────────────────────────────────────────
// Sabitler — kavşak kol adları (görüntüleme için)
// ─────────────────────────────────────────────────────────────────────────────

const ARM_NAMES: Record<number, Record<string, string>> = {
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
// Yardımcı Fonksiyonlar
// ─────────────────────────────────────────────────────────────────────────────

function minuteIndexNow(): number {
  const now = new Date()
  return now.getHours() * 6 + Math.floor(now.getMinutes() / 10)
}

function timeLabelNow(): string {
  const now = new Date()
  const slotMinute = Math.floor(now.getMinutes() / 10) * 10
  return `${String(now.getHours()).padStart(2, '0')}:${String(slotMinute).padStart(2, '0')}`
}

// ─────────────────────────────────────────────────────────────────────────────
// Service
// ─────────────────────────────────────────────────────────────────────────────

export class PhasesService {
  constructor(
    private readonly kayseriClient: KayseriClientService,
    private readonly pythonModel: PythonModelService,
    private readonly calculator: PhaseCalculatorService,
  ) {}

  /** Moving average fallback using current + previous time slot. */
  private movingAvgFallback(
    dataByJunction: Record<number, Array<Record<string, string | number>>>,
    minuteIdx: number,
  ): Record<number, Record<string, number>> {
    const result: Record<number, Record<string, number>> = {}
    for (const [jidStr, arms] of Object.entries(dataByJunction)) {
      const armCounts: Record<string, number> = {}
      for (const armData of arms) {
        const direction = String(armData['edge_direction'] ?? '').trim().toUpperCase()
        if (direction) {
          const cur = Number(armData[String(minuteIdx)] ?? 0)
          const prev = Number(armData[String(Math.max(0, minuteIdx - 1))] ?? 0)
          armCounts[direction] = Math.max(0, (cur + prev) / 2)
        }
      }
      if (Object.keys(armCounts).length > 0) {
        result[Number(jidStr)] = armCounts
      }
    }
    return result
  }

  /** Builds the junction list for a region phase response. */
  private buildJunctionList(
    regionPhases: Record<number, ReturnType<PhaseCalculatorService['computePhases']>>,
    region: string,
  ): JunctionPhase[] {
    const result: JunctionPhase[] = []

    for (const [jidStr, phaseData] of Object.entries(regionPhases)) {
      const jid = Number(jidStr)
      const cycleTime = phaseData['_cycle_time'] as number
      const totalVehicles = phaseData['_total_vehicles'] as number
      const junctionName = JUNCTION_NAMES[jid] ?? `Kavşak ${jid}`
      const armNameMap = ARM_NAMES[jid] ?? {}

      const arms: ArmPhase[] = []

      for (const [key, value] of Object.entries(phaseData)) {
        if (key.startsWith('_') || typeof value !== 'object') continue

        const armData = value as {
          green: number; yellow: number; red: number; cycle_time: number
          vehicle_count: number; lanes: number; load: number; status: 'low' | 'medium' | 'high'
        }

        arms.push({
          arm: key,
          armName: armNameMap[key] ?? `Kol ${key}`,
          vehicleCount: armData.vehicle_count,
          lanes: armData.lanes,
          load: armData.load,
          status: armData.status,
          green: armData.green,
          yellow: armData.yellow,
          red: armData.red,
          cycleTime: armData.cycle_time,
        })
      }

      arms.sort((a, b) => a.arm.localeCompare(b.arm))

      result.push({ junctionId: jid, junctionName, cycleTime, totalVehicles, arms })
    }

    result.sort((a, b) => a.junctionId - b.junctionId)
    return result
  }

  // ────────────────────────────────────────────────────────────────────────────

  async buildRegionPhases(region: string): Promise<RegionPhaseResponse> {
    const config = REGION_CONFIG[region]
    if (!config) throw new Error(`Bilinmeyen bölge: ${region}`)

    const minuteIdx = minuteIndexNow()
    let kayseriOk = true
    let dataByJunction: Record<number, Array<Record<string, string | number>>> | null = null

    try {
      dataByJunction = await this.kayseriClient.fetchRegion(region)
    } catch (err) {
      console.warn(`[phases] Kayseri API ulaşılamıyor (${region}):`, err)
      kayseriOk = false
    }

    let predictions: Record<number, Record<string, number>>
    let predictionSource: 'AFDGCN' | 'moving_average' = 'moving_average'

    if (dataByJunction && config.useModel) {
      const modelResult = await this.pythonModel.predictNextTimestep(dataByJunction, minuteIdx)
      if (modelResult) {
        predictions = modelResult
        predictionSource = 'AFDGCN'
      } else {
        predictions = this.movingAvgFallback(dataByJunction, minuteIdx)
      }
    } else if (dataByJunction) {
      predictions = this.movingAvgFallback(dataByJunction, minuteIdx)
    } else {
      predictions = Object.fromEntries(config.junctionIds.map((jid) => [jid, {}]))
    }

    const regionPhases = this.calculator.computeRegionPhases(predictions, region)
    const junctions = this.buildJunctionList(regionPhases, region)

    return {
      region,
      city: config.city,
      timestamp: new Date().toISOString(),
      timeLabel: timeLabelNow(),
      predictionSource,
      kayseriApiStatus: kayseriOk ? 'connected' : 'unavailable',
      junctions,
    }
  }

  async buildIldemPhases(): Promise<RegionPhaseResponse> {
    return this.buildRegionPhases('ildem')
  }

  async buildTunaPhases(): Promise<RegionPhaseResponse> {
    return this.buildRegionPhases('tuna')
  }

  /** Tüm aktif bölgeleri WebSocket istemcilerine yayınlar. */
  async pushAllRegions(broadcast: (payload: object) => void): Promise<void> {
    try {
      const entries = await Promise.all(
        Object.keys(REGION_CONFIG).map(async (region) => {
          const resp = await this.buildRegionPhases(region)
          return [region, resp] as const
        }),
      )
      broadcast({
        type: 'phase_update',
        regions: Object.fromEntries(entries),
      })
    } catch (err) {
      console.error('[phases] Broadcast hatası:', err)
    }
  }
}
