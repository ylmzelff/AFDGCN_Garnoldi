/**
 * Phases Service
 * ==============
 * İldem, Tuna ve diğer bölgeler için faz yanıtları oluşturur.
 * Python phases.py mantığının TypeScript karşılığı.
 */

import type { TrafficClient } from '../../predict/services/traffic-client.interface'
import type { PythonModelService } from '../../predict/services/python-model.service'
import type { PhaseCalculatorService } from './phase-calculator.service'
import { JUNCTION_NAMES, ARM_DISPLAY_NAMES } from './phase-calculator.service'
import { REGION_CONFIG } from '../../predict/services/real-time-predictor.service'
import type { ArmPhase, JunctionPhase, PhaseSeriesItem, RegionPhaseResponse } from '../types/phases.types'

// ─────────────────────────────────────────────────────────────────────────────
// Sabitler — kavşak kol adları (görüntüleme için)
// ARM_DISPLAY_NAMES artık phase-calculator.service.ts'den export edilmektedir.
// ─────────────────────────────────────────────────────────────────────────────
const ARM_NAMES = ARM_DISPLAY_NAMES

// ─────────────────────────────────────────────────────────────────────────────
// Yardımcı Fonksiyonlar
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Belediye API'leri saat anahtarlarını ("0".."23") Türkiye yerel saatiyle döner.
 * Sunucu farklı bir saat diliminde (ör. HF Spaces container'ı UTC) çalışabildiği
 * için `Date.getHours()` yerine her zaman Europe/Istanbul'a göre hesaplanır.
 */
function nowInIstanbul(): { hours: number; minutes: number } {
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: 'Europe/Istanbul',
    hour: 'numeric',
    minute: 'numeric',
    hourCycle: 'h23',
  }).formatToParts(new Date())
  const hours = Number(parts.find((p) => p.type === 'hour')?.value ?? 0)
  const minutes = Number(parts.find((p) => p.type === 'minute')?.value ?? 0)
  return { hours, minutes }
}

function minuteIndexNow(slotMinutes: number): number {
  const { hours, minutes } = nowInIstanbul()
  return hours * (60 / slotMinutes) + Math.floor(minutes / slotMinutes)
}

function timeLabelNow(slotMinutes: number): string {
  const { hours, minutes } = nowInIstanbul()
  const slotMinute = Math.floor(minutes / slotMinutes) * slotMinutes
  return `${String(hours).padStart(2, '0')}:${String(slotMinute).padStart(2, '0')}`
}

// ─────────────────────────────────────────────────────────────────────────────
// Service
// ─────────────────────────────────────────────────────────────────────────────

export class PhasesService {
  constructor(
    private readonly clients: Record<string, TrafficClient>,
    private readonly pythonModel: PythonModelService,
    private readonly calculator: PhaseCalculatorService,
  ) {}

  private clientFor(city: string): TrafficClient {
    const client = this.clients[city]
    if (!client) throw new Error(`[phases] '${city}' için tanımlı veri istemcisi yok.`)
    return client
  }

  /**
   * Geri dönük MA tahmini: her slot için yalnızca geçmiş slotlar kullanılır.
   * slot i tahmini = (actual[i-1] + actual[max(0, i-2)]) / 2
   * Veri sızıntısı yoktur.
   */
  private buildFallbackPredictionSeries(
    dataByJunction: Record<number, Array<Record<string, string | number>>>,
    completedIdx: number,
  ): Record<number, Record<string, number[]>> {
    const result: Record<number, Record<string, number[]>> = {}
    for (const [jidStr, arms] of Object.entries(dataByJunction)) {
      const armSeries: Record<string, number[]> = {}
      for (const armData of arms) {
        const direction = String(armData['edge_direction'] ?? '').trim().toUpperCase()
        if (!direction) continue
        const series: number[] = []
        for (let i = 0; i <= completedIdx; i++) {
          if (i === 0) {
            series.push(Math.max(0, Number(armData['0'] ?? 0)))
          } else {
            const prev1 = Number(armData[String(i - 1)] ?? 0)
            const prev2 = Number(armData[String(Math.max(0, i - 2))] ?? 0)
            series.push(Math.max(0, Math.round((prev1 + prev2) / 2)))
          }
        }
        armSeries[direction] = series
      }
      if (Object.keys(armSeries).length > 0) {
        result[Number(jidStr)] = armSeries
      }
    }
    return result
  }

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
          arm_name: string; green: number; yellow: number; red: number; cycle_time: number
          vehicle_count: number; lanes: number; load: number; status: 'low' | 'medium' | 'high'
        }

        arms.push({
          arm: key,
          armName: armData.arm_name ?? (armNameMap[key] ?? `Kol ${key}`),
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

    const client = this.clientFor(config.city)
    const slotMinutes = client.getSlotMinutes()
    const minuteIdx = minuteIndexNow(slotMinutes)
    let kayseriOk = true
    let dataByJunction: Record<number, Array<Record<string, string | number>>> | null = null

    try {
      dataByJunction = await client.fetchRegion(region)
    } catch (err) {
      console.warn(`[phases] '${config.city}' API ulaşılamıyor (${region}):`, err)
      kayseriOk = false
    }

    let predictions: Record<number, Record<string, number>>
    let predictionSource: 'AFDGCN' | 'moving_average' = 'moving_average'

    if (dataByJunction && config.useModel) {
      const modelResult = await this.pythonModel.predictNextTimestep(region, dataByJunction, minuteIdx)
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

    // ── Faz Serisi: gün başından mevcut dilime kadar her slot için faz önerisi ──
    let phaseSeries: Record<number, PhaseSeriesItem[]> | undefined

    if (dataByJunction) {
      const completedIdx = Math.max(0, minuteIdx - 1)
      let predictionSeries: Record<number, Record<string, number[]>> | null = null

      if (config.useModel) {
        const seriesResult = await this.pythonModel.predictSeries(region, dataByJunction, completedIdx)
        predictionSeries = seriesResult?.series ?? null
      }

      if (!predictionSeries) {
        // Fallback: backward-only MA (veri sızıntısı yok)
        predictionSeries = this.buildFallbackPredictionSeries(dataByJunction, completedIdx)
      }

      phaseSeries = this.calculator.computePhaseSeries(predictionSeries, region, ARM_NAMES, slotMinutes)
    }

    return {
      region,
      city: config.city,
      timestamp: new Date().toISOString(),
      timeLabel: timeLabelNow(slotMinutes),
      predictionSource,
      kayseriApiStatus: kayseriOk ? 'connected' : 'unavailable',
      junctions,
      phase_series: phaseSeries,
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
