/**
 * Real-Time Predictor Service
 * ============================
 * Belediye API'sından anlık veri çekip tahmin yapan ve cache'leyen servis.
 * Python real_time_predictor.py'nin TypeScript karşılığı.
 */

import type { KayseriClientService, ArmData } from './kayseri-client.service'
import type { PythonModelService } from './python-model.service'
import type { PhaseCalculatorService } from '../../phases/services/phase-calculator.service'

// ─────────────────────────────────────────────────────────────────────────────
// Bölge Konfigürasyonu (DB'den startup'ta yüklenir — başlangıçta boş)
// ─────────────────────────────────────────────────────────────────────────────

export interface RegionConfigItem {
  city: string
  junctionIds: number[]
  useModel: boolean
  description: string
}

export const REGION_CONFIG: Record<string, RegionConfigItem> = {}

/**
 * DB'den gelen bölge kayıtlarını in-memory registry'e yükler.
 * server.ts startup()'ta çağrılır.
 */
export function loadRegionConfigs(
  configs: Array<{ city: string; region: string; junctionIds: number[]; useModel: boolean; description: string }>,
): void {
  for (const c of configs) {
    REGION_CONFIG[c.region] = { city: c.city, junctionIds: c.junctionIds, useModel: c.useModel, description: c.description }
  }
  console.info(`[region-config] ${configs.length} bölge yüklendi: ${configs.map((c) => c.region).join(', ')}`)
}

// ─────────────────────────────────────────────────────────────────────────────
// Prediction Cache
// ─────────────────────────────────────────────────────────────────────────────

class PredictionCache {
  private cache = new Map<string, { data: RegionPredictionResult; timestamp: number }>()

  constructor(private readonly ttlMs: number = 300_000) {}

  get(key: string): RegionPredictionResult | null {
    const entry = this.cache.get(key)
    if (!entry) return null
    if (Date.now() - entry.timestamp > this.ttlMs) {
      this.cache.delete(key)
      return null
    }
    return entry.data
  }

  set(key: string, data: RegionPredictionResult): void {
    this.cache.set(key, { data, timestamp: Date.now() })
  }

  clearExpired(): void {
    const now = Date.now()
    for (const [key, entry] of this.cache.entries()) {
      if (now - entry.timestamp > this.ttlMs) this.cache.delete(key)
    }
  }

  getStatus(): object {
    return {
      size: this.cache.size,
      ttl_ms: this.ttlMs,
      keys: Array.from(this.cache.keys()),
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export interface RegionPredictionResult {
  region: string
  timestamp: string
  time_label: string
  predictions: Record<number, Record<string, number>>
  source: 'AFDGCN' | 'moving_average'
  kayseri_ok: boolean
  phases: Record<number, unknown>
  junction_count: number
}

// ─────────────────────────────────────────────────────────────────────────────
// Service
// ─────────────────────────────────────────────────────────────────────────────

export class RealTimePredictorService {
  private readonly cache = new PredictionCache(300_000)

  constructor(
    private readonly kayseriClient: KayseriClientService,
    private readonly pythonModel: PythonModelService,
    private readonly phaseCalculator: PhaseCalculatorService,
  ) {}

  private minuteIndexNow(): number {
    const now = new Date()
    return now.getHours() * 6 + Math.floor(now.getMinutes() / 10)
  }

  private timeLabelNow(): string {
    const now = new Date()
    const slot = Math.floor(now.getMinutes() / 10) * 10
    return `${String(now.getHours()).padStart(2, '0')}:${String(slot).padStart(2, '0')}`
  }

  private movingAverageForRegion(
    dataByJunction: Record<number, ArmData[]>,
    minuteIdx: number,
  ): Record<number, Record<string, number>> {
    const result: Record<number, Record<string, number>> = {}
    for (const [jidStr, arms] of Object.entries(dataByJunction)) {
      const armCounts: Record<string, number> = {}
      for (const armData of arms) {
        const direction = String(armData['edge_direction'] ?? '').trim().toUpperCase()
        if (!direction) continue
        const cur = Number(armData[String(minuteIdx)] ?? 0)
        const prev = Number(armData[String(Math.max(0, minuteIdx - 1))] ?? 0)
        armCounts[direction] = Math.max(0, (cur + prev) / 2)
      }
      if (Object.keys(armCounts).length > 0) {
        result[Number(jidStr)] = armCounts
      }
    }
    return result
  }

  async predictRegion(region: string): Promise<RegionPredictionResult> {
    const config = REGION_CONFIG[region]
    if (!config) throw new Error(`Bilinmeyen bölge: ${region}`)

    const cached = this.cache.get(region)
    if (cached) return cached

    const minuteIdx = this.minuteIndexNow()
    let kayseriOk = true
    let dataByJunction: Record<number, ArmData[]> | null = null

    try {
      dataByJunction = await this.kayseriClient.fetchRegion(region)
    } catch {
      kayseriOk = false
    }

    let predictions: Record<number, Record<string, number>>
    let source: 'AFDGCN' | 'moving_average' = 'moving_average'

    if (dataByJunction && config.useModel) {
      const modelResult = await this.pythonModel.predictNextTimestep(dataByJunction, minuteIdx)
      if (modelResult) {
        predictions = modelResult
        source = 'AFDGCN'
      } else {
        predictions = this.movingAverageForRegion(dataByJunction, minuteIdx)
      }
    } else if (dataByJunction) {
      predictions = this.movingAverageForRegion(dataByJunction, minuteIdx)
    } else {
      predictions = Object.fromEntries(
        config.junctionIds.map((jid) => [jid, {}]),
      )
    }

    const phases = this.phaseCalculator.computeRegionPhases(predictions, region)

    const result: RegionPredictionResult = {
      region,
      timestamp: new Date().toISOString(),
      time_label: this.timeLabelNow(),
      predictions,
      source,
      kayseri_ok: kayseriOk,
      phases,
      junction_count: Object.keys(predictions).length,
    }

    this.cache.set(region, result)
    return result
  }

  async predictJunctionDetail(region: string, junctionId: number): Promise<object> {
    const config = REGION_CONFIG[region]
    if (!config) throw new Error(`Bilinmeyen bölge: ${region}`)
    if (!config.junctionIds.includes(junctionId)) {
      throw new Error(`Kavşak ${junctionId} '${region}' bölgesinde bulunamadı.`)
    }

    const regionResult = await this.predictRegion(region)
    const junctionPredictions = regionResult.predictions[junctionId] ?? {}
    const junctionPhases = regionResult.phases[junctionId] ?? {}

    return {
      junction_id: junctionId,
      region,
      timestamp: regionResult.timestamp,
      time_label: regionResult.time_label,
      source: regionResult.source,
      kayseri_ok: regionResult.kayseri_ok,
      predictions: junctionPredictions,
      phases: junctionPhases,
    }
  }

  async getCacheStatus(): Promise<object> {
    return this.cache.getStatus()
  }

  clearCache(): void {
    this.cache.clearExpired()
  }
}
