/**
 * Real-Time Predictor Service
 * ============================
 * Belediye API'sından anlık veri çekip tahmin yapan ve cache'leyen servis.
 * Python real_time_predictor.py'nin TypeScript karşılığı.
 */

import type { TrafficClient, ArmData } from './traffic-client.interface'
import type { PythonModelService } from './python-model.service'
import type { PhaseCalculatorService } from '../../phases/services/phase-calculator.service'
import type { PhaseSeriesItem } from '../../phases/types/phases.types'

// ─────────────────────────────────────────────────────────────────────────────
// Bölge Konfigürasyonu (DB'den startup'ta yüklenir — başlangıçta boş)
// ─────────────────────────────────────────────────────────────────────────────

export interface RegionConfigItem {
  city: string
  junctionIds: number[]
  useModel: boolean
  description: string
  /** API bolgeAdi parametresi (örn. "İLDEM", "TUNA", "KIZILIRMAK") — DB'den gelir */
  bolgeAdi: string
}

export const REGION_CONFIG: Record<string, RegionConfigItem> = {}

/**
 * DB'den gelen bölge kayıtlarını in-memory registry'e yükler.
 * server.ts startup()'ta çağrılır.
 */
export function loadRegionConfigs(
  configs: Array<{ city: string; region: string; junctionIds: number[]; useModel: boolean; description: string; bolgeAdi: string }>,
): void {
  for (const c of configs) {
    REGION_CONFIG[c.region] = {
      city: c.city,
      junctionIds: c.junctionIds,
      useModel: c.useModel,
      description: c.description,
      bolgeAdi: c.bolgeAdi,
    }
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
  /** Gün başından bugüne tüm gerçek araç sayıları — grafik için */
  time_series?: Record<number, Record<string, number[]>>
  /** time_series dizisinin hangi slot index'ten başladığı (her zaman 0) */
  time_series_start_slot?: number
  /** Bu bölgenin veri kaynağının doğal zaman dilimi (dakika) — Kayseri: 10, Sivas: 60 */
  slot_minutes?: number
  /** Anlık gerçek araç sayısı — grafik güncelleme için */
  raw_data?: Record<number, Record<string, number>>
  /**
   * Her slot için bir önceki slottan üretilen MA tahmini + son slotta AFDGCN.
   * Frontend turuncu çizgi için kullanır. time_series ile aynı indeks yapısı.
   */
  prediction_series?: Record<number, Record<string, number[]>>
  /**
   * prediction_series'ten üretilen faz önerileri: her 10 dk'lık slot için
   * tahmin edilen araç sayıları Webster faz hesaplayıcısına beslenir.
   * junctionId → [PhaseSeriesItem per slot]
   */
  phase_series?: Record<number, PhaseSeriesItem[]>
}

// ─────────────────────────────────────────────────────────────────────────────
// Service
// ─────────────────────────────────────────────────────────────────────────────

export class RealTimePredictorService {
  private readonly cache = new PredictionCache(300_000)

  constructor(
    private readonly clients: Record<string, TrafficClient>,
    private readonly pythonModel: PythonModelService,
    private readonly phaseCalculator: PhaseCalculatorService,
  ) {}

  private clientFor(city: string): TrafficClient {
    const client = this.clients[city]
    if (!client) throw new Error(`[real-time-predictor] '${city}' için tanımlı veri istemcisi yok.`)
    return client
  }

  /** slotMinutes: bölgenin veri kaynağının doğal dilim uzunluğu (Kayseri: 10, Sivas: 60). */
  private minuteIndexNow(slotMinutes: number): number {
    const now = new Date()
    return now.getHours() * (60 / slotMinutes) + Math.floor(now.getMinutes() / slotMinutes)
  }

  private timeLabelNow(slotMinutes: number): string {
    const now = new Date()
    const slot = Math.floor(now.getMinutes() / slotMinutes) * slotMinutes
    return `${String(now.getHours()).padStart(2, '0')}:${String(slot).padStart(2, '0')}`
  }

  /** Gün başından (slot 0) bugünkü dilimine kadar gerçek araç sayılarını döner */
  private buildTimeSeries(
    dataByJunction: Record<number, ArmData[]>,
    currentIdx: number,
  ): Record<number, Record<string, number[]>> {
    const result: Record<number, Record<string, number[]>> = {}
    for (const [jidStr, arms] of Object.entries(dataByJunction)) {
      const armSeries: Record<string, number[]> = {}
      for (const armData of arms) {
        const direction = String(armData['edge_direction'] ?? '').trim().toUpperCase()
        if (!direction) continue
        const series: number[] = []
        for (let i = 0; i <= currentIdx; i++) {
          series.push(Number(armData[String(i)] ?? 0))
        }
        armSeries[direction] = series
      }
      if (Object.keys(armSeries).length > 0) {
        result[Number(jidStr)] = armSeries
      }
    }
    return result
  }

  /** Anlık (mevcut dilim) gerçek araç sayısını döner (grafik güncelleme için) */
  private buildRawData(
    dataByJunction: Record<number, ArmData[]>,
    currentIdx: number,
  ): Record<number, Record<string, number>> {
    const result: Record<number, Record<string, number>> = {}
    for (const [jidStr, arms] of Object.entries(dataByJunction)) {
      const armValues: Record<string, number> = {}
      for (const armData of arms) {
        const direction = String(armData['edge_direction'] ?? '').trim().toUpperCase()
        if (!direction) continue
        armValues[direction] = Number(armData[String(currentIdx)] ?? 0)
      }
      if (Object.keys(armValues).length > 0) {
        result[Number(jidStr)] = armValues
      }
    }
    return result
  }

  /**
   * TS-tarafı yedek tahmin serisi: yalnızca geçmiş veriler kullanılır, veri sızıntısı yok.
   * slot i tahmini = (actual[i-1] + actual[max(0, i-2)]) / 2
   * Model sunucusu erişilemez olduğunda veya model kullanılmayan bölgelerde devreye girer.
   */
  private buildFallbackPredictionSeries(
    dataByJunction: Record<number, ArmData[]>,
    completedIdx: number,
  ): Record<number, Record<string, number[]>> {
    const result: Record<number, Record<string, number[]>> = {}
    for (const [jidStr, arms] of Object.entries(dataByJunction)) {
      const jid = Number(jidStr)
      const armSeries: Record<string, number[]> = {}
      for (const armData of arms) {
        const direction = String(armData['edge_direction'] ?? '').trim().toUpperCase()
        if (!direction) continue
        const series: number[] = []
        for (let i = 0; i <= completedIdx; i++) {
          if (i === 0) {
            // İlk slot: geçmiş yok, gerçek değeri kullan
            series.push(Math.max(0, Number(armData['0'] ?? 0)))
          } else {
            // Yalnızca geçmiş slotlara bak (veri sızıntısı yok)
            const prev1 = Number(armData[String(i - 1)] ?? 0)
            const prev2 = Number(armData[String(Math.max(0, i - 2))] ?? 0)
            series.push(Math.max(0, Math.round((prev1 + prev2) / 2)))
          }
        }
        armSeries[direction] = series
      }
      if (Object.keys(armSeries).length > 0) {
        result[jid] = armSeries
      }
    }
    return result
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

    const client = this.clientFor(config.city)
    const slotMinutes = client.getSlotMinutes()
    const minuteIdx = this.minuteIndexNow(slotMinutes)
    let kayseriOk = true
    let dataByJunction: Record<number, ArmData[]> | null = null

    try {
      dataByJunction = await client.fetchRegion(region)
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
      time_label: this.timeLabelNow(slotMinutes),
      predictions,
      source,
      kayseri_ok: kayseriOk,
      phases,
      junction_count: Object.keys(predictions).length,
      slot_minutes: slotMinutes,
    }

    if (dataByJunction) {
      // currentIdx henüz sayılıyor (0 veya eksik) → son tamamlanan slot bir önceki
      const completedIdx = Math.max(0, minuteIdx - 1)
      result.time_series = this.buildTimeSeries(dataByJunction, completedIdx)
      result.time_series_start_slot = 0
      result.raw_data = this.buildRawData(dataByJunction, completedIdx)

      if (config.useModel) {
        // Model kullanan bölgeler: Python sunucusundan gerçek AFDGCN rolling tahminlerini al
        const seriesResult = await this.pythonModel.predictSeries(dataByJunction, completedIdx)
        if (seriesResult) {
          result.prediction_series = seriesResult.series
          // Kaynak etiketini model sunucusu yanıtıyla güncelle
          result.source = seriesResult.source
        } else {
          // Fallback: yalnızca geçmiş verilerle hesaplanan MA (veri sızıntısı yok)
          result.prediction_series = this.buildFallbackPredictionSeries(dataByJunction, completedIdx)
        }
      } else {
        // Model kullanılmayan bölgeler: TS-tarafı backward-looking MA
        result.prediction_series = this.buildFallbackPredictionSeries(dataByJunction, completedIdx)
      }

      // Tahmin serisi → Webster faz önerisi serisi
      if (result.prediction_series) {
        result.phase_series = this.phaseCalculator.computePhaseSeries(
          result.prediction_series,
          region,
          undefined,
          slotMinutes,
        )
      }
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

  /**
   * Hedef tarih icin tum gunu (144 slot) otoregresif tahmin eder.
   * targetDate: "YYYY-MM-DD" — tahmin edilecek gun
   * Tohum: bir onceki gunun gercek verisi Kayseri API'den cekilir.
   */
  async predictDayAhead(
    region: string,
    targetDate: string,
  ): Promise<{
    region: string
    date: string
    prediction_series: Record<number, Record<string, number[]>>
    slot_labels: string[]
    slot_minutes: number
    source: 'AFDGCN' | 'moving_average'
  }> {
    const config = REGION_CONFIG[region]
    if (!config) throw new Error(`Bilinmeyen bolge: ${region}`)

    const client = this.clientFor(config.city)
    const slotMinutes = client.getSlotMinutes()
    const slotsPerDay = 1440 / slotMinutes

    // Tohum: hedef tarihten 1 gun oncesi
    const targetMs = new Date(targetDate).getTime()
    const seedDate = new Date(targetMs - 86_400_000).toISOString().slice(0, 10)

    let seedData: Record<number, ArmData[]> | null = null
    try {
      seedData = await client.fetchRegionForDate(region, seedDate)
    } catch {
      console.warn(`[day-ahead] '${config.city}' API'den tohum veri alinamadi (${seedDate}), bos tohum kullanilacak`)
    }

    const slotLabels = Array.from({ length: slotsPerDay }, (_, i) => this.slotToLabel(i, slotMinutes))

    let predictionSeries: Record<number, Record<string, number[]>>
    let source: 'AFDGCN' | 'moving_average' = 'moving_average'

    if (seedData && config.useModel) {
      const result = await this.pythonModel.predictDayAhead(seedData, slotsPerDay - 1)
      if (result) {
        predictionSeries = result.series
        source = result.source
      } else {
        predictionSeries = this.buildFallbackDayAhead(config.junctionIds, slotsPerDay, slotMinutes)
      }
    } else {
      predictionSeries = this.buildFallbackDayAhead(config.junctionIds, slotsPerDay, slotMinutes, seedData ?? undefined)
    }

    return { region, date: targetDate, prediction_series: predictionSeries, slot_labels: slotLabels, slot_minutes: slotMinutes, source }
  }

  /** slot index → "HH:MM" dönüştürür (slotMinutes'e göre genelleştirilmiş). */
  private slotToLabel(slotIdx: number, slotMinutes: number): string {
    const totalMinutes = slotIdx * slotMinutes
    const hh = Math.floor(totalMinutes / 60)
    const mm = totalMinutes % 60
    return `${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')}`
  }

  /** Model yokken deterministik saatlik ortalama fallback uretir. */
  private buildFallbackDayAhead(
    junctionIds: number[],
    slotsPerDay: number,
    slotMinutes: number,
    seedData?: Record<number, ArmData[]>,
  ): Record<number, Record<string, number[]>> {
    const result: Record<number, Record<string, number[]>> = {}
    const PEAK_PATTERN = Array.from({ length: slotsPerDay }, (_, i) => {
      const h = Math.floor((i * slotMinutes) / 60)
      if (h >= 7 && h <= 9) return 1.8
      if (h >= 17 && h <= 19) return 1.5
      if (h >= 12 && h <= 13) return 1.2
      if (h < 5 || h >= 23) return 0.3
      return 0.8
    })
    const midSlot = Math.floor(slotsPerDay / 2)

    for (const jid of junctionIds) {
      const seedArms = seedData?.[jid]
      result[jid] = {}
      const arms = seedArms?.map((a) => String(a['edge_direction'] ?? '').toUpperCase()).filter(Boolean) ?? ['A', 'B', 'C', 'D']
      for (const arm of arms) {
        const seedArm = seedArms?.find((a) => String(a['edge_direction'] ?? '').toUpperCase() === arm)
        const baseVal = seedArm ? (Number(seedArm[String(midSlot)] ?? 20)) : 20
        result[jid][arm] = PEAK_PATTERN.map((factor) => Math.round(baseVal * factor))
      }
    }
    return result
  }

  async getCacheStatus(): Promise<object> {
    return this.cache.getStatus()
  }

  clearCache(): void {
    this.cache.clearExpired()
  }
}
