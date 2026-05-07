/**
 * Background Fetcher Service
 * ===========================
 * Tüm bölgeler için periyodik olarak tahmin çeker ve bellekte tutar.
 * Python background_fetcher.py'nin TypeScript karşılığı.
 */

import type { RealTimePredictorService, RegionPredictionResult } from './real-time-predictor.service'
import { REGION_CONFIG } from './real-time-predictor.service'
import { env } from '../../common/config'

export class BackgroundFetcherService {
  private timer: ReturnType<typeof setInterval> | null = null
  private latestPredictions = new Map<string, RegionPredictionResult>()
  private running = false

  constructor(private readonly predictor: RealTimePredictorService) {}

  start(intervalMs: number = env.bgFetchIntervalMs): void {
    if (this.running) return
    this.running = true
    console.info(`[background-fetcher] Başlatılıyor (interval=${intervalMs}ms)`)

    this.timer = setInterval(() => {
      void this.fetchAll()
    }, intervalMs)
  }

  stop(): void {
    if (this.timer) {
      clearInterval(this.timer)
      this.timer = null
    }
    this.running = false
    console.info('[background-fetcher] Durduruldu')
  }

  private async fetchAll(): Promise<void> {
    for (const region of Object.keys(REGION_CONFIG)) {
      try {
        const result = await this.predictor.predictRegion(region)
        this.latestPredictions.set(region, result)
        console.debug(`[background-fetcher] ${region}: ${Object.keys(result.predictions).length} kavşak`)
      } catch (err) {
        console.warn(`[background-fetcher] ${region} fetch hatası:`, err)
      }
    }
  }

  getLatest(region: string): RegionPredictionResult | null {
    return this.latestPredictions.get(region) ?? null
  }

  getAllLatest(): Record<string, RegionPredictionResult> {
    return Object.fromEntries(this.latestPredictions.entries())
  }

  getStatus(): object {
    return {
      running: this.running,
      regions_cached: Array.from(this.latestPredictions.keys()),
    }
  }
}
