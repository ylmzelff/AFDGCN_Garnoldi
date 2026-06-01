/**
 * Python AFDGCN Model İstemcisi
 * ================================
 * PYTHON_MODEL_URL adresinde çalışan Python model sunucusunu çağırır.
 * Model sunucusu yoksa veya hata alırsa null döner ve caller moving average'a düşer.
 *
 * Python model sunucusu başlatmak için:
 *   python -m backend.model_server  (veya mevcut FastAPI backend port 9002'de)
 */

import axios, { type AxiosInstance } from 'axios'
import { env } from '../../common/config'
import type { ArmData } from './kayseri-client.service'

interface ModelPredictRequest {
  data_by_junction: Record<number, ArmData[]>
  minute_index: number
}

interface ModelPredictResponse {
  predictions: Record<number, Record<string, number>>
  source: 'AFDGCN' | 'moving_average'
}

interface ModelSeriesRequest {
  data_by_junction: Record<number, ArmData[]>
  completed_idx: number
}

interface ModelSeriesResponse {
  prediction_series: Record<number, Record<string, number[]>>
  source: 'AFDGCN' | 'moving_average'
}

export class PythonModelService {
  private http: AxiosInstance
  private available = true
  private lastCheckAt = 0
  private readonly CHECK_COOLDOWN_MS = 30_000

  constructor() {
    this.http = axios.create({
      baseURL: env.pythonModelUrl,
      timeout: 15_000,
      headers: { 'Content-Type': 'application/json' },
    })
  }

  /**
   * AFDGCN modelini çalıştırır.
   * @returns predictions dict veya null (model sunucusu yoksa)
   */
  async predictNextTimestep(
    dataByJunction: Record<number, ArmData[]>,
    minuteIndex: number,
  ): Promise<Record<number, Record<string, number>> | null> {
    // Bilinenin unavailable olduğunu düzenli aralıklarla yeniden dene
    if (!this.available && Date.now() - this.lastCheckAt < this.CHECK_COOLDOWN_MS) {
      return null
    }

    try {
      const payload: ModelPredictRequest = {
        data_by_junction: dataByJunction,
        minute_index: minuteIndex,
      }
      const resp = await this.http.post<ModelPredictResponse>('/predict/next', payload)
      this.available = true
      return resp.data.predictions
    } catch {
      this.available = false
      this.lastCheckAt = Date.now()
      console.warn('[python-model] /predict/next ulaşılamıyor → moving average devreye alınıyor')
      return null
    }
  }

  /**
   * Gün başından completedIdx dahil her slot için AFDGCN rolling tahmin serisi.
   * Her slotun tahmini yalnızca o slottan ÖNCEKİ gerçek verilerle üretilir.
   * @returns {series, source} veya null (model sunucusu yoksa)
   */
  async predictSeries(
    dataByJunction: Record<number, ArmData[]>,
    completedIdx: number,
  ): Promise<{ series: Record<number, Record<string, number[]>>; source: 'AFDGCN' | 'moving_average' } | null> {
    if (!this.available && Date.now() - this.lastCheckAt < this.CHECK_COOLDOWN_MS) {
      return null
    }
    try {
      const payload: ModelSeriesRequest = {
        data_by_junction: dataByJunction,
        completed_idx: completedIdx,
      }
      // Seri tahmini uzun sürebilir — timeout büyük tutuldu
      const resp = await this.http.post<ModelSeriesResponse>('/predict/series', payload, { timeout: 60_000 })
      this.available = true
      return { series: resp.data.prediction_series, source: resp.data.source }
    } catch {
      this.available = false
      this.lastCheckAt = Date.now()
      console.warn('[python-model] /predict/series ulaşılamıyor → TS fallback devreye alınıyor')
      return null
    }
  }

  /**
   * Bir onceki gunun gercek verisini tohum alarak bir sonraki gun icin
   * tum 144 slot otoregresif tahmin uretir.
   */
  async predictDayAhead(
    seedDataByJunction: Record<number, ArmData[]>,
    seedCompletedIdx = 143,
  ): Promise<{ series: Record<number, Record<string, number[]>>; source: 'AFDGCN' | 'moving_average' } | null> {
    if (!this.available && Date.now() - this.lastCheckAt < this.CHECK_COOLDOWN_MS) {
      return null
    }
    try {
      const resp = await this.http.post<{ prediction_series: Record<number, Record<string, number[]>>; source: string }>(
        '/predict/day-ahead',
        { seed_data_by_junction: seedDataByJunction, seed_completed_idx: seedCompletedIdx },
        { timeout: 120_000 },
      )
      this.available = true
      return {
        series: resp.data.prediction_series,
        source: resp.data.source as 'AFDGCN' | 'moving_average',
      }
    } catch {
      this.available = false
      this.lastCheckAt = Date.now()
      console.warn('[python-model] /predict/day-ahead ulasilamiyor')
      return null
    }
  }

  async getModelStatus(): Promise<Record<string, unknown>> {
    try {
      const resp = await this.http.get<Record<string, unknown>>('/model/status', { timeout: 5000 })
      return resp.data
    } catch {
      return { available: false, error: 'Model sunucusu ulaşılamıyor' }
    }
  }

  /**
   * Python model sunucusuna verilen .pth dosyasını yüklemesini söyler (dosya yolu ile).
   */
  async loadModel(params: {
    path: string
    numNodes: number
    lag: number
    horizon: number
    scalerMean: number
    scalerStd: number
  }): Promise<{ success: boolean; message: string }> {
    try {
      const resp = await this.http.post<{ success: boolean; message: string }>('/model/load', {
        path: params.path,
        num_nodes: params.numNodes,
        lag: params.lag,
        horizon: params.horizon,
        scaler_mean: params.scalerMean,
        scaler_std: params.scalerStd,
      }, { timeout: 30_000 })
      this.available = true
      return resp.data
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      return { success: false, message: `Model yükleme hatası: ${msg}` }
    }
  }

  /**
   * Ham .pth bytes'ını Python model sunucusuna gönderir (DB bytea → Python).
   * weights null ise filePath ile eski yönteme düşer.
   */
  async loadModelFromBytes(params: {
    weights: Buffer | Uint8Array | null
    filePath: string
    numNodes: number
    lag: number
    horizon: number
    scalerMean: number
    scalerStd: number
  }): Promise<{ success: boolean; message: string }> {
    if (!params.weights || params.weights.length === 0) {
      // Eski kayıt: weights yok, dosya yoluna geri dön
      return this.loadModel({
        path: params.filePath,
        numNodes: params.numNodes,
        lag: params.lag,
        horizon: params.horizon,
        scalerMean: params.scalerMean,
        scalerStd: params.scalerStd,
      })
    }
    try {
      const resp = await this.http.post<{ success: boolean; message: string }>(
        '/model/load-from-bytes',
        Buffer.isBuffer(params.weights) ? params.weights : Buffer.from(params.weights),
        {
          timeout: 30_000,
          params: {
            num_nodes: params.numNodes,
            lag: params.lag,
            horizon: params.horizon,
            scaler_mean: params.scalerMean,
            scaler_std: params.scalerStd,
          },
          headers: { 'Content-Type': 'application/octet-stream' },
        },
      )
      this.available = true
      return resp.data
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      return { success: false, message: `load-from-bytes hatası: ${msg}` }
    }
  }

  isAvailable(): boolean {
    return this.available
  }
}
