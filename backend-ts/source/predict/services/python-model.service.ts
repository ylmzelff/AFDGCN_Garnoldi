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
      console.warn('[python-model] Model sunucusuna ulaşılamıyor → moving average devreye alınıyor')
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
   * Python model sunucusuna verilen .pth dosyasını yüklemesini söyler.
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

  isAvailable(): boolean {
    return this.available
  }
}
