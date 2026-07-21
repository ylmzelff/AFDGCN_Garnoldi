/**
 * Python AFDGCN Model İstemcisi
 * ================================
 * PYTHON_MODEL_URL adresinde çalışan Python model sunucusunu çağırır.
 * Model sunucusu yoksa veya hata alırsa null döner ve caller moving average'a düşer.
 *
 * Her çağrı hangi bölge (region) için olduğunu belirtir — Python tarafı
 * bölge başına ayrı model tutar, böylece birden fazla şehir aynı anda
 * birbirini etkilemeden serve edilebilir.
 *
 * Python model sunucusu başlatmak için:
 *   python model_server.py  (FastAPI, port 9002)
 */

import axios, { type AxiosInstance } from 'axios'
import { env } from '../../common/config'
import type { ArmData } from './kayseri-client.service'

interface ModelPredictRequest {
  region: string
  data_by_junction: Record<number, ArmData[]>
  minute_index: number
}

interface ModelPredictResponse {
  predictions: Record<number, Record<string, number>>
  source: 'AFDGCN' | 'moving_average'
}

interface ModelSeriesRequest {
  region: string
  data_by_junction: Record<number, ArmData[]>
  completed_idx: number
}

interface ModelSeriesResponse {
  prediction_series: Record<number, Record<string, number[]>>
  source: 'AFDGCN' | 'moving_average'
}

export interface NodeMap {
  [junctionId: number]: Record<string, number>
}

export type GraphEdges = Array<[number, number] | [number, number, number]>

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
    region: string,
    dataByJunction: Record<number, ArmData[]>,
    minuteIndex: number,
  ): Promise<Record<number, Record<string, number>> | null> {
    // Bilinenin unavailable olduğunu düzenli aralıklarla yeniden dene
    if (!this.available && Date.now() - this.lastCheckAt < this.CHECK_COOLDOWN_MS) {
      return null
    }

    try {
      const payload: ModelPredictRequest = {
        region,
        data_by_junction: dataByJunction,
        minute_index: minuteIndex,
      }
      const resp = await this.http.post<ModelPredictResponse>('/predict/next', payload)
      this.available = true
      return resp.data.predictions
    } catch {
      this.available = false
      this.lastCheckAt = Date.now()
      console.warn(`[python-model] /predict/next ulaşılamıyor (region=${region}) → moving average devreye alınıyor`)
      return null
    }
  }

  /**
   * Gün başından completedIdx dahil her slot için AFDGCN rolling tahmin serisi.
   * Her slotun tahmini yalnızca o slottan ÖNCEKİ gerçek verilerle üretilir.
   * @returns {series, source} veya null (model sunucusu yoksa)
   */
  async predictSeries(
    region: string,
    dataByJunction: Record<number, ArmData[]>,
    completedIdx: number,
  ): Promise<{ series: Record<number, Record<string, number[]>>; source: 'AFDGCN' | 'moving_average' } | null> {
    if (!this.available && Date.now() - this.lastCheckAt < this.CHECK_COOLDOWN_MS) {
      return null
    }
    try {
      const payload: ModelSeriesRequest = {
        region,
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
      console.warn(`[python-model] /predict/series ulaşılamıyor (region=${region}) → TS fallback devreye alınıyor`)
      return null
    }
  }

  /**
   * region=null ise yüklü tüm bölgelerin durumunu döner.
   */
  async getModelStatus(region?: string): Promise<Record<string, unknown>> {
    try {
      const resp = await this.http.get<Record<string, unknown>>('/model/status', {
        timeout: 5000,
        params: region ? { region } : undefined,
      })
      return resp.data
    } catch {
      return { available: false, error: 'Model sunucusu ulaşılamıyor' }
    }
  }

  /**
   * Python model sunucusuna verilen .pth dosyasını yüklemesini söyler (dosya yolu ile).
   */
  async loadModel(params: {
    region: string
    path: string
    numNodes: number
    lag: number
    horizon: number
    scalerMean: number
    scalerStd: number
    nodeMap?: NodeMap
    graphEdges?: GraphEdges
  }): Promise<{ success: boolean; message: string }> {
    try {
      const resp = await this.http.post<{ success: boolean; message: string }>('/model/load', {
        region: params.region,
        path: params.path,
        num_nodes: params.numNodes,
        lag: params.lag,
        horizon: params.horizon,
        scaler_mean: params.scalerMean,
        scaler_std: params.scalerStd,
        node_map: params.nodeMap ?? null,
        graph_edges: params.graphEdges ?? null,
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
    region: string
    weights: Buffer | Uint8Array | null
    filePath: string
    numNodes: number
    lag: number
    horizon: number
    scalerMean: number
    scalerStd: number
    nodeMap?: NodeMap
    graphEdges?: GraphEdges
  }): Promise<{ success: boolean; message: string }> {
    if (!params.weights || params.weights.length === 0) {
      // Eski kayıt: weights yok, dosya yoluna geri dön
      return this.loadModel({
        region: params.region,
        path: params.filePath,
        numNodes: params.numNodes,
        lag: params.lag,
        horizon: params.horizon,
        scalerMean: params.scalerMean,
        scalerStd: params.scalerStd,
        nodeMap: params.nodeMap,
        graphEdges: params.graphEdges,
      })
    }
    try {
      const resp = await this.http.post<{ success: boolean; message: string }>(
        '/model/load-from-bytes',
        Buffer.isBuffer(params.weights) ? params.weights : Buffer.from(params.weights),
        {
          timeout: 30_000,
          params: {
            region: params.region,
            num_nodes: params.numNodes,
            lag: params.lag,
            horizon: params.horizon,
            scaler_mean: params.scalerMean,
            scaler_std: params.scalerStd,
            node_map: params.nodeMap ? JSON.stringify(params.nodeMap) : undefined,
            graph_edges: params.graphEdges ? JSON.stringify(params.graphEdges) : undefined,
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
