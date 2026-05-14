/**
 * AusTKM Kayseri Belediye API HTTP Istemcisi
 * ============================================
 * Gercek AusTKM API'sine static Bearer JWT token ile baglanir.
 * SSL sertifikasi self-signed oldugu icin dogrulama atlanmaktadir.
 *
 * Endpoint: GET /api/SensorVerileri?bolgeAdi=ILDEM&tarih=YYYY-MM-DD
 * Auth:     Bearer <AUSTM_API_TOKEN>
 */

import axios, { type AxiosInstance } from 'axios'
import https from 'https'
import { env } from '../../common/config'
import { REGION_CONFIG } from './real-time-predictor.service'

/** "HH:MM" -> 10 dakikalik dilim indeksi (0..143) */
function timeToSlotIndex(hhmm: string): number {
  const colonIdx = hhmm.indexOf(':')
  const h = parseInt(hhmm.slice(0, colonIdx), 10)
  const m = parseInt(hhmm.slice(colonIdx + 1), 10)
  return h * 6 + Math.floor(m / 10)
}

export type ArmData = Record<string, string | number>

// -----------------------------------------------------------------------------
// AusTKM API Tipleri
// -----------------------------------------------------------------------------

interface AusTkmResponse {
  success: boolean
  bolgeAdi?: string
  tarih?: string
  toplamKayit?: number
  data: AusTkmSensorItem[]
}

interface AusTkmSensorItem {
  intersectionId: number
  edgeDirection: string
  edgeName?: string
  saatlikVeriler: Record<string, number>
}

// -----------------------------------------------------------------------------
// Servis
// -----------------------------------------------------------------------------

export class KayseriClientService {
  private readonly baseUrl: string
  private readonly token: string
  private readonly http: AxiosInstance

  constructor(baseUrl?: string, token?: string) {
    this.baseUrl = (baseUrl ?? env.austmApiUrl).replace(/\/$/, '')
    this.token = token ?? env.austmApiToken

    // Belediye sunucusu self-signed sertifika kullandigindan SSL dogrulamasi atlaniyor
    const httpsAgent = new https.Agent({ rejectUnauthorized: false })

    this.http = axios.create({
      baseURL: this.baseUrl,
      timeout: 30_000,
      httpsAgent,
      headers: {
        Authorization: `Bearer ${this.token}`,
        Accept: 'application/json',
      },
    })
  }

  // ---------------------------------------------------------------------------
  // Veri Cekme
  // ---------------------------------------------------------------------------

  /**
   * Belirli bir bolge icin bugunun anlik trafik verisini ceker.
   * Donen format: Record<intersectionId, ArmData[]>
   * Her ArmData: { edge_direction, edge_name, "0": count, ..., "143": count }
   * Slot indeksi: saat*6 + dakika/10  (ornek: 07:30 -> 45)
   */
  async fetchRegion(
    region: string,
    _city = 'kayseri',
  ): Promise<Record<number, ArmData[]>> {
    const regionKey = region.toLowerCase()
    const regionCfg = REGION_CONFIG[regionKey]
    const bolgeAdi = regionCfg?.bolgeAdi || regionKey.toUpperCase()

    const tarih = new Date().toISOString().slice(0, 10) // YYYY-MM-DD

    console.info(`[kayseri-client] AusTKM veri cekiliyor: bolgeAdi=${bolgeAdi}, tarih=${tarih}`)

    const resp = await this.http.get<AusTkmResponse>('/api/SensorVerileri', {
      params: { bolgeAdi, tarih },
    })

    if (!resp.data.success || !Array.isArray(resp.data.data)) {
      throw new Error(
        `AusTKM API basarisiz yanit: ${JSON.stringify(resp.data).slice(0, 200)}`,
      )
    }

    console.info(
      `[kayseri-client] ${bolgeAdi}: ${resp.data.toplamKayit ?? resp.data.data.length} kayit alindi`,
    )

    return this.parseResponse(resp.data.data)
  }

  private parseResponse(items: AusTkmSensorItem[]): Record<number, ArmData[]> {
    // intersectionId -> { edgeDirection -> ArmData }
    const byJunction: Record<number, Record<string, ArmData>> = {}

    for (const item of items) {
      const jid = item.intersectionId
      if (!byJunction[jid]) byJunction[jid] = {}

      const direction = item.edgeDirection.trim().toUpperCase()

      const armData: ArmData = {
        edge_direction: direction,
        edge_name: item.edgeName ?? '',
      }

      // "HH:MM" -> slot indeksi -> count
      for (const [timeKey, count] of Object.entries(item.saatlikVeriler ?? {})) {
        const slotIdx = timeToSlotIndex(timeKey)
        armData[String(slotIdx)] = Number(count)
      }

      byJunction[jid][direction] = armData
    }

    const output: Record<number, ArmData[]> = {}
    for (const [jidStr, armMap] of Object.entries(byJunction)) {
      output[Number(jidStr)] = Object.values(armMap)
    }

    return output
  }

  // ---------------------------------------------------------------------------
  // Durum & Yonetim
  // ---------------------------------------------------------------------------

  async healthCheck(): Promise<boolean> {
    try {
      const tarih = new Date().toISOString().slice(0, 10)
      // İlk mevcut bölgenin bolgeAdi'sini kullan, yoksa fallback
      const firstBolgeAdi = Object.values(REGION_CONFIG)[0]?.bolgeAdi ?? '\u0130LDEM'
      await this.http.get('/api/SensorVerileri', {
        params: { bolgeAdi: firstBolgeAdi, tarih },
        timeout: 5_000,
      })
      return true
    } catch {
      return false
    }
  }

  getStatus(): Record<string, unknown> {
    return {
      base_url: this.baseUrl,
      token_configured: this.token.length > 0,
      api: 'AusTKM',
    }
  }

  getBaseUrl(): string {
    return this.baseUrl
  }

  /** Legacy uyumluluk - artık no-op */
  updateCredentials(_username: string, _password: string): void {
    console.warn('[kayseri-client] updateCredentials: AusTKM static token kullaniyor, bu islem etkisiz.')
  }

  /** AusTKM statik token kullandigindan authenticate gerekmez - uyumluluk icin */
  async ensureAuthenticated(): Promise<void> {
    if (!this.token) {
      throw new Error('[kayseri-client] AUSTM_API_TOKEN tanimlanmamis!')
    }
  }

  /** AusTKM icin base URL guncellemesi - uyumluluk icin */
  async updateBaseUrl(url: string): Promise<void> {
    console.info(`[kayseri-client] updateBaseUrl: ${url} (bilgi amacli, AusTKM statik URL kullanir)`)
  }
}

