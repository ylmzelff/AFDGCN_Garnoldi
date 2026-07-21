/**
 * Sivas Belediyesi Ulasim Sayac API Istemcisi
 * ==============================================
 * Endpoint: GET /apir/counter/listemiz/{tarih}/{id1,id2,...}
 * Auth gerektirmez. Kayseri'nin AusTKM API'sinden farkli olarak:
 *   - Yanitta junction ID donmez, sadece "kavsakAd" (isim) doner -> isim/ID
 *     eslemesi STATIC_JUNCTION_NAMES uzerinden yapilir.
 *   - Veri dogal olarak SAATLIK (gunde 24 nokta). Kayseri'nin 10 dakikalik
 *     (144 slot) verisi gibi yapay olarak bolunmez/interpole edilmez —
 *     ArmData "0".."23" anahtarlariyla saatlik tutulur. getSlotMinutes()
 *     60 dondurerek downstream servislere (RealTimePredictorService,
 *     PhasesService) bu kaynagin dogal granularitesini bildirir.
 */

import axios, { type AxiosInstance } from 'axios'
import { env } from '../../common/config'
import { REGION_CONFIG } from './real-time-predictor.service'
import type { TrafficClient, ArmData } from './traffic-client.interface'

// -----------------------------------------------------------------------------
// Bilinen Kavsaklar (canli API'den kesfedilmis gercek isim/ID kayitlari)
// -----------------------------------------------------------------------------

export const SIVAS_JUNCTION_NAMES: Record<number, string> = {
  2: 'KÖY HİZMETLERİ',
  9: 'TSO',
  3: 'EVCİLER',
}

function normalizeName(name: string): string {
  return name.trim().toLocaleUpperCase('tr')
}

const NAME_TO_ID: Record<string, number> = Object.fromEntries(
  Object.entries(SIVAS_JUNCTION_NAMES).map(([id, name]) => [normalizeName(name), Number(id)]),
)

/**
 * Canli API "yon" alanini gercek yer adiyla dondurur (ör. "Hilton", "Universite->4 İşletme"),
 * "A"/"B"/"C"/"D" harfi degil. node_map / egitim verisi harf bazli oldugu icin burada
 * kavsak bazinda isim -> harf eslemesi yapiyoruz. Eslenmeyen yonler (ör. TSO'nun 4. yonu,
 * EVCİLER'in "Karayollari" yonu) modelin egitildigi semada olmadigi icin atlanir.
 */
const DIRECTION_NAME_TO_ARM: Record<number, Record<string, string>> = {
  9: { // TSO
    [normalizeName('Universite->4 İşletme')]: 'A',
    [normalizeName('4 İşletme->Üniversite')]: 'B',
    [normalizeName('Kültür Müdürlüğü->Üniversite')]: 'D',
  },
  3: { // EVCİLER
    [normalizeName('Hilton')]: 'A',
    [normalizeName('Stad Geliş')]: 'B',
  },
}

// -----------------------------------------------------------------------------
// Sivas API Tipleri
// -----------------------------------------------------------------------------

interface SivasSayimItem {
  saat: string // "HH:MM:SS"
  sayi: number
}

interface SivasYonItem {
  yon: string
  sayim: SivasSayimItem[]
}

interface SivasKavsakItem {
  tarih: string
  kavsakAd: string
  kavsakYonler: SivasYonItem[]
}

// -----------------------------------------------------------------------------
// Servis
// -----------------------------------------------------------------------------

export class SivasClientService implements TrafficClient {
  private baseUrl: string
  private http: AxiosInstance

  constructor(baseUrl?: string) {
    this.baseUrl = (baseUrl ?? env.sivasApiUrl).replace(/\/$/, '')
    this.http = this.buildHttpClient()
  }

  private buildHttpClient(): AxiosInstance {
    return axios.create({
      baseURL: this.baseUrl,
      timeout: 30_000,
      headers: { Accept: 'application/json' },
    })
  }

  // ---------------------------------------------------------------------------
  // Veri Cekme
  // ---------------------------------------------------------------------------

  async fetchRegion(region: string, _city = 'sivas'): Promise<Record<number, ArmData[]>> {
    const tarih = new Date().toISOString().slice(0, 10)
    return this.fetchRegionForDate(region, tarih, _city)
  }

  async fetchRegionForDate(
    region: string,
    tarih: string,
    _city = 'sivas',
  ): Promise<Record<number, ArmData[]>> {
    const regionKey = region.toLowerCase()
    const junctionIds = REGION_CONFIG[regionKey]?.junctionIds ?? []
    if (junctionIds.length === 0) {
      throw new Error(`[sivas-client] '${region}' bölgesi için kavşak tanımlanmamış.`)
    }

    console.info(`[sivas-client] Sivas veri çekiliyor: ids=${junctionIds.join(',')}, tarih=${tarih}`)

    const resp = await this.http.get<SivasKavsakItem[]>(
      `/apir/counter/listemiz/${tarih}/${junctionIds.join(',')}`,
    )

    if (!Array.isArray(resp.data)) {
      throw new Error(`Sivas API beklenmeyen yanıt: ${JSON.stringify(resp.data).slice(0, 200)}`)
    }

    console.info(`[sivas-client] ${resp.data.length} kavşak verisi alındı`)

    return this.parseResponse(resp.data);
  }

  private parseResponse(items: SivasKavsakItem[]): Record<number, ArmData[]> {
    const output: Record<number, ArmData[]> = {}

    for (const item of items) {
      const jid = NAME_TO_ID[normalizeName(item.kavsakAd)]
      if (jid === undefined) {
        console.warn(`[sivas-client] Bilinmeyen kavşak adı, atlanıyor: '${item.kavsakAd}'`)
        continue
      }

      output[jid] = item.kavsakYonler
        .map((y) => this.toArmData(y, jid))
        .filter((a): a is ArmData => a !== null)
    }

    return output
  }

  /**
   * Saatlik veriyi ArmData'ya çevirir — "0".."23" anahtarları, her biri o
   * saatin toplam araç sayısı. Aynı saat için birden fazla kayıt gelirse
   * (API'de görülen bir durum) toplanır, asla bölünmez/interpole edilmez.
   * Kavsagin bilinen bir yon->harf eslemesi yoksa (jid DIRECTION_NAME_TO_ARM'da
   * yok) veya bu spesifik yon eslenemiyorsa null doner (cagiran filtreler).
   */
  private toArmData(yon: SivasYonItem, jid: number): ArmData | null {
    const arm = DIRECTION_NAME_TO_ARM[jid]?.[normalizeName(yon.yon)]
    if (!arm) return null

    const hourTotals = new Array<number>(24).fill(0)

    for (const { saat, sayi } of yon.sayim) {
      const hour = parseInt(saat.slice(0, 2), 10)
      if (hour >= 0 && hour < 24) {
        hourTotals[hour] += Number(sayi) || 0
      }
    }

    const armData: ArmData = { edge_direction: arm, edge_name: yon.yon }

    for (let hour = 0; hour < 24; hour++) {
      armData[String(hour)] = hourTotals[hour]
    }

    return armData
  }

  // ---------------------------------------------------------------------------
  // Durum & Yonetim
  // ---------------------------------------------------------------------------

  async healthCheck(): Promise<boolean> {
    try {
      const tarih = new Date().toISOString().slice(0, 10)
      const firstId = Object.keys(SIVAS_JUNCTION_NAMES)[0] ?? '2'
      await this.http.get(`/apir/counter/listemiz/${tarih}/${firstId}`, { timeout: 5_000 })
      return true
    } catch {
      return false
    }
  }

  getStatus(): Record<string, unknown> {
    return {
      base_url: this.baseUrl,
      token_configured: true, // auth gerekmiyor
      api: 'Sivas Ulaşım Sayaç API',
    }
  }

  getBaseUrl(): string {
    return this.baseUrl
  }

  /** Sivas sayaç verisi saatlik gelir (günde 24 nokta) — 10dk'lık veriye interpole edilmez. */
  getSlotMinutes(): number {
    return 60
  }

  /** Sivas API auth gerektirmediğinden no-op - uyumluluk için. */
  updateCredentials(_username: string, _password: string): void {
    console.info('[sivas-client] updateCredentials: Sivas API auth gerektirmiyor, bu işlem etkisiz.')
  }

  /** Sivas API auth gerektirmediğinden her zaman çözümlenir. */
  async ensureAuthenticated(): Promise<void> {
    return Promise.resolve()
  }

  async updateBaseUrl(url: string): Promise<void> {
    this.baseUrl = url.replace(/\/$/, '')
    this.http = this.buildHttpClient()
    console.info(`[sivas-client] Base URL güncellendi: ${this.baseUrl}`)
  }
}
