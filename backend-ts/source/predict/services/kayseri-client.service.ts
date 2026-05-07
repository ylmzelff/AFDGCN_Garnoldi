/**
 * Kayseri Belediye API HTTP İstemcisi
 * =====================================
 * JWT token yönetimi ile port 9000'daki Kayseri API'sine bağlanır.
 * Python kayseri_client.py'nin TypeScript karşılığı.
 */

import axios, { type AxiosInstance } from 'axios'
import { env } from '../../common/config'

const TOKEN_REFRESH_MARGIN_MS = 300_000 // 5 dakika önce yenile

export type ArmData = Record<string, string | number>

export class KayseriClientService {
  private baseUrl: string
  private username: string
  private password: string

  private token: string | null = null
  private tokenExpiresAt = 0

  private http: AxiosInstance

  constructor(
    baseUrl?: string,
    username?: string,
    password?: string,
  ) {
    this.baseUrl = (baseUrl ?? env.kayseriApiUrl).replace(/\/$/, '')
    this.username = username ?? env.kayseriUsername
    this.password = password ?? env.kayseriPassword

    this.http = axios.create({
      baseURL: this.baseUrl,
      timeout: 30_000,
      headers: { 'Content-Type': 'application/json' },
    })
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Auth
  // ─────────────────────────────────────────────────────────────────────────

  async ensureAuthenticated(): Promise<void> {
    if (this.isTokenValid()) return
    await this.login()
  }

  private isTokenValid(): boolean {
    return this.token !== null && Date.now() < this.tokenExpiresAt - TOKEN_REFRESH_MARGIN_MS
  }

  private async login(): Promise<void> {
    const resp = await this.http.post<{ access_token: string; expires_in?: number }>(
      '/auth/login',
      { username: this.username, password: this.password },
    )
    this.token = resp.data.access_token
    const expiresIn = (resp.data.expires_in ?? 86400) * 1000
    this.tokenExpiresAt = Date.now() + expiresIn
    console.info(`[kayseri-client] Login başarılı (expires_in=${resp.data.expires_in ?? 86400}s)`)
  }

  private authHeader(): Record<string, string> {
    return { Authorization: `Bearer ${this.token}` }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Veri Çekme
  // ─────────────────────────────────────────────────────────────────────────

  async fetchRegion(
    region: string,
    city = 'kayseri',
  ): Promise<Record<number, ArmData[]>> {
    await this.ensureAuthenticated()

    try {
      const resp = await this.http.get<object>(`/${city}/${region}`, {
        headers: this.authHeader(),
      })
      return this.parseRegionResponse(resp.data)
    } catch (err: unknown) {
      const status = (err as { response?: { status: number } }).response?.status
      if (status === 401) {
        console.warn('[kayseri-client] Token reddedildi, yeniden login deneniyor...')
        this.token = null
        await this.login()
        const resp = await this.http.get<object>(`/${city}/${region}`, {
          headers: this.authHeader(),
        })
        return this.parseRegionResponse(resp.data)
      }
      throw err
    }
  }

  private parseRegionResponse(payload: object): Record<number, ArmData[]> {
    const result: Record<number, ArmData[]> = {}
    const data = payload as { junctions?: Array<{ junction_id: number; time_slots?: unknown[] }> }

    if (!data.junctions) return result

    for (const junction of data.junctions) {
      const jid = junction.junction_id
      const slots = junction.time_slots ?? []

      const byArm: Record<string, ArmData> = {}
      for (const slot of slots as Array<Record<string, unknown>>) {
        const direction = String(slot['edge_direction'] ?? '').trim().toUpperCase()
        if (!direction) continue

        if (!byArm[direction]) byArm[direction] = { edge_direction: direction }

        const slotIdx = slot['slot_index'] !== undefined ? String(slot['slot_index']) : null
        if (slotIdx !== null) {
          byArm[direction][slotIdx] = Number(slot['vehicle_count'] ?? 0)
        }
      }

      if (Object.keys(byArm).length > 0) {
        result[jid] = Object.values(byArm)
      }
    }

    return result
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Durum & Yönetim
  // ─────────────────────────────────────────────────────────────────────────

  async healthCheck(): Promise<boolean> {
    try {
      await this.http.get('/health', { timeout: 5000 })
      return true
    } catch {
      return false
    }
  }

  getStatus(): Record<string, unknown> {
    return {
      base_url: this.baseUrl,
      authenticated: this.isTokenValid(),
      token_expires_at: this.tokenExpiresAt > 0 ? new Date(this.tokenExpiresAt).toISOString() : null,
    }
  }

  updateCredentials(username: string, password: string): void {
    this.username = username
    this.password = password
    this.token = null
    this.tokenExpiresAt = 0
  }

  getBaseUrl(): string {
    return this.baseUrl
  }

  getCredentials(): { username: string; password: string } {
    return { username: this.username, password: this.password }
  }

  async updateBaseUrl(url: string): Promise<void> {
    this.baseUrl = url.replace(/\/$/, '')
    this.http = axios.create({
      baseURL: this.baseUrl,
      timeout: 30_000,
      headers: { 'Content-Type': 'application/json' },
    })
    this.token = null
    this.tokenExpiresAt = 0
    try {
      await this.ensureAuthenticated()
    } catch (err) {
      console.warn('[kayseri-client] Yeni URL ile login hatası:', err)
    }
  }
}
