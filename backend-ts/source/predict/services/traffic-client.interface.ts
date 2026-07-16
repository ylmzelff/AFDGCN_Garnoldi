/**
 * Ortak Trafik Veri Istemcisi Arayuzu
 * =====================================
 * Her sehir kendi API formatini/auth yontemini farkli sekilde uygular
 * (KayseriClientService: AusTKM Bearer token; SivasClientService: auth'suz
 * canlı sayac API'si). RealTimePredictorService ve PhasesService bu arayuz
 * uzerinden calisir, sehire ozel kod icermez.
 */

export type ArmData = Record<string, string | number>

export interface TrafficClient {
  fetchRegion(region: string, city?: string): Promise<Record<number, ArmData[]>>
  fetchRegionForDate(region: string, tarih: string, city?: string): Promise<Record<number, ArmData[]>>
  healthCheck(): Promise<boolean>
  getStatus(): Record<string, unknown>
  getBaseUrl(): string
  updateCredentials(username: string, password: string): void
  ensureAuthenticated(): Promise<void>
  updateBaseUrl(url: string): Promise<void>
  /** Bu veri kaynağının doğal zaman dilimi (dakika). Kayseri: 10, Sivas: 60 (saatlik). */
  getSlotMinutes(): number
}
