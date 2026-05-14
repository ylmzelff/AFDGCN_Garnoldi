/**
 * In-memory uygulama durumu.
 * Python backend'deki AppState sınıfının TypeScript karşılığı.
 */

export interface UserInMemory {
  username: string
  hashedPassword: string
  disabled: boolean
}

export interface RegionConfigEntry {
  city: string
  region: string
  junctionIds: number[]
  useModel: boolean
  description: string
}

class AppState {
  /** In-memory kullanıcı deposu (demo kullanıcılar dahil). */
  usersDb = new Map<string, UserInMemory>()

  /** Veritabanı bağlantısı aktif mi? */
  dbEnabled = false

  /** DB'den yüklenen bölge konfigürasyonları. Key = region adı. */
  regionConfigs = new Map<string, RegionConfigEntry>()
}

export const appState = new AppState()
