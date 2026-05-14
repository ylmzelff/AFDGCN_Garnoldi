export interface ArmPhase {
  arm: string
  armName: string
  vehicleCount: number
  lanes: number
  load: number
  status: 'low' | 'medium' | 'high'
  green: number
  yellow: number
  red: number
  cycleTime: number
}

export interface JunctionPhase {
  junctionId: number
  junctionName: string
  cycleTime: number
  totalVehicles: number
  arms: ArmPhase[]
}

/**
 * Tek bir 10 dakikalık dilim için bir kavşağın faz önerisi.
 * Tahmin edilen araç sayıları Webster algoritmasına beslenerek üretilir.
 */
export interface PhaseSeriesItem {
  /** Gün başından itibaren slot indeksi (0 = 00:00, 1 = 00:10, …) */
  slot_index: number
  /** Zaman etiketi "HH:MM" biçiminde */
  time_label: string
  /** Webster döngü süresi (saniye) */
  cycle_time: number
  /** Toplam araç sayısı */
  total_vehicles: number
  /** Her kolun yeşil/sarı/kırmızı süreleri */
  arms: ArmPhase[]
}

export interface RegionPhaseResponse {
  region: string
  city: string
  timestamp: string
  timeLabel: string
  predictionSource: 'AFDGCN' | 'moving_average'
  kayseriApiStatus: 'connected' | 'unavailable'
  junctions: JunctionPhase[]
  /**
   * Gün başından mevcut dilime kadar her slot için faz önerileri.
   * junctionId → [PhaseSeriesItem per slot]
   */
  phase_series?: Record<number, PhaseSeriesItem[]>
}
