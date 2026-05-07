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

export interface RegionPhaseResponse {
  region: string
  city: string
  timestamp: string
  timeLabel: string
  predictionSource: 'AFDGCN' | 'moving_average'
  kayseriApiStatus: 'connected' | 'unavailable'
  junctions: JunctionPhase[]
}
