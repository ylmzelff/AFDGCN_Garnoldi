// API response types — backend schemas/phases.py ile birebir eşleşir

export interface LoginRequest {
  username: string;
  password: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export type ArmStatus = 'low' | 'medium' | 'high';
export type PredictionSource = 'AFDGCN' | 'moving_average';
export type ApiStatus = 'connected' | 'unavailable';

export interface ArmPhase {
  arm: string;            // "A" | "B" | "C" | "D"
  arm_name: string;       // "Sivas Blv."
  vehicle_count: number;
  lanes: number;
  load: number;
  status: ArmStatus;
  green: number;          // saniye
  yellow: number;
  red: number;
  cycle_time: number;
}

export interface JunctionPhase {
  junction_id: number;
  junction_name: string;
  cycle_time: number;
  total_vehicles: number;
  arms: ArmPhase[];
}

export interface RegionPhaseResponse {
  region: string;
  city: string;
  timestamp: string;
  time_label: string;
  prediction_source: PredictionSource;
  kayseri_api_status: ApiStatus;
  junctions: JunctionPhase[];
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  model: Record<string, unknown>;
  kayseri_api: ApiStatus;
  ws_clients: number;
  version: string;
}

// WebSocket push payload
export interface WsPhaseUpdate {
  type: 'phase_update';
  regions: {
    ildem: RegionPhaseResponse;
    tuna: RegionPhaseResponse;
  };
}

// New Real-time Prediction Types
export interface PredictionResponse {
  region: string;
  timestamp: string;
  minute_index: number;
  predictions: Record<number, Record<string, number>>; // {jid: {arm: count, ...}, ...}
  phases: Record<number, Record<string, any>>;        // {jid: {cycle_time, arms, ...}, ...}
  source: PredictionSource;
  kayseri_ok: boolean;
  description: string;
}

export interface ArmDetail {
  name: string;
  vehicle_count: number;
  lanes: number;
  load: number;
  status: ArmStatus;
  green: number;
  yellow: number;
  red: number;
}

export interface JunctionDetailResponse {
  junction_id: number;
  junction_name: string;
  region: string;
  arms: Record<string, ArmDetail>;
  phase_recommendation: {
    cycle_time: number;
    total_vehicles: number;
  };
  timestamp: string;
  source: PredictionSource;
}

export interface PredictionStatus {
  cache: {
    size: number;
    ttl_seconds: number;
  };
  model: {
    loaded: boolean;
    num_nodes: number;
    fallback_active: boolean;
  };
  kayseri_api: {
    authenticated: boolean;
    token_valid: boolean;
  };
}

export interface RegionInfo {
  name: string;
  description: string;
  junction_count: number;
  junction_ids: number[];
  use_model: boolean;
}
