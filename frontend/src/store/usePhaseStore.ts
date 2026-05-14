import { create } from 'zustand';
import type { RegionPhaseResponse } from '@/api/types';

type WsStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

interface PhaseState {
  ildem: RegionPhaseResponse | null;
  tuna: RegionPhaseResponse | null;
  wsStatus: WsStatus;
  lastUpdated: string | null;
  setIldem: (data: RegionPhaseResponse) => void;
  setTuna: (data: RegionPhaseResponse) => void;
  setWsStatus: (status: WsStatus) => void;
}

export const usePhaseStore = create<PhaseState>()((set) => ({
  ildem: null,
  tuna: null,
  wsStatus: 'disconnected',
  lastUpdated: null,

  setIldem: (data) => set({ ildem: data, lastUpdated: new Date().toISOString() }),
  setTuna: (data) => set({ tuna: data, lastUpdated: new Date().toISOString() }),
  setWsStatus: (wsStatus) => set({ wsStatus }),
}));
