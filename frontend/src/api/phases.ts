import { apiClient } from './client';
import type { RegionPhaseResponse, HealthResponse } from './types';

// Legacy endpoints (backward compatible)
export async function getIldemPhases(): Promise<RegionPhaseResponse> {
  const res = await apiClient.get<RegionPhaseResponse>('/phases/ildem');
  return res.data;
}

export async function getTunaPhases(): Promise<RegionPhaseResponse> {
  const res = await apiClient.get<RegionPhaseResponse>('/phases/tuna');
  return res.data;
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await apiClient.get<HealthResponse>('/health');
  return res.data;
}

// New prediction endpoints
export async function predictRegion(region: string): Promise<RegionPhaseResponse> {
  const res = await apiClient.post<RegionPhaseResponse>(`/predict/region/${region}`);
  return res.data;
}

export async function predictJunction(junctionId: number, region?: string): Promise<any> {
  const params = region ? `?region=${region}` : '';
  const res = await apiClient.post<any>(`/predict/junction/${junctionId}${params}`);
  return res.data;
}

export async function getPredictionStatus(): Promise<any> {
  const res = await apiClient.get<any>('/predict/status');
  return res.data;
}

export async function listRegions(): Promise<any> {
  const res = await apiClient.get<any>('/predict/regions');
  return res.data;
}
