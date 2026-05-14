import { apiClient } from './client';
import type { LoginRequest, TokenResponse } from './types';

export async function login(data: LoginRequest): Promise<TokenResponse> {
  const res = await apiClient.post<TokenResponse>('/auth/login', data);
  return res.data;
}

export async function register(data: LoginRequest): Promise<TokenResponse> {
  const res = await apiClient.post<TokenResponse>('/auth/register', data);
  return res.data;
}

export async function getMe(): Promise<{ username: string }> {
  const res = await apiClient.get<{ username: string }>('/auth/me');
  return res.data;
}
