import axios from 'axios';
import { useAuthStore } from '@/store/useAuthStore';

const BASE_URL = import.meta.env.VITE_API_URL ?? '/api';

export const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 15_000,
  headers: { 'Content-Type': 'application/json' },
});

// Request interceptor: her isteğe JWT ekle
apiClient.interceptors.request.use((config) => {
  const token = useAuthStore.getState().token;
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Response interceptor: 401 → logout
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      useAuthStore.getState().logout();
    }
    return Promise.reject(error);
  }
);

const _wsBase = import.meta.env.VITE_WS_URL
  ?? `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}`;
export const WS_URL = _wsBase + '/ws/live';
