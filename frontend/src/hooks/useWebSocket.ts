import { useEffect, useRef, useCallback } from 'react';
import { WS_URL } from '@/api/client';
import { useAuthStore } from '@/store/useAuthStore';
import { usePhaseStore } from '@/store/usePhaseStore';
import type { WsPhaseUpdate } from '@/api/types';

const RECONNECT_DELAY = 3000; // ms
const PING_INTERVAL = 25000;  // ms

export function useWebSocket() {
  const token = useAuthStore((s) => s.token);
  const { setIldem, setTuna, setWsStatus } = usePhaseStore();

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pingTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const isMounted = useRef(true);

  const clearTimers = useCallback(() => {
    if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    if (pingTimer.current) clearInterval(pingTimer.current);
  }, []);

  const connect = useCallback(() => {
    if (!token || !isMounted.current) return;

    const url = `${WS_URL}?token=${encodeURIComponent(token)}`;
    setWsStatus('connecting');

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      if (!isMounted.current) return;
      setWsStatus('connected');

      // Ping her 25 saniyede bir
      pingTimer.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send('ping');
        }
      }, PING_INTERVAL);
    };

    ws.onmessage = (event) => {
      if (!isMounted.current) return;
      try {
        const payload: WsPhaseUpdate = JSON.parse(event.data as string);
        if (payload.type === 'phase_update') {
          if (payload.regions.ildem) setIldem(payload.regions.ildem);
          if (payload.regions.tuna) setTuna(payload.regions.tuna);
        }
      } catch {
        // pong veya bilinmeyen mesaj — yoksay
      }
    };

    ws.onerror = () => {
      if (!isMounted.current) return;
      setWsStatus('error');
    };

    ws.onclose = () => {
      clearTimers();
      if (!isMounted.current) return;
      setWsStatus('disconnected');
      // Auto-reconnect
      reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY);
    };
  }, [token, setIldem, setTuna, setWsStatus, clearTimers]);

  useEffect(() => {
    isMounted.current = true;
    connect();

    return () => {
      isMounted.current = false;
      clearTimers();
      wsRef.current?.close();
    };
  }, [connect, clearTimers]);
}
