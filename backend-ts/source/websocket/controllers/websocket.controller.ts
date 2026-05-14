/**
 * WebSocket Controller
 * =====================
 * /ws/live endpoint'ini yönetir.
 * Python websocket.py'nin TypeScript karşılığı.
 */

import { WebSocket, type WebSocketServer } from 'ws'
import type { IncomingMessage } from 'http'
import { URL } from 'url'
import { authService } from '../../auth/services'
import { websocketService } from '../../common/services/websocket.service'
import { phasesService } from '../../phases/services'
import { appState } from '../../common/services/app-state.service'

export function handleWebSocketConnection(ws: WebSocket, req: IncomingMessage): void {
  // Token URL query'den alınır: /ws/live?token=...
  const rawUrl = req.url ?? ''
  let token: string | null = null

  try {
    const parsed = new URL(rawUrl, 'ws://localhost')
    token = parsed.searchParams.get('token')
  } catch {
    ws.close(4001, 'Geçersiz URL')
    return
  }

  if (!token) {
    ws.close(4001, 'Token gerekli')
    return
  }

  const username = authService.decodeToken(token)
  if (!username || !appState.usersDb.has(username)) {
    ws.close(4001, 'Geçersiz token')
    return
  }

  websocketService.add(ws)
  console.info(`[ws] Bağlandı: ${username} (toplam=${websocketService.clientCount})`)

  // İlk bağlantıda anlık veri gönder
  void phasesService
    .pushAllRegions((payload) => websocketService.broadcast(payload))
    .catch((err) => console.error('[ws] İlk push hatası:', err))

  ws.on('message', (data) => {
    const msg = data.toString()
    if (msg === 'ping') {
      ws.send('pong')
    }
  })

  ws.on('close', () => {
    websocketService.remove(ws)
    console.info(`[ws] Bağlantı kesildi: ${username} (kalan=${websocketService.clientCount})`)
  })

  ws.on('error', (err) => {
    console.warn(`[ws] Hata (${username}):`, err)
    websocketService.remove(ws)
  })
}

export function initWebSocketServer(wss: WebSocketServer): void {
  wss.on('connection', handleWebSocketConnection)
}
