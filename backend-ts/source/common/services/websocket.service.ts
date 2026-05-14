import { WebSocket } from 'ws'

/**
 * Thread-safe (single-threaded Node.js) WebSocket client registry.
 * Handles broadcast and dead-connection cleanup.
 */
class WebSocketService {
  private clients = new Set<WebSocket>()

  add(ws: WebSocket): void {
    this.clients.add(ws)
  }

  remove(ws: WebSocket): void {
    this.clients.delete(ws)
  }

  broadcast(payload: object): void {
    const message = JSON.stringify(payload)
    const dead: WebSocket[] = []

    for (const ws of this.clients) {
      if (ws.readyState === WebSocket.OPEN) {
        try {
          ws.send(message)
        } catch {
          dead.push(ws)
        }
      } else {
        dead.push(ws)
      }
    }

    for (const ws of dead) {
      this.clients.delete(ws)
    }
  }

  get clientCount(): number {
    return this.clients.size
  }
}

export const websocketService = new WebSocketService()
