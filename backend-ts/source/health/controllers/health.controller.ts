import type { Request, Response } from 'express'
import { kayseriClient, pythonModel } from '../../predict/services'
import { websocketService } from '../../common/services/websocket.service'
import { env } from '../../common/config'

export const getHealth = async (_req: Request, res: Response): Promise<void> => {
  const kayseriOk = await kayseriClient.healthCheck()
  const modelStatus = await pythonModel.getModelStatus()

  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    version: env.apiVersion,
    model: modelStatus,
    kayseri_api: kayseriOk ? 'connected' : 'unavailable',
    ws_clients: websocketService.clientCount,
  })
}
