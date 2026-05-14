import type { Request, Response } from 'express'
import { realTimePredictor, kayseriClient, pythonModel } from '../services'
import { REGION_CONFIG } from '../services/real-time-predictor.service'
import { logPrediction } from '../../common/services/db-logger.service'

export const predictRegion = async (req: Request, res: Response): Promise<void> => {
  const region = String(req.params['region'] ?? '').toLowerCase().trim()

  if (!REGION_CONFIG[region]) {
    const available = Object.keys(REGION_CONFIG).join(', ')
    res.status(400).json({
      error: true,
      code: 'REGION_NOT_FOUND',
      message: `Bilinmeyen bölge. Mevcut: ${available}`,
    })
    return
  }

  try {
    const result = await realTimePredictor.predictRegion(region)
    void logPrediction(result)
    res.json(result)
  } catch (err) {
    res.error(err)
  }
}

export const predictJunction = async (req: Request, res: Response): Promise<void> => {
  const junctionId = Number(req.params['junction_id'])
  let region = req.query['region'] as string | undefined

  if (!region) {
    for (const [r, config] of Object.entries(REGION_CONFIG)) {
      if (config.junctionIds.includes(junctionId)) {
        region = r
        break
      }
    }
  }

  if (!region) {
    res.status(404).json({
      error: true,
      code: 'REGION_NOT_FOUND',
      message: `Kavşak ${junctionId} hiçbir bölgede bulunamadı.`,
    })
    return
  }

  try {
    const result = await realTimePredictor.predictJunctionDetail(region, junctionId)
    // Kavşak detayı RegionPhaseResponse değil — bölge önbelleğini logla (fire-and-forget)
    void realTimePredictor.predictRegion(region).then((r) => void logPrediction(r)).catch(() => void 0)
    res.json(result)
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err)
    if (message.includes('bulunamadı')) {
      res.status(404).json({ error: true, code: 'REGION_NOT_FOUND', message })
    } else {
      res.error(err)
    }
  }
}

export const getPredictionStatus = async (_req: Request, res: Response): Promise<void> => {
  try {
    const cacheStatus = await realTimePredictor.getCacheStatus()
    const modelStatus = await pythonModel.getModelStatus()
    const apiStatus = kayseriClient.getStatus()

    res.json({
      cache: cacheStatus,
      model: modelStatus,
      kayseri_api: apiStatus,
    })
  } catch (err) {
    res.error(err)
  }
}

export const listRegions = (_req: Request, res: Response): void => {
  const regions = Object.entries(REGION_CONFIG).map(([name, config]) => ({
    name,
    city: config.city,
    description: config.description,
    junction_count: config.junctionIds.length,
    junction_ids: config.junctionIds,
    use_model: config.useModel,
  }))
  res.json({ regions })
}
