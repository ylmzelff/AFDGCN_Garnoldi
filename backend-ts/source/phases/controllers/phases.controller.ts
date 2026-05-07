import type { Request, Response } from 'express'
import { phasesService } from '../services'
import { logPrediction } from '../../common/services/db-logger.service'
import { REGION_CONFIG } from '../../predict/services/real-time-predictor.service'

export const getRegionPhases = async (req: Request, res: Response): Promise<void> => {
  const region = req.params.region?.toLowerCase().trim()
  if (!region || !REGION_CONFIG[region]) {
    res.status(404).json({ error: true, code: 'REGION_NOT_FOUND', message: `Bölge '${region}' bulunamadı.` })
    return
  }
  try {
    const response = await phasesService.buildRegionPhases(region)
    void logPrediction(response)
    res.json(response)
  } catch (err) {
    res.error(err)
  }
}

/** Geriye dönük uyumluluk için tutulan kısayollar */
export const getIldemPhases = async (_req: Request, res: Response): Promise<void> => {
  try {
    const response = await phasesService.buildRegionPhases('ildem')
    void logPrediction(response)
    res.json(response)
  } catch (err) {
    res.error(err)
  }
}

export const getTunaPhases = async (_req: Request, res: Response): Promise<void> => {
  try {
    const response = await phasesService.buildRegionPhases('tuna')
    void logPrediction(response)
    res.json(response)
  } catch (err) {
    res.error(err)
  }
}
