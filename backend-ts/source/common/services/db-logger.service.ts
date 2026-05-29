import prisma from '../../database/prisma'
import type { RegionPhaseResponse } from '../../phases/types/phases.types'
import type { RegionPredictionResult } from '../../predict/services/real-time-predictor.service'

let dbEnabled = false

export function setDbEnabled(enabled: boolean): void {
  dbEnabled = enabled
}

export function isDbEnabled(): boolean {
  return dbEnabled
}

function minuteIndex(): number {
  const now = new Date()
  return now.getHours() * 6 + Math.floor(now.getMinutes() / 10)
}

export async function logPrediction(response: RegionPhaseResponse | RegionPredictionResult): Promise<void> {
  if (!dbEnabled) return

  try {
    // RegionPhaseResponse (has .junctions[]) veya RegionPredictionResult (has .predictions{}) olabilir
    let totalVehicles = 0
    let junctionCount = 0
    let timeLabel = ''
    let predictionSource = ''
    let kayseriApiStatus: string = 'false'
    let city = 'kayseri'

    if ('junctions' in response && Array.isArray(response.junctions)) {
      // RegionPhaseResponse
      totalVehicles = response.junctions.reduce((sum, j) => sum + (j.totalVehicles ?? 0), 0)
      junctionCount = response.junctions.length
      timeLabel = response.timeLabel ?? ''
      predictionSource = response.predictionSource ?? ''
      kayseriApiStatus = response.kayseriApiStatus ?? 'false'
      city = response.city ?? 'kayseri'
    } else if ('predictions' in response && response.predictions) {
      // RegionPredictionResult
      totalVehicles = Object.values(response.predictions)
        .flatMap((arms) => Object.values(arms as Record<string, number>))
        .reduce((sum, v) => sum + v, 0)
      junctionCount = Object.keys(response.predictions).length
      timeLabel = response.time_label ?? ''
      predictionSource = response.source ?? ''
      kayseriApiStatus = String(response.kayseri_ok ?? false)
    }

    await prisma.phasePrediction.create({
      data: {
        region: response.region,
        city,
        timeLabel,
        minuteIndex: minuteIndex(),
        predictionSource,
        kayseriApiStatus,
        junctionCount,
        totalVehicles,
        payload: response as unknown as object,
      },
    })
  } catch (err) {
    // Non-critical — log quietly
    console.debug('[db-logger] logPrediction hatası (kritik değil):', err)
  }
}

export async function logModelEvent(
  eventType: string,
  modelPath = '',
  details = '',
): Promise<void> {
  if (!dbEnabled) return

  try {
    await prisma.modelEvent.create({
      data: { eventType, modelPath, details },
    })
  } catch (err) {
    console.debug('[db-logger] logModelEvent hatası (kritik değil):', err)
  }
}
