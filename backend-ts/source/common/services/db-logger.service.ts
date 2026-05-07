import prisma from '../../database/prisma'
import type { RegionPhaseResponse } from '../../phases/types/phases.types'

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

export async function logPrediction(response: RegionPhaseResponse): Promise<void> {
  if (!dbEnabled) return

  try {
    const totalVehicles = response.junctions.reduce((sum, j) => sum + j.totalVehicles, 0)

    await prisma.phasePrediction.create({
      data: {
        region: response.region,
        city: response.city,
        timeLabel: response.timeLabel,
        minuteIndex: minuteIndex(),
        predictionSource: response.predictionSource,
        kayseriApiStatus: response.kayseriApiStatus,
        junctionCount: response.junctions.length,
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
