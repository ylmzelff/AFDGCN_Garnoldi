import { KayseriClientService } from './kayseri-client.service'
import { SivasClientService } from './sivas-client.service'
import { PythonModelService } from './python-model.service'
import { RealTimePredictorService } from './real-time-predictor.service'
import { BackgroundFetcherService } from './background-fetcher.service'
import { PhaseCalculatorService } from '../../phases/services/phase-calculator.service'
import type { TrafficClient } from './traffic-client.interface'

export const kayseriClient = new KayseriClientService()
export const sivasClient = new SivasClientService()
export const pythonModel = new PythonModelService()

/** Şehir → veri istemcisi. Yeni bir şehir eklenince buraya bir satır eklenir. */
export const clients: Record<string, TrafficClient> = {
  kayseri: kayseriClient,
  sivas: sivasClient,
}

const _phaseCalculator = new PhaseCalculatorService()
export const realTimePredictor = new RealTimePredictorService(clients, pythonModel, _phaseCalculator)
export const backgroundFetcher = new BackgroundFetcherService(realTimePredictor)
