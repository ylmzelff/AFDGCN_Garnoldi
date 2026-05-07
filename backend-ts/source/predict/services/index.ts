import { KayseriClientService } from './kayseri-client.service'
import { PythonModelService } from './python-model.service'
import { RealTimePredictorService } from './real-time-predictor.service'
import { BackgroundFetcherService } from './background-fetcher.service'
import { PhaseCalculatorService } from '../../phases/services/phase-calculator.service'

export const kayseriClient = new KayseriClientService()
export const pythonModel = new PythonModelService()

const _phaseCalculator = new PhaseCalculatorService()
export const realTimePredictor = new RealTimePredictorService(kayseriClient, pythonModel, _phaseCalculator)
export const backgroundFetcher = new BackgroundFetcherService(realTimePredictor)
