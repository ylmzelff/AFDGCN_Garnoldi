import { PhaseCalculatorService } from './phase-calculator.service'
import { PhasesService } from './phases.service'
import { clients, pythonModel } from '../../predict/services'

export const phaseCalculator = new PhaseCalculatorService()
export const phasesService = new PhasesService(clients, pythonModel, phaseCalculator)
