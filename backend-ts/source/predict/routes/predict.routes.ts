import { Router } from 'express'
import { authMiddleware } from '../../common/middlewares/auth.middleware'
import * as PredictController from '../controllers/predict.controller'

const router = Router()

router.get('/regions', authMiddleware, PredictController.listRegions)
router.get('/status', authMiddleware, PredictController.getPredictionStatus)
router.post('/region/:region', authMiddleware, PredictController.predictRegion)
router.post('/junction/:junction_id', authMiddleware, PredictController.predictJunction)
router.get('/junction/:junction_id/yesterday', authMiddleware, PredictController.getJunctionYesterday)
router.get('/junction/:junction_id/day-phases', authMiddleware, PredictController.getJunctionDayPhases)

export default router
