import { Router } from 'express'
import { authMiddleware } from '../../common/middlewares/auth.middleware'
import * as PhasesController from '../controllers/phases.controller'

const router = Router()

// Generic route — yeni bölgeler için kod değişikliği gerekmez
router.get('/:region', authMiddleware, PhasesController.getRegionPhases)

export default router
