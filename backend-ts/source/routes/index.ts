import { Router } from 'express'
import AuthRoutes from '../auth/routes/auth.routes'
import PhasesRoutes from '../phases/routes/phases.routes'
import PredictRoutes from '../predict/routes/predict.routes'
import AdminRoutes from '../admin/routes/admin.routes'
import HealthRoutes from '../health/routes/health.routes'

const router = Router()

router.use('/auth', AuthRoutes)
router.use('/phases', PhasesRoutes)
router.use('/predict', PredictRoutes)
router.use('/admin', AdminRoutes)
router.use('/health', HealthRoutes)

export default router
