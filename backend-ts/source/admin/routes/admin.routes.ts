import { Router } from 'express'
import { authMiddleware } from '../../common/middlewares/auth.middleware'
import * as AdminController from '../controllers/admin.controller'

const router = Router()

// ── Şehir API Konfigürasyonu ──────────────────────────────────────────────────
router.get('/api-configs', authMiddleware, AdminController.getApiConfigs)
router.put('/api-configs', authMiddleware, AdminController.upsertApiConfig)
router.delete('/api-configs/:city', authMiddleware, AdminController.deleteApiConfig)

// ── Bölge Konfigürasyonu ──────────────────────────────────────────────────────
router.get('/region-configs', authMiddleware, AdminController.getRegionConfigs)
router.put('/region-configs', authMiddleware, AdminController.upsertRegionConfig)
router.delete('/region-configs/:city/:region', authMiddleware, AdminController.deleteRegionConfig)

// ── Kavşak Bazlı Faz Ayarları ─────────────────────────────────────────────────
router.get('/phase-configs', authMiddleware, AdminController.getPhaseConfigs)
router.put('/phase-configs', authMiddleware, AdminController.upsertPhaseConfig)
router.delete('/phase-configs/:region/:junctionId', authMiddleware, AdminController.deletePhaseConfig)

// ── Model Versiyonları ────────────────────────────────────────────────────────
router.get('/models', authMiddleware, AdminController.listModelVersions)
router.post('/models/upload', authMiddleware, AdminController.modelUpload.single('file'), AdminController.uploadModelVersion)
router.post('/models/:id/activate', authMiddleware, AdminController.activateModelVersion)
router.delete('/models/:id', authMiddleware, AdminController.deleteModelVersion)

// ── Kullanıcı CRUD ────────────────────────────────────────────────────────────
router.get('/users', authMiddleware, AdminController.getUsers)
router.delete('/users/:username', authMiddleware, AdminController.deleteUser)
router.patch('/users/:username', authMiddleware, AdminController.toggleUser)

// ── Geçmiş ────────────────────────────────────────────────────────────────────
router.get('/predictions', authMiddleware, AdminController.getPredictions)
router.get('/model-events', authMiddleware, AdminController.getModelEvents)

export default router
