import { Router } from 'express'
import * as AuthController from '../controllers/auth.controller'
import { authMiddleware } from '../../common/middlewares/auth.middleware'
import { validateBody } from '../../common/middlewares/validation.middleware'
import { LoginSchema, RegisterSchema } from '../schemas/auth.schemas'

const router = Router()

router.post('/login', validateBody(LoginSchema), AuthController.login)
router.get('/me', authMiddleware, AuthController.me)
router.post('/register', validateBody(RegisterSchema), AuthController.register)

export default router
