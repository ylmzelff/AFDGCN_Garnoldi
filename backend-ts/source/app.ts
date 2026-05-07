import 'dotenv/config'
import express from 'express'
import cors from 'cors'
import helmet from 'helmet'

import routes from './routes'
import { errorMiddleware, globalErrorHandler } from './common/middlewares/error.middleware'

const app = express()

// ─────────────────────────────────────────────────────────────────────────────
// Güvenlik & Parsing
// ─────────────────────────────────────────────────────────────────────────────

app.use(helmet())
app.use(cors({ origin: true, credentials: true }))
app.use(express.json({ limit: '10mb' }))
app.use(express.urlencoded({ extended: true }))

// ─────────────────────────────────────────────────────────────────────────────
// Yardımcı Middleware (res.error eklenir)
// ─────────────────────────────────────────────────────────────────────────────

app.use(errorMiddleware)

// ─────────────────────────────────────────────────────────────────────────────
// API Rotaları
// ─────────────────────────────────────────────────────────────────────────────

app.use('/api', routes)

// ─────────────────────────────────────────────────────────────────────────────
// Global Hata İşleyici
// ─────────────────────────────────────────────────────────────────────────────

app.use(globalErrorHandler)

export default app
