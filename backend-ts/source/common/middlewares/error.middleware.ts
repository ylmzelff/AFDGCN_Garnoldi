import type { Request, Response, NextFunction } from 'express'
import { ERRORS } from '../config'
import type { CustomError } from '../../types'

/**
 * Attaches `res.error(err)` helper on every response.
 * Must be registered BEFORE route handlers.
 */
export function errorMiddleware(_req: Request, res: Response, next: NextFunction): void {
  res.error = (err: unknown) => {
    const custom = err as CustomError

    if (custom?.code && custom?.status) {
      res.status(custom.status).json({
        error: true,
        code: custom.code,
        message: custom.message ?? 'Bir hata oluştu.',
      })
      return
    }

    const unexpected = ERRORS.UNEXPECTED
    console.error('[Unexpected Error]', err)
    res.status(unexpected.status).json({
      error: true,
      code: unexpected.code,
      message: unexpected.message,
    })
  }

  next()
}

/**
 * Global error handler — catches errors thrown or passed via next(err).
 */
export function globalErrorHandler(
  err: unknown,
  _req: Request,
  res: Response,
  _next: NextFunction,
): void {
  if (res.headersSent) return

  const custom = err as CustomError

  if (custom?.status && custom?.code) {
    res.status(custom.status).json({
      error: true,
      code: custom.code,
      message: custom.message ?? 'Bir hata oluştu.',
    })
    return
  }

  console.error('[Global Error Handler]', err)
  res.status(500).json({
    error: true,
    code: 'UNEXPECTED',
    message: 'Beklenmedik bir hata oluştu.',
  })
}
