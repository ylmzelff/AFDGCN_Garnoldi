import type { Request, Response, NextFunction } from 'express'
import { z, ZodSchema } from 'zod'

function sendValidationError(res: Response, errors: z.ZodIssue[]): void {
  const details = errors.map((e) => ({
    field: e.path.join('.'),
    message: e.message,
  }))
  res.status(422).json({
    error: true,
    code: 'VALIDATION',
    message: 'İstek doğrulama hatası.',
    details,
  })
}

export function validateBody<T>(schema: ZodSchema<T>) {
  return (req: Request, res: Response, next: NextFunction): void => {
    const result = schema.safeParse(req.body)
    if (!result.success) {
      sendValidationError(res, result.error.issues)
      return
    }
    req.body = result.data as typeof req.body
    next()
  }
}

export function validateParams<T>(schema: ZodSchema<T>) {
  return (req: Request, res: Response, next: NextFunction): void => {
    const result = schema.safeParse(req.params)
    if (!result.success) {
      sendValidationError(res, result.error.issues)
      return
    }
    req.params = result.data as typeof req.params
    next()
  }
}

export function validateQuery<T>(schema: ZodSchema<T>) {
  return (req: Request, res: Response, next: NextFunction): void => {
    const result = schema.safeParse(req.query)
    if (!result.success) {
      sendValidationError(res, result.error.issues)
      return
    }
    next()
  }
}
