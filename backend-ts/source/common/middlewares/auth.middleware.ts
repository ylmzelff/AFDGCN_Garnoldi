import type { Request, Response, NextFunction } from 'express'
import jwt from 'jsonwebtoken'
import { env } from '../config'

export function authMiddleware(req: Request, res: Response, next: NextFunction): void {
  const authHeader = req.headers['authorization']

  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    res.status(401).json({
      error: true,
      code: 'UNAUTHENTICATED',
      message: 'Kimlik doğrulaması gerekli.',
    })
    return
  }

  const token = authHeader.slice(7)

  try {
    const payload = jwt.verify(token, env.jwtSecret) as { sub: string }
    req.username = payload.sub
    next()
  } catch {
    res.status(401).json({
      error: true,
      code: 'INVALID_TOKEN',
      message: 'Geçersiz veya süresi dolmuş token.',
    })
  }
}
