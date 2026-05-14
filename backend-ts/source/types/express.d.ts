import type { ApiErrorBody } from './global.d'

declare global {
  namespace Express {
    interface Response {
      error: (err: unknown) => void
    }
    interface Request {
      username?: string
    }
  }
}

declare module 'express-serve-static-core' {
  interface Response {
    error: (err: unknown) => void
  }
}

export type { ApiErrorBody }
