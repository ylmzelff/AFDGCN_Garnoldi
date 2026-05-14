import type { Request, Response } from 'express'
import { authService } from '../services'
import type { LoginInput, RegisterInput } from '../schemas/auth.schemas'

export const login = async (req: Request, res: Response): Promise<void> => {
  try {
    const { username, password } = req.body as LoginInput
    const result = await authService.login(username, password)
    if (!result) {
      res.status(401).json({
        error: true,
        code: 'UNAUTHENTICATED',
        message: 'Geçersiz kullanıcı adı veya şifre.',
      })
      return
    }
    res.json(result)
  } catch (err) {
    res.error(err)
  }
}

export const me = (req: Request, res: Response): void => {
  res.json({ username: req.username })
}

export const register = async (req: Request, res: Response): Promise<void> => {
  try {
    const { username, password } = req.body as RegisterInput
    const result = await authService.register(username, password)
    res.status(201).json(result)
  } catch (err) {
    res.error(err)
  }
}
