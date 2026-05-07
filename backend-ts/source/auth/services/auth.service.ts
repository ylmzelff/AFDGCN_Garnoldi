import bcrypt from 'bcryptjs'
import jwt from 'jsonwebtoken'
import { env } from '../../common/config'
import { appState } from '../../common/services/app-state.service'
import prisma from '../../database/prisma'
import type { TokenResponse } from '../types/auth.types'

export class AuthService {
  hashPassword(password: string): string {
    return bcrypt.hashSync(password, 10)
  }

  verifyPassword(plain: string, hashed: string): boolean {
    return bcrypt.compareSync(plain, hashed)
  }

  createToken(username: string): string {
    return jwt.sign({ sub: username }, env.jwtSecret, {
      expiresIn: env.jwtExpiresInSeconds,
      algorithm: 'HS256',
    })
  }

  decodeToken(token: string): string | null {
    try {
      const payload = jwt.verify(token, env.jwtSecret) as { sub: string }
      return payload.sub ?? null
    } catch {
      return null
    }
  }

  buildTokenResponse(username: string): TokenResponse {
    return {
      access_token: this.createToken(username),
      token_type: 'bearer',
      expires_in: env.jwtExpiresInSeconds,
    }
  }

  async login(username: string, password: string): Promise<TokenResponse | null> {
    let user = appState.usersDb.get(username)

    // Memory'de yoksa DB'den yükle
    if (!user && appState.dbEnabled) {
      const dbUser = await prisma.user.findUnique({ where: { username } })
      if (dbUser) {
        user = { username: dbUser.username, hashedPassword: dbUser.hashedPassword, disabled: dbUser.disabled }
        appState.usersDb.set(username, user)
      }
    }

    if (!user || !this.verifyPassword(password, user.hashedPassword)) return null
    if (user.disabled) throw Object.assign(new Error('Hesap devre dışı.'), { status: 403, code: 'ACCOUNT_DISABLED' })
    return this.buildTokenResponse(username)
  }

  async register(username: string, password: string): Promise<TokenResponse> {
    // DB'de de kontrol et
    if (appState.dbEnabled) {
      const existing = await prisma.user.findUnique({ where: { username } })
      if (existing) {
        throw Object.assign(new Error('Bu kullanıcı adı zaten kullanılıyor.'), { status: 409, code: 'DUPLICATE' })
      }
    } else if (appState.usersDb.has(username)) {
      throw Object.assign(new Error('Bu kullanıcı adı zaten kullanılıyor.'), { status: 409, code: 'DUPLICATE' })
    }

    const hashedPassword = this.hashPassword(password)

    // DB'ye kaydet
    if (appState.dbEnabled) {
      await prisma.user.create({ data: { username, hashedPassword, disabled: false } })
    }

    // Memory'e de ekle
    appState.usersDb.set(username, { username, hashedPassword, disabled: false })

    return this.buildTokenResponse(username)
  }
}
