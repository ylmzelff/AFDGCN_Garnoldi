import { z } from 'zod'

export const LoginSchema = z.object({
  username: z.string().min(1, 'Kullanıcı adı gerekli.'),
  password: z.string().min(1, 'Şifre gerekli.'),
})

export const RegisterSchema = z.object({
  username: z.string().min(3).max(32),
  password: z.string().min(6),
})

export type LoginInput = z.infer<typeof LoginSchema>
export type RegisterInput = z.infer<typeof RegisterSchema>
