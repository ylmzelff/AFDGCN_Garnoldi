import { z } from 'zod'

export const RegionParamSchema = z.object({
  region: z.string().min(1).max(50).regex(/^[a-z0-9_-]+$/, 'Geçersiz bölge adı'),
})

export type RegionParam = z.infer<typeof RegionParamSchema>
