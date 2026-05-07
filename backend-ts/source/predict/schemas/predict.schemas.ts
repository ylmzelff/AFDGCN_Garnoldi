import { z } from 'zod'

export const RegionParamSchema = z.object({
  region: z.string().min(1),
})

export const JunctionParamSchema = z.object({
  junction_id: z.string().regex(/^\d+$/, 'junction_id sayısal olmalı'),
})

export const JunctionQuerySchema = z.object({
  region: z.string().optional(),
})

export type RegionParamInput = z.infer<typeof RegionParamSchema>
export type JunctionParamInput = z.infer<typeof JunctionParamSchema>
