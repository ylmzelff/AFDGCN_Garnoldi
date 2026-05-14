/**
 * Kavşak görüntüleme metadatası.
 * Yeni şehir/bölge kavşakları buraya eklenir; backend kodu değişmez.
 * Bilinmeyen kavşaklar için varsayılan isim ve kol dizisi kullanılır.
 */

export const DEFAULT_ARMS: readonly string[] = ['A', 'B', 'C', 'D']

export interface JunctionMeta {
  name: string
  arms: readonly string[]
}

export const JUNCTION_DISPLAY: Record<number, JunctionMeta> = {
  // ── İldem (Kayseri) ────────────────────────────────────────────────────────
  89:  { name: 'Gesi',       arms: ['A', 'B', 'C', 'D'] },
  95:  { name: 'Beyazşehir', arms: ['A', 'B', 'C', 'D'] },
  117: { name: 'İldem 3',    arms: ['A', 'C', 'D'] },
  121: { name: 'Toki',       arms: ['A', 'B', 'C', 'D'] },
  184: { name: 'İldem 1',    arms: ['A', 'B', 'D'] },
  187: { name: 'Serkent',    arms: ['A', 'B', 'C', 'D'] },
  188: { name: 'İldem 2',    arms: ['A', 'B', 'C', 'D'] },
  192: { name: 'İldem 4',    arms: ['A', 'B', 'C', 'D'] },
  194: { name: 'İldem 5',    arms: ['A', 'B', 'C', 'D'] },
  // ── Tuna (Kayseri) ─────────────────────────────────────────────────────────
  3:   { name: 'Tuna 3',     arms: ['A', 'B', 'C', 'D'] },
  5:   { name: 'Tuna 5',     arms: ['A', 'B', 'C', 'D'] },
  7:   { name: 'Tuna 7',     arms: ['A', 'B', 'C', 'D'] },
  25:  { name: 'Tuna 25',    arms: ['A', 'B', 'C', 'D'] },
  26:  { name: 'Tuna 26',    arms: ['A', 'B', 'C', 'D'] },
  27:  { name: 'Tuna 27',    arms: ['A', 'B', 'C', 'D'] },
  87:  { name: 'Tuna 87',    arms: ['A', 'B', 'C', 'D'] },
}

export function junctionName(id: number): string {
  return JUNCTION_DISPLAY[id]?.name ?? `Kavşak #${id}`
}

export function junctionArms(id: number): readonly string[] {
  return JUNCTION_DISPLAY[id]?.arms ?? DEFAULT_ARMS
}
