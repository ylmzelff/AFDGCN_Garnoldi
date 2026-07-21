/**
 * Kavşak görüntüleme metadatası.
 * Yeni şehir/bölge kavşakları buraya eklenir; backend kodu değişmez.
 * Anahtar "{city}:{id}" şeklindedir — farklı şehirlerin aynı kavşak ID'sini
 * kullanması (örn. Kayseri Tuna ve Sivas Merkez) çakışmaz.
 * Bilinmeyen kavşaklar için varsayılan isim ve kol dizisi kullanılır.
 */

export const DEFAULT_ARMS: readonly string[] = ['A', 'B', 'C', 'D']

export interface JunctionMeta {
  name: string
  arms: readonly string[]
}

function key(city: string, id: number): string {
  return `${city.toLowerCase()}:${id}`
}

export const JUNCTION_DISPLAY: Record<string, JunctionMeta> = {
  // ── İldem (Kayseri) ────────────────────────────────────────────────────────
  'kayseri:89':  { name: 'Gesi',       arms: ['A', 'B', 'C', 'D'] },
  'kayseri:95':  { name: 'Beyazşehir', arms: ['A', 'B', 'C', 'D'] },
  'kayseri:117': { name: 'İldem 3',    arms: ['A', 'C', 'D'] },
  'kayseri:121': { name: 'Toki',       arms: ['A', 'B', 'C', 'D'] },
  'kayseri:184': { name: 'İldem 1',    arms: ['A', 'B', 'D'] },
  'kayseri:187': { name: 'Serkent',    arms: ['A', 'B', 'C', 'D'] },
  'kayseri:188': { name: 'İldem 2',    arms: ['A', 'B', 'C', 'D'] },
  'kayseri:192': { name: 'İldem 4',    arms: ['A', 'B', 'C', 'D'] },
  'kayseri:194': { name: 'İldem 5',    arms: ['A', 'B', 'C', 'D'] },
  // ── Tuna (Kayseri) ─────────────────────────────────────────────────────────
  'kayseri:3':   { name: 'Tuna 3',     arms: ['A', 'B', 'C', 'D'] },
  'kayseri:5':   { name: 'Tuna 5',     arms: ['A', 'B', 'C', 'D'] },
  'kayseri:7':   { name: 'Tuna 7',     arms: ['A', 'B', 'C', 'D'] },
  'kayseri:25':  { name: 'Tuna 25',    arms: ['A', 'B', 'C', 'D'] },
  'kayseri:26':  { name: 'Tuna 26',    arms: ['A', 'B', 'C', 'D'] },
  'kayseri:27':  { name: 'Tuna 27',    arms: ['A', 'B', 'C', 'D'] },
  'kayseri:87':  { name: 'Tuna 87',    arms: ['A', 'B', 'C', 'D'] },
  // ── Merkez (Sivas) ─────────────────────────────────────────────────────────
  'sivas:9': { name: 'TSO', arms: ['A', 'B', 'D'] },
  'sivas:3': { name: 'Evciler', arms: ['A', 'B'] },
}

export function junctionName(city: string, id: number): string {
  return JUNCTION_DISPLAY[key(city, id)]?.name ?? `Kavşak #${id}`
}

export function junctionArms(city: string, id: number): readonly string[] {
  return JUNCTION_DISPLAY[key(city, id)]?.arms ?? DEFAULT_ARMS
}
