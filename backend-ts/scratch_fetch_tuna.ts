/**
 * Tuna bolgesi icin AusTKM API'sinden gecmis veri ceker (2026-03-01 - 2026-07-31).
 * Tek seferlik veri toplama scripti - egitim verisi hazirlamak icin.
 * Calistirma: npx tsx scratch_fetch_tuna.ts
 */
import 'dotenv/config'
import axios from 'axios'
import https from 'https'
import fs from 'fs'

const BASE_URL = (process.env.AUSTM_API_URL ?? '').replace(/\/$/, '')
const TOKEN = process.env.AUSTM_API_TOKEN ?? ''
const BOLGE_ADI = 'TUNA'

const START = '2026-03-01'
const END = '2026-07-31'
const OUT_PATH = '../scratch_tuna_history.json'

const http = axios.create({
  baseURL: BASE_URL,
  timeout: 30_000,
  httpsAgent: new https.Agent({ rejectUnauthorized: false }),
  headers: { Authorization: `Bearer ${TOKEN}`, Accept: 'application/json' },
})

function timeToSlotIndex(hhmm: string): number {
  const colonIdx = hhmm.indexOf(':')
  const h = parseInt(hhmm.slice(0, colonIdx), 10)
  const m = parseInt(hhmm.slice(colonIdx + 1), 10)
  return h * 6 + Math.floor(m / 10)
}

interface AusTkmSensorItem {
  intersectionId: number
  edgeDirection: string
  edgeName?: string
  saatlikVeriler: Record<string, number>
}
interface AusTkmResponse {
  success: boolean
  toplamKayit?: number
  data: AusTkmSensorItem[]
}

function dateRange(start: string, end: string): string[] {
  const dates: string[] = []
  const cur = new Date(start + 'T00:00:00Z')
  const endDate = new Date(end + 'T00:00:00Z')
  while (cur <= endDate) {
    dates.push(cur.toISOString().slice(0, 10))
    cur.setUTCDate(cur.getUTCDate() + 1)
  }
  return dates
}

async function fetchDay(tarih: string): Promise<Record<number, Record<string, number[]>> | null> {
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const resp = await http.get<AusTkmResponse>('/api/SensorVerileri', {
        params: { bolgeAdi: BOLGE_ADI, tarih },
      })
      if (!resp.data.success || !Array.isArray(resp.data.data) || resp.data.data.length === 0) {
        return null
      }
      const byJunction: Record<number, Record<string, number[]>> = {}
      for (const item of resp.data.data) {
        const jid = item.intersectionId
        const dir = item.edgeDirection.trim().toUpperCase()
        if (!byJunction[jid]) byJunction[jid] = {}
        const slots = new Array<number>(144).fill(0)
        for (const [timeKey, count] of Object.entries(item.saatlikVeriler ?? {})) {
          const idx = timeToSlotIndex(timeKey)
          if (idx >= 0 && idx < 144) slots[idx] = Number(count) || 0
        }
        byJunction[jid][dir] = slots
      }
      return byJunction
    } catch (err) {
      if (attempt === 2) {
        console.warn(`  ${tarih} -> HATA: ${err instanceof Error ? err.message : String(err)}`)
        return null
      }
      await new Promise((r) => setTimeout(r, 1000))
    }
  }
  return null
}

async function main() {
  const dates = dateRange(START, END)
  console.log(`Taranacak gun sayisi: ${dates.length} (${START} -> ${END})`)

  const results: Record<string, Record<number, Record<string, number[]>>> = {}
  let found = 0

  for (let i = 0; i < dates.length; i++) {
    const d = dates[i]
    const dayData = await fetchDay(d)
    if (dayData && Object.keys(dayData).length > 0) {
      results[d] = dayData
      found++
      console.log(`[${i + 1}/${dates.length}] ${d} -> OK (${Object.keys(dayData).length} kavsak)`)
    } else {
      console.log(`[${i + 1}/${dates.length}] ${d} -> veri yok`)
    }
    await new Promise((r) => setTimeout(r, 100))
  }

  fs.writeFileSync(OUT_PATH, JSON.stringify(results))
  console.log(`\nTaranan gun: ${dates.length}, veri bulunan gun: ${found}`)
  console.log(`Kaydedildi: ${OUT_PATH}`)
}

main().catch((e) => {
  console.error('SCRIPT HATASI:', e)
  process.exit(1)
})
