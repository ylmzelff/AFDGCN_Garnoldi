import 'dotenv/config'
import 'module-alias/register'

import http from 'http'
import path from 'path'
import { WebSocketServer } from 'ws'

import app from './source/app'
import { env } from './source/common/config'
import { appState } from './source/common/services/app-state.service'
import { setDbEnabled } from './source/common/services/db-logger.service'
import { kayseriClient, backgroundFetcher, pythonModel } from './source/predict/services'
import { phasesService } from './source/phases/services'
import { websocketService } from './source/common/services/websocket.service'
import { initWebSocketServer } from './source/websocket/controllers/websocket.controller'
import { authService } from './source/auth/services'
import prisma from './source/database/prisma'
import { logModelEvent } from './source/common/services/db-logger.service'
import { loadRegionConfigs } from './source/predict/services/real-time-predictor.service'

// ─────────────────────────────────────────────────────────────────────────────
// HTTP + WebSocket Sunucusu
// ─────────────────────────────────────────────────────────────────────────────

const httpServer = http.createServer(app)

const wss = new WebSocketServer({ server: httpServer, path: '/ws/live' })
initWebSocketServer(wss)

// ─────────────────────────────────────────────────────────────────────────────
// Başlangıç Rutini
// ─────────────────────────────────────────────────────────────────────────────

async function startup(): Promise<void> {
  console.info('🚀 Phase API (TypeScript) başlatılıyor...')

  // 1. Demo kullanıcı oluştur (geçici önbellek — DB hazır olunca upsert edilecek)
  const demoHashedPw = authService.hashPassword('demo123')
  appState.usersDb.set('demo', { username: 'demo', hashedPassword: demoHashedPw, disabled: false })
  console.info('👤 Demo kullanıcı: username=demo / password=demo123')

  // 2. Kayseri API bağlantısı
  try {
    await kayseriClient.ensureAuthenticated()
    console.info('✅ Kayseri API bağlantısı kuruldu')
  } catch (err) {
    console.warn('⚠️  Kayseri API şu an ulaşılamıyor:', err)
  }

  // 3. Veritabanı bağlantısı
  try {
    await prisma.$connect()
    setDbEnabled(true)
    appState.dbEnabled = true

    // Demo kullanıcıyı DB'ye upsert et
    await prisma.user.upsert({
      where: { username: 'demo' },
      update: {},
      create: { username: 'demo', hashedPassword: demoHashedPw, disabled: false },
    })

    // DB'deki kullanıcıları memory'e yükle
    const dbUsers = await prisma.user.findMany()
    for (const u of dbUsers) {
      appState.usersDb.set(u.username, {
        username: u.username,
        hashedPassword: u.hashedPassword,
        disabled: u.disabled,
      })
    }
    console.info(`✅ Veritabanı hazır (${dbUsers.length} kullanıcı yüklendi)`)

    // ── Şehir API konfigürasyonu seed ──────────────────────────────────────
    await prisma.apiConfig.upsert({
      where: { city: 'kayseri' },
      update: {},
      create: { city: 'kayseri', baseUrl: env.kayseriApiUrl, username: env.kayseriUsername, password: env.kayseriPassword },
    })
    const cityConf = await prisma.apiConfig.findUnique({ where: { city: 'kayseri' } })
    if (cityConf?.isActive) {
      kayseriClient.updateCredentials(cityConf.username, cityConf.password)
      await kayseriClient.updateBaseUrl(cityConf.baseUrl).catch(() => void 0)
      console.info('✅ Kayseri API config DB\'den yüklendi')
    }

    // ── Bölge konfigürasyonu seed ──────────────────────────────────────────
    const defaultRegions = [
      { city: 'kayseri', region: 'ildem', bolgeAdi: '\u0130LDEM', junctionIds: [89, 187, 95, 121, 184, 188, 117, 192, 194], useModel: true, description: 'İldem (AFDGCN)' },
      { city: 'kayseri', region: 'tuna', bolgeAdi: 'TUNA', junctionIds: [5, 3, 87, 25, 26, 27, 7], useModel: false, description: 'Tuna (Moving Average)' },
      { city: 'kayseri', region: 'kizilirmak', bolgeAdi: 'KIZILIRMAK', junctionIds: [130, 38, 176], useModel: false, description: 'Kızılırmak (Moving Average)' },
    ]
    for (const r of defaultRegions) {
      await prisma.regionConfig.upsert({
        where: { city_region: { city: r.city, region: r.region } },
        update: { bolgeAdi: r.bolgeAdi },
        create: { ...r, junctionIds: r.junctionIds },
      })
    }
    const regionRows = await prisma.regionConfig.findMany({ where: { isActive: true } })
    loadRegionConfigs(regionRows.map((r) => ({
      city: r.city,
      region: r.region,
      junctionIds: r.junctionIds as number[],
      useModel: r.useModel,
      description: r.description,
      bolgeAdi: r.bolgeAdi,
    })))

    // ── Model versiyon seed ────────────────────────────────────────────────
    // Eğer DB'de model yoksa veya weights null ise disk'teki dummy .pth ile seed et
    {
      const projectRoot = path.join(__dirname, '..')
      const dummyPthPath = path.join(projectRoot, 'saved_models', 'kayseri_ildem_34_dummy.pth')
      const hasDummy = require('fs').existsSync(dummyPthPath)

      const existingModel = await prisma.modelVersion.findUnique({
        where: { city_region_name: { city: 'kayseri', region: 'ildem', name: 'kayseri_ildem_v1' } },
        select: { id: true, weights: true },
      })

      if (!existingModel && hasDummy) {
        // İlk kez: oluştur
        const weights = require('fs').readFileSync(dummyPthPath) as Buffer
        await prisma.modelVersion.create({
          data: {
            name: 'kayseri_ildem_v1',
            description: 'Kayseri İldem bölgesi AFDGCN modeli (v1, seed)',
            city: 'kayseri', region: 'ildem',
            filePath: '',
            weights,
            numNodes: 34, lag: 1, horizon: 1,
            scalerMean: 28.53, scalerStd: 38.72,
            isActive: true,
          },
        })
        console.info('🌱 Seed model DB\'ye yüklendi (kayseri_ildem_v1, 34 node)')
      } else if (existingModel && !existingModel.weights && hasDummy) {
        // Eski kayıt: weights ekle + numNodes güncelle
        const weights = require('fs').readFileSync(dummyPthPath) as Buffer
        await prisma.modelVersion.update({
          where: { id: existingModel.id },
          data: { weights, numNodes: 34, lag: 1, filePath: '' },
        })
        console.info('🔄 Mevcut model kaydı 34-node dummy weights ile güncellendi')
      }
    }

    // ── Aktif model DB bytes'ından Python sunucusuna yükle ─────────────────
    const activeModel = await prisma.modelVersion.findFirst({
      where: { city: 'kayseri', region: 'ildem', isActive: true },
      select: {
        id: true, name: true, weights: true, filePath: true,
        numNodes: true, lag: true, horizon: true, scalerMean: true, scalerStd: true,
      },
    })
    if (activeModel) {
      const loadResult = await pythonModel.loadModelFromBytes({
        weights: activeModel.weights,
        filePath: activeModel.filePath,
        numNodes: activeModel.numNodes,
        lag: activeModel.lag,
        horizon: activeModel.horizon,
        scalerMean: activeModel.scalerMean,
        scalerStd: activeModel.scalerStd,
      })
      console.info(`🤖 Model yükleme (${activeModel.name}): ${loadResult.message}`)
    }

    // Model startup event
    void logModelEvent('startup', '', 'TypeScript backend başlatıldı')
  } catch (err) {
    console.warn('⚠️  Veritabanına bağlanılamadı, DB logging devre dışı:', err)
    
    // DB kapalıysa bile sistemin çalışabilmesi için varsayılan bölgeleri memory'e yükle
    const defaultRegions = [
      { city: 'kayseri', region: 'ildem', bolgeAdi: '\u0130LDEM', junctionIds: [89, 187, 95, 121, 184, 188, 117, 192, 194], useModel: true, description: 'İldem (AFDGCN)' },
      { city: 'kayseri', region: 'tuna', bolgeAdi: 'TUNA', junctionIds: [5, 3, 87, 25, 26, 27, 7], useModel: false, description: 'Tuna (Moving Average)' },
      { city: 'kayseri', region: 'kizilirmak', bolgeAdi: 'KIZILIRMAK', junctionIds: [130, 38, 176], useModel: false, description: 'Kızılırmak (Moving Average)' },
    ]
    loadRegionConfigs(defaultRegions)
    console.info('✅ Varsayılan bölge konfigürasyonları yüklendi (Fallback)')
  }

  // 4. Arka plan çekici başlat
  backgroundFetcher.start(env.bgFetchIntervalMs)
  console.info(`✅ Arka plan çekici başlatıldı (interval=${env.bgFetchIntervalMs}ms)`)

  // 5. WebSocket yayın döngüsü
  const broadcastLoop = setInterval(() => {
    void phasesService
      .pushAllRegions((payload) => websocketService.broadcast(payload))
      .catch((err) => console.error('[broadcast] Hata:', err))
  }, env.wsBroadcastIntervalMs)

  // Graceful shutdown için referansı tut
  process.on('SIGINT', () => void shutdown(broadcastLoop))
  process.on('SIGTERM', () => void shutdown(broadcastLoop))

  console.info(`✅ Phase API hazır — port ${env.port}`)
}

// ─────────────────────────────────────────────────────────────────────────────
// Kapatma Rutini
// ─────────────────────────────────────────────────────────────────────────────

async function shutdown(broadcastLoop: ReturnType<typeof setInterval>): Promise<void> {
  console.info('👋 Phase API kapatılıyor...')
  clearInterval(broadcastLoop)
  backgroundFetcher.stop()
  wss.close()
  httpServer.close()
  await prisma.$disconnect()
  process.exit(0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Sunucuyu Başlat
// ─────────────────────────────────────────────────────────────────────────────

httpServer.listen(env.port, async () => {
  await startup()
  console.info(`🌐 http://localhost:${env.port}`)
  console.info(`🔌 ws://localhost:${env.port}/ws/live`)
})

export default app
