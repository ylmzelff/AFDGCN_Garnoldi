import type { Request, Response } from 'express'
import path from 'path'
import fs from 'fs'
import { v4 as uuidv4 } from 'uuid'
import multer from 'multer'
import { kayseriClient, pythonModel } from '../../predict/services'
import prisma from '../../database/prisma'
import { appState } from '../../common/services/app-state.service'
import { loadRegionConfigs, REGION_CONFIG } from '../../predict/services/real-time-predictor.service'

// ─────────────────────────────────────────────────────────────────────────────
// Multer — Model dosyası yükleme konfigürasyonu
// ─────────────────────────────────────────────────────────────────────────────

const PROJECT_ROOT = path.join(process.cwd(), '..')
const UPLOADS_DIR = path.join(PROJECT_ROOT, 'uploads', 'models')

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => {
    fs.mkdirSync(UPLOADS_DIR, { recursive: true })
    cb(null, UPLOADS_DIR)
  },
  filename: (_req, _file, cb) => cb(null, `${uuidv4()}.pth`),
})

export const modelUpload = multer({
  storage,
  fileFilter: (_req, file, cb) => {
    if (file.originalname.endsWith('.pth')) cb(null, true)
    else cb(new Error('Sadece .pth dosyaları kabul edilir'))
  },
  limits: { fileSize: 500 * 1024 * 1024 },
})

// ─────────────────────────────────────────────────────────────────────────────
// Şehir API Konfigürasyonu (ApiConfig)
// ─────────────────────────────────────────────────────────────────────────────

export const getApiConfigs = async (_req: Request, res: Response): Promise<void> => {
  const configs = await prisma.apiConfig.findMany({ orderBy: { city: 'asc' } })
  res.json(configs.map((c) => ({ ...c, password: '***' })))
}

export const upsertApiConfig = async (req: Request, res: Response): Promise<void> => {
  const { city, baseUrl, username, password } = req.body as {
    city?: string; baseUrl?: string; username?: string; password?: string
  }
  if (!city || !baseUrl || !username || !password) {
    res.status(400).json({ error: true, code: 'VALIDATION', message: 'city, baseUrl, username, password gerekli.' })
    return
  }
  const conf = await prisma.apiConfig.upsert({
    where: { city },
    update: { baseUrl, username, password },
    create: { city, baseUrl, username, password },
  })
  if (city === 'kayseri') {
    kayseriClient.updateCredentials(username, password)
    await kayseriClient.updateBaseUrl(baseUrl).catch(() => void 0)
  }
  res.json({ ...conf, password: '***' })
}

export const deleteApiConfig = async (req: Request, res: Response): Promise<void> => {
  const { city } = req.params as { city: string }
  const existing = await prisma.apiConfig.findUnique({ where: { city } })
  if (!existing) {
    res.status(404).json({ error: true, code: 'NOT_FOUND', message: `'${city}' şehri bulunamadı.` })
    return
  }
  await prisma.apiConfig.delete({ where: { city } })
  res.json({ status: 'deleted', city })
}

// ─────────────────────────────────────────────────────────────────────────────
// Bölge Konfigürasyonu (RegionConfig)
// ─────────────────────────────────────────────────────────────────────────────

export const getRegionConfigs = async (req: Request, res: Response): Promise<void> => {
  const city = req.query['city'] as string | undefined
  const configs = await prisma.regionConfig.findMany({
    where: city ? { city } : undefined,
    orderBy: [{ city: 'asc' }, { region: 'asc' }],
  })
  res.json(configs)
}

export const upsertRegionConfig = async (req: Request, res: Response): Promise<void> => {
  const { city, region, description, junctionIds, useModel } = req.body as {
    city?: string; region?: string; description?: string; junctionIds?: number[]; useModel?: boolean
  }
  if (!city || !region || !Array.isArray(junctionIds)) {
    res.status(400).json({ error: true, code: 'VALIDATION', message: 'city, region, junctionIds (array) gerekli.' })
    return
  }
  const conf = await prisma.regionConfig.upsert({
    where: { city_region: { city, region } },
    update: { description: description ?? '', junctionIds, useModel: useModel ?? false },
    create: { city, region, description: description ?? '', junctionIds, useModel: useModel ?? false },
  })
  REGION_CONFIG[region] = { city, junctionIds, useModel: useModel ?? false, description: description ?? '' }
  res.json(conf)
}

export const deleteRegionConfig = async (req: Request, res: Response): Promise<void> => {
  const { city, region } = req.params as { city: string; region: string }
  const existing = await prisma.regionConfig.findUnique({ where: { city_region: { city, region } } })
  if (!existing) {
    res.status(404).json({ error: true, code: 'NOT_FOUND', message: `'${city}/${region}' bulunamadı.` })
    return
  }
  await prisma.regionConfig.delete({ where: { city_region: { city, region } } })
  delete REGION_CONFIG[region]
  res.json({ status: 'deleted', city, region })
}

// ─────────────────────────────────────────────────────────────────────────────
// Model Versiyonları (ModelVersion)
// ─────────────────────────────────────────────────────────────────────────────

export const listModelVersions = async (req: Request, res: Response): Promise<void> => {
  const city = req.query['city'] as string | undefined
  const region = req.query['region'] as string | undefined
  const versions = await prisma.modelVersion.findMany({
    where: { ...(city ? { city } : {}), ...(region ? { region } : {}) },
    orderBy: { uploadedAt: 'desc' },
    select: {
      id: true, name: true, description: true, city: true, region: true,
      filePath: true, numNodes: true, lag: true, horizon: true,
      scalerMean: true, scalerStd: true, isActive: true, uploadedAt: true,
    },
  })
  res.json(versions)
}

export const uploadModelVersion = async (req: Request, res: Response): Promise<void> => {
  if (!req.file) {
    res.status(400).json({ error: true, code: 'NO_FILE', message: '.pth dosyası gerekli.' })
    return
  }
  const { name, description, city, region, numNodes, lag, horizon, scalerMean, scalerStd } = req.body as Record<string, string>
  if (!name || !city || !region || !numNodes) {
    fs.unlinkSync(req.file.path)
    res.status(400).json({ error: true, code: 'VALIDATION', message: 'name, city, region, numNodes gerekli.' })
    return
  }
  const version = await prisma.modelVersion.create({
    data: {
      name, description: description ?? '', city, region,
      filePath: req.file.path,
      numNodes: Number(numNodes),
      lag: Number(lag ?? 1),
      horizon: Number(horizon ?? 1),
      scalerMean: Number(scalerMean ?? 0),
      scalerStd: Number(scalerStd ?? 1),
      isActive: false,
    },
  })
  res.status(201).json(version)
}

export const activateModelVersion = async (req: Request, res: Response): Promise<void> => {
  const { id } = req.params as { id: string }
  const version = await prisma.modelVersion.findUnique({ where: { id } })
  if (!version) {
    res.status(404).json({ error: true, code: 'NOT_FOUND', message: `Model versiyonu '${id}' bulunamadı.` })
    return
  }
  const loadResult = await pythonModel.loadModel({
    path: version.filePath,
    numNodes: version.numNodes,
    lag: version.lag,
    horizon: version.horizon,
    scalerMean: version.scalerMean,
    scalerStd: version.scalerStd,
  })
  if (!loadResult.success) {
    res.status(500).json({ error: true, code: 'MODEL_LOAD_FAILED', message: loadResult.message })
    return
  }
  await prisma.modelVersion.updateMany({
    where: { city: version.city, region: version.region, isActive: true, id: { not: id } },
    data: { isActive: false },
  })
  const activated = await prisma.modelVersion.update({ where: { id }, data: { isActive: true } })
  res.json({ status: 'activated', version: activated, modelLoad: loadResult })
}

export const deleteModelVersion = async (req: Request, res: Response): Promise<void> => {
  const { id } = req.params as { id: string }
  const version = await prisma.modelVersion.findUnique({ where: { id } })
  if (!version) {
    res.status(404).json({ error: true, code: 'NOT_FOUND', message: `Model versiyonu '${id}' bulunamadı.` })
    return
  }
  if (version.isActive) {
    res.status(400).json({ error: true, code: 'ACTIVE_MODEL', message: 'Aktif model versiyonu silinemez. Önce başka bir versiyonu aktive edin.' })
    return
  }
  if (version.filePath.includes(`${path.sep}uploads${path.sep}`)) {
    fs.unlink(version.filePath, () => void 0)
  }
  await prisma.modelVersion.delete({ where: { id } })
  res.json({ status: 'deleted', id })
}

// ─────────────────────────────────────────────────────────────────────────────
// Kullanıcı CRUD
// ─────────────────────────────────────────────────────────────────────────────

export const getUsers = async (_req: Request, res: Response): Promise<void> => {
  const users = await prisma.user.findMany({
    select: { id: true, username: true, disabled: true, createdAt: true },
    orderBy: { createdAt: 'asc' },
  })
  res.json({ users, total: users.length })
}

export const deleteUser = async (req: Request, res: Response): Promise<void> => {
  const { username } = req.params as { username: string }
  const user = await prisma.user.findUnique({ where: { username } })
  if (!user) {
    res.status(404).json({ error: true, code: 'NOT_FOUND', message: `Kullanıcı '${username}' bulunamadı.` })
    return
  }
  await prisma.user.delete({ where: { username } })
  appState.usersDb.delete(username)
  res.json({ status: 'deleted', username })
}

export const toggleUser = async (req: Request, res: Response): Promise<void> => {
  const { username } = req.params as { username: string }
  const { disabled } = req.body as { disabled?: boolean }
  if (typeof disabled !== 'boolean') {
    res.status(400).json({ error: true, code: 'VALIDATION', message: 'disabled (boolean) gerekli.' })
    return
  }
  const user = await prisma.user.findUnique({ where: { username } })
  if (!user) {
    res.status(404).json({ error: true, code: 'NOT_FOUND', message: `Kullanıcı '${username}' bulunamadı.` })
    return
  }
  const updated = await prisma.user.update({ where: { username }, data: { disabled } })
  const cached = appState.usersDb.get(username)
  if (cached) appState.usersDb.set(username, { ...cached, disabled })
  res.json({ status: 'updated', username, disabled: updated.disabled })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tahmin Geçmişi
// ─────────────────────────────────────────────────────────────────────────────

export const getPredictions = async (req: Request, res: Response): Promise<void> => {
  const limit = Math.min(Number(req.query['limit'] ?? 50), 500)
  const offset = Number(req.query['offset'] ?? 0)
  const region = req.query['region'] as string | undefined
  const [predictions, total] = await prisma.$transaction([
    prisma.phasePrediction.findMany({
      where: region ? { region } : undefined,
      orderBy: { createdAt: 'desc' },
      take: limit,
      skip: offset,
      select: {
        id: true, createdAt: true, region: true, city: true,
        timeLabel: true, predictionSource: true, kayseriApiStatus: true,
        junctionCount: true, totalVehicles: true,
      },
    }),
    prisma.phasePrediction.count({ where: region ? { region } : undefined }),
  ])
  res.json({ predictions, total, limit, offset })
}

// ─────────────────────────────────────────────────────────────────────────────
// Model Olayları Geçmişi
// ─────────────────────────────────────────────────────────────────────────────

export const getModelEvents = async (req: Request, res: Response): Promise<void> => {
  const limit = Math.min(Number(req.query['limit'] ?? 50), 200)
  const offset = Number(req.query['offset'] ?? 0)
  const [events, total] = await prisma.$transaction([
    prisma.modelEvent.findMany({ orderBy: { createdAt: 'desc' }, take: limit, skip: offset }),
    prisma.modelEvent.count(),
  ])
  res.json({ events, total, limit, offset })
}
