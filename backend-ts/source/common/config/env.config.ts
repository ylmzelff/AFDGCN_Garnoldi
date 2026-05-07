import "dotenv/config";

function requireEnv(key: string, fallback: string): string {
  return process.env[key] ?? fallback;
}

export const env = {
  port: Number(process.env["PORT"]) || 9001,
  nodeEnv: requireEnv("NODE_ENV", "development"),
  logLevel: requireEnv("LOG_LEVEL", "info"),
  apiVersion: requireEnv("API_VERSION", "1.0.0"),

  // JWT
  jwtSecret: requireEnv(
    "JWT_SECRET",
    "kayseri-traffic-api-secret-key-2026-change-in-production",
  ),
  jwtExpiresInSeconds: Number(process.env["JWT_EXPIRES_IN_SECONDS"]) || 86400,

  // Veritabanı
  databaseUrl: requireEnv(
    "DATABASE_URL",
    "postgresql://postgres:postgres123@localhost:5432/afdgcn",
  ),

  // Kayseri Belediye API
  kayseriApiUrl: requireEnv("KAYSERI_API_URL", "http://localhost:9000"),
  kayseriUsername: requireEnv("KAYSERI_USERNAME", "demo"),
  kayseriPassword: requireEnv("KAYSERI_PASSWORD", "demo123"),

  // Python AFDGCN Model Sunucusu
  pythonModelUrl: requireEnv("PYTHON_MODEL_URL", "http://localhost:9002"),

  // WebSocket yayın aralığı (ms)
  wsBroadcastIntervalMs:
    Number(process.env["WS_BROADCAST_INTERVAL_MS"]) || 600_000,

  // Arka plan çekme aralığı (ms)
  bgFetchIntervalMs: Number(process.env["BG_FETCH_INTERVAL_MS"]) || 60_000,
} as const;
