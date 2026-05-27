# AFDGCN Trafik Faz Tahmin Sistemi

Kayseri trafik kavşakları için gerçek zamanlı faz tahmini yapan tam yığın uygulama.  
Model: **AFDGCN** (Adaptive Frequency-Domain Graph Convolutional Network) + Arnoldi iterasyonu.

---

## Mimariye Genel Bakış

| Katman | Teknoloji | Port |
|--------|-----------|------|
| Python model sunucusu | FastAPI + PyTorch | `9002` |
| Backend API | Node.js / TypeScript / Express | `9001` |
| Frontend | React + Vite + TailwindCSS | `5173` |
| Prisma Studio | Prisma ORM görsel arayüzü | `5555` |
| Veritabanı | PostgreSQL | `5432` |

---

## Ön Koşullar

- **Node.js** ≥ 18
- **Python** ≥ 3.10 (sanal ortam: `.venv/`)
- **PostgreSQL** çalışıyor olmalı (`localhost:5432`, veritabanı adı: `afdgcn`)

---

## Servisleri Başlatma

### 1. Python Model Sunucusu (port 9002)

```powershell
# Proje kökünden
.venv\Scripts\python.exe model_server.py
```

### 2. Backend API (port 9001)

```powershell
cd backend-ts
npm install        # ilk kurulumda
npm run dev        # nodemon ile otomatik yeniden başlatma
```

buradaki id değişebilir 
Get-NetTCPConnection -LocalPort 9001 | Select-Object LocalAddress,LocalPort,State,OwningProcess
Get-Process -Id 39424 
Stop-Process -Id 39424 -Force

Backend hazır olduğunda terminalde şu çıktıyı görürsünüz:

```
✅ Phase API hazır — port 9001
🌐 http://localhost:9001
🔌 ws://localhost:9001/ws/live
```

### 3. Frontend (port 5173)

```powershell
cd frontend
npm install        # ilk kurulumda
npm run dev
```

Tarayıcıda açın: **http://localhost:5173**

### 4. Prisma Studio (port 5555)

Veritabanını görsel olarak incelemek için:

```powershell
cd backend-ts
npm run db:studio
```

Tarayıcıda açın: **http://localhost:5555**

---

## Veritabanı İşlemleri

```powershell
cd backend-ts

# Migration çalıştır (geliştirme)
npm run db:migrate

# Migration uygula (üretim)
npm run db:deploy

# Prisma client yeniden üret
npm run db:generate

# Veritabanını sıfırla (DİKKAT: tüm veri silinir)
npm run db:reset
```

### Varsayılan `.env` değerleri

```env
PORT=9001
DATABASE_URL=postgresql://postgres:postgres123@localhost:5432/afdgcn
JWT_SECRET=kayseri-traffic-api-secret-key-2026-change-in-production
PYTHON_MODEL_URL=http://localhost:9002
KAYSERI_API_URL=http://localhost:9000
```

---

## Demo Kullanıcı

Backend ilk başladığında otomatik oluşturulur:

| Alan | Değer |
|------|-------|
| Kullanıcı adı | `demo` |
| Şifre | `demo123` |

---

## Tüm Servisleri Tek Seferde Başlatmak (PowerShell)

Dört ayrı terminal penceresinde sırayla çalıştırın:

```powershell
# Terminal 1 — Python model
.venv\Scripts\python.exe model_server.py

# Terminal 2 — Backend
cd backend-ts ; npm run dev

# Terminal 3 — Frontend
cd frontend ; npm run dev

# Terminal 4 — Prisma Studio
cd backend-ts ; npm run db:studio
```

---

## Proje Yapısı

```
AFDGCN_Garnoldi/
├── model_server.py          # FastAPI Python model sunucusu
├── model/AFDGCN.py          # Model mimarisi
├── saved_models/            # Eğitilmiş model ağırlıkları (.pth)
├── conf/                    # Şehir/bölge konfigürasyon dosyaları
├── backend-ts/              # TypeScript Express backend
│   ├── server.ts
│   ├── source/              # Route, controller, service katmanları
│   └── prisma/              # Prisma şeması ve migration'lar
├── frontend/                # React + Vite SPA
│   └── src/
├── lib/                     # Veri yükleme, normalizasyon yardımcıları
└── data/                    # Ham veri setleri (Kayseri, Konya, Sivas, PEMS)
```
