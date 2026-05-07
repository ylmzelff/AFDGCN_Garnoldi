# 🚀 Quick Start Guide - AFDGCN Garnoldi

Bu rehberde sistemi 5 dakikada ayağa kaldıracaksınız.

## 📋 Önkoşullar

- **Python 3.11+** (Check: `python --version`)
- **Node.js 18+** (Check: `node --version`)
- **pip** package manager
- **Git** (optional)

## ⚡ 1-Minute Setup (Fastest)

```bash
# Backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn backend.app.main:app --port 9001

# Frontend (new terminal)
cd frontend
npm install
npm run dev
# Open http://localhost:5000
```

**Login**: demo / demo123

---

## 📖 5-Minute Setup (Recommended)

### Step 1: Backend Configuration (2 min)

```bash
# 1.1: Create virtual environment
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate

# 1.2: Install dependencies
pip install -r requirements.txt

# 1.3: Create environment file
cp .env.example .env

# Edit .env if needed (default values work for local dev)
# nano .env  OR  code .env
```

### Step 2: Start Belediye API (Mock) (1 min)

```bash
# In a new terminal
source .venv/bin/activate
python kayseri_api.py
# Runs on http://localhost:9000
```

### Step 3: Start Phase API (1 min)

```bash
# In another terminal
source .venv/bin/activate
uvicorn backend.app.main:app --host 0.0.0.0 --port 9001 --reload

# Watch for:
# ✅ Kayseri API bağlantısı kuruldu
# ✅ Veritabanı hazır
# 👤 Demo kullanıcı: username=demo / password=demo123
# ✅ Phase API hazır — port 9001
```

### Step 4: Start Frontend (1 min)

```bash
# In a new terminal
cd frontend
npm install  # First time only
npm run dev

# Open: http://localhost:5000
```

### Step 5: Test the System

```bash
# 5.1: Login
# Username: demo
# Password: demo123

# 5.2: View Real-time Dashboard
# Select region (İldem/Tuna)
# Click on a junction to see details
# Watch phase recommendations update

# 5.3: Check API
# POST http://localhost:9001/auth/login
# {"username": "demo", "password": "demo123"}
```

---

## 🧪 Verification Checklist

- [ ] Backend running on `http://localhost:9001` ✅
- [ ] Frontend running on `http://localhost:5000` ✅
- [ ] Can login with demo/demo123 ✅
- [ ] Real-time dashboard loads ✅
- [ ] Junction data visible ✅
- [ ] Phase recommendations shown ✅
- [ ] API responding at `/api/v1/predict/regions` ✅

---

## 📚 Next Steps

### Testing the Prediction API

```bash
# 1. Get JWT token
curl -X POST http://localhost:9001/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"demo123"}' | jq '.access_token'

# 2. Use token in requests
TOKEN="your_token_here"

# Get region predictions
curl -X POST http://localhost:9001/api/v1/predict/region/ildem \
  -H "Authorization: Bearer $TOKEN"

# Get junction details
curl -X POST http://localhost:9001/api/v1/predict/junction/89 \
  -H "Authorization: Bearer $TOKEN"

# Get system status
curl -X GET http://localhost:9001/api/v1/predict/status \
  -H "Authorization: Bearer $TOKEN"

# List regions
curl -X GET http://localhost:9001/api/v1/predict/regions \
  -H "Authorization: Bearer $TOKEN"
```

### Exploring API Documentation

- **Swagger UI**: http://localhost:9001/docs
- **ReDoc**: http://localhost:9001/redoc

---

## 🔧 Troubleshooting

### Port Already in Use

```bash
# Backend (9001)
lsof -i :9001  # macOS/Linux
netstat -ano | findstr :9001  # Windows
# Kill the process and retry

# Frontend (5000)
# Vite automatically uses next available port
```

### Module Not Found Error

```bash
# Make sure you're in virtual environment
source .venv/bin/activate
# Then reinstall
pip install -r requirements.txt
```

### Belediye API Unreachable

```bash
# Start it in a separate terminal
python kayseri_api.py --host 0.0.0.0 --port 9000
# System works with or without it (uses fallback)
```

### Model Loading Error

```
⚠️ Model yüklenemedi → moving average kullanılıyor
# This is normal and intentional fallback
# System continues to work
```

### Database Connection Error

```bash
# Edit .env
DATABASE_URL=sqlite+aiosqlite:///./afdgcn.db
# Or start PostgreSQL and configure URL
```

---

## 🚀 Production Deployment

For production, see [SETUP.md](../SETUP.md) for:
- PostgreSQL setup
- Environment configuration
- Docker deployment
- Security hardening
- Performance tuning

---

## 📞 Support

If you encounter issues:

1. Check [SETUP.md](../SETUP.md) for detailed configuration
2. Review logs: Backend terminal should show detailed error messages
3. Verify all processes are running:
   ```bash
   # Backend
   ps aux | grep uvicorn
   # Frontend  
   ps aux | grep vite
   # Belediye API
   ps aux | grep kayseri_api
   ```

---

## 🎯 What's Working

✅ Real-time traffic predictions (AFDGCN model)
✅ Signal timing recommendations (Webster's method)
✅ WebSocket live updates
✅ JWT authentication
✅ Multi-region support (İldem, Tuna, Kızılırmak)
✅ React + TypeScript frontend
✅ Real-time charts and dashboards
✅ API documentation (Swagger/ReDoc)
✅ Caching and optimization
✅ Error handling and fallbacks

---

**Time to first prediction**: ~2 minutes ⚡

Enjoy! 🚦
