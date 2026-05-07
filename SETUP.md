# 🚦 AFDGCN Garnoldi - Real-Time Traffic Prediction System

Production-ready end-to-end AI traffic prediction system using AFDGCN model.

## 🎯 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Belediye Sensor API                        │
│        (https://ausapi.kayseri.bel.tr:18880/...)            │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────▼──────────────┐
        │   Backend (FastAPI 9001)  │
        │                           │
        │  ┌─────────────────────┐  │
        │  │ KayseriAPIClient    │  │ ◄── Fetches data from Belediye
        │  └─────────────────────┘  │
        │                           │
        │  ┌─────────────────────┐  │
        │  │ AFDGCN Model        │  │ ◄── Deep learning predictions
        │  │ (34-node GNN)       │  │
        │  └─────────────────────┘  │
        │                           │
        │  ┌─────────────────────┐  │
        │  │ Phase Calculator    │  │ ◄── Webster-based timing
        │  │ (signal timing)     │  │
        │  └─────────────────────┘  │
        │                           │
        │  ┌─────────────────────┐  │
        │  │ Background Fetcher  │  │ ◄── Continuous data polling
        │  │ (60s interval)      │  │
        │  └─────────────────────┘  │
        │                           │
        │  ┌─────────────────────┐  │
        │  │ Cache Layer (300s)  │  │ ◄── Performance optimization
        │  └─────────────────────┘  │
        │                           │
        │  ┌─────────────────────┐  │
        │  │ WebSocket Broadcast │  │ ◄── Real-time push to clients
        │  │ (10min interval)    │  │
        │  └─────────────────────┘  │
        └───┬──────┬──────┬─────────┘
            │      │      │
   HTTP API │      │      └─── WebSocket Push
            │      │           (Phase Updates)
    /api/v1/│      │
    predict/│      │
   /region/ │      │
            ▼      ▼
        ┌──────────────────────┐
        │  Frontend (React)    │
        │  (Vite 5000)         │
        │                      │
        │  ┌────────────────┐  │
        │  │ Real-time      │  │
        │  │ Dashboard      │  │
        │  │                │  │
        │  │ ✓ Live Charts  │  │
        │  │ ✓ Phase Info   │  │
        │  │ ✓ Status       │  │
        │  └────────────────┘  │
        └──────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL (optional, SQLite fallback works)
- CUDA 11.8+ (optional, CPU works)

### Backend Setup

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file
cat > .env << 'EOF'
# Backend Config
KAYSERI_API_URL=http://localhost:9000
KAYSERI_USERNAME=demo
KAYSERI_PASSWORD=demo123

# Database (PostgreSQL)
DATABASE_URL=postgresql+asyncpg://postgres:postgres123@localhost:5432/afdgcn
# Or use SQLite for testing:
# DATABASE_URL=sqlite+aiosqlite:///./afdgcn.db

# JWT & Security
JWT_SECRET_KEY=kayseri-traffic-api-secret-key-2026-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# WebSocket
WS_BROADCAST_INTERVAL=600

# Logging
LOG_LEVEL=INFO
EOF

# 4. Start Belediye API (port 9000) - in another terminal
# This serves as the data source
# If you need to mock: python kayseri_api.py

# 5. Start Phase API (port 9001)
uvicorn backend.app.main:app --host 0.0.0.0 --port 9001 --reload
```

### Frontend Setup

```bash
# 1. Navigate to frontend
cd frontend

# 2. Create .env file
cat > .env.local << 'EOF'
VITE_API_URL=http://localhost:9001
VITE_WS_URL=ws://localhost:9001
EOF

# 3. Install dependencies
npm install

# 4. Start dev server
npm run dev
# Open http://localhost:5000
```

## 📊 Available Endpoints

### Authentication
```bash
POST /auth/login
# Request: {"username": "demo", "password": "demo123"}
# Response: {"access_token": "...", "token_type": "bearer", "expires_in": 86400}
```

### Real-time Predictions
```bash
# Get full region prediction
POST /api/v1/predict/region/ildem
# Response: {
#   "region": "ildem",
#   "predictions": {89: {"A": 25.5, "B": 18.3, ...}, ...},
#   "phases": {89: {"cycle_time": 65, ...}, ...},
#   "source": "AFDGCN" | "moving_average",
#   ...
# }

# Get specific junction details
POST /api/v1/predict/junction/89
# Response: {
#   "junction_id": 89,
#   "junction_name": "Gesi",
#   "arms": {
#     "A": {
#       "vehicle_count": 25.5,
#       "green": 18,
#       "yellow": 3,
#       "red": 44,
#       ...
#     },
#     ...
#   },
#   ...
# }

# Get system status
GET /api/v1/predict/status
# Response: {
#   "cache": {"size": 5, "ttl_seconds": 300},
#   "model": {"loaded": true, "num_nodes": 34, "fallback_active": false},
#   "kayseri_api": {"authenticated": true, "token_valid": true}
# }

# List available regions
GET /api/v1/predict/regions
# Response: {
#   "regions": [
#     {
#       "name": "ildem",
#       "description": "İldem (AFDGCN)",
#       "junction_count": 9,
#       "use_model": true
#     },
#     ...
#   ]
# }
```

### Legacy Phase Endpoints (Backward Compatible)
```bash
GET /phases/ildem    # İldem region phases
GET /phases/tuna     # Tuna region phases
```

### WebSocket Real-time Push
```
ws://localhost:9001/ws/live?token=<JWT>

Message Format:
{
  "type": "phase_update",
  "regions": {
    "ildem": { ...full region response... },
    "tuna": { ...full region response... }
  }
}

Auto-pushes every 10 minutes (configurable).
```

## 🏗️ Project Structure

```
AFDGCN_Garnoldi/
├── backend/
│   └── app/
│       ├── api/v1/
│       │   ├── predict.py          ✨ NEW: Real-time prediction endpoints
│       │   ├── phases.py           ✓ Phase recommendations
│       │   ├── auth.py             ✓ JWT authentication
│       │   ├── admin.py            ✓ Admin operations
│       │   └── websocket.py        ✓ WebSocket real-time push
│       ├── services/
│       │   ├── real_time_predictor.py  ✨ NEW: Real-time prediction logic
│       │   ├── background_fetcher.py   ✨ NEW: Background data polling
│       │   ├── prediction_wrapper.py   ✓ AFDGCN model wrapper
│       │   ├── phase_calculator.py     ✓ Signal timing (Webster)
│       │   └── kayseri_client.py       ✓ Belediye API client
│       ├── db/
│       │   ├── models.py           ✓ SQLAlchemy ORM models
│       │   └── session.py          ✓ Database sessions
│       ├── core/
│       │   ├── config.py           ✓ Settings (pydantic)
│       │   ├── security.py         ✓ JWT security
│       │   └── exceptions.py       ✓ Error handling
│       ├── schemas/
│       │   └── phases.py           ✓ Pydantic models
│       ├── main.py                 ✓ FastAPI app entry
│       └── state.py                ✓ Global app state
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   │   ├── client.ts           ✓ Axios config + interceptors
│   │   │   ├── phases.ts           ✨ UPDATED: New prediction APIs
│   │   │   ├── auth.ts             ✓ Auth functions
│   │   │   └── types.ts            ✨ UPDATED: New type definitions
│   │   ├── components/
│   │   │   ├── RealTimeDashboard.tsx ✨ NEW: Real-time dashboard
│   │   │   ├── JunctionCard.tsx       ✓ Junction card component
│   │   │   ├── ArmCard.tsx            ✓ Arm details component
│   │   │   └── ...other components
│   │   ├── pages/
│   │   │   ├── DashboardPage.tsx   ✓ Main dashboard
│   │   │   └── LoginPage.tsx       ✓ Authentication
│   │   ├── hooks/
│   │   │   ├── usePhases.ts        ✓ Phase data hooks
│   │   │   └── useWebSocket.ts     ✓ WebSocket hook
│   │   ├── store/
│   │   │   ├── useAuthStore.ts     ✓ Auth state (Zustand)
│   │   │   └── usePhaseStore.ts    ✓ Phase state (Zustand)
│   │   ├── App.tsx                 ✓ Root component + routing
│   │   └── main.tsx                ✓ React entry
│   └── vite.config.ts
├── model/
│   └── AFDGCN.py                   ✓ Model architecture
├── saved_models/
│   ├── kayseri_ildem_v1.pth        ✓ Pre-trained weights
│   └── ...other checkpoints
├── conf/
│   └── Kayseri_*.conf              ✓ Model config files
├── data/
│   └── Kayseri/                    ✓ Graph & training data
├── lib/
│   ├── load_dataset.py             ✓ Data loading utilities
│   ├── load_graph.py               ✓ Graph loading
│   ├── metrics.py                  ✓ Evaluation metrics
│   └── ...other utilities
└── scripts/
    ├── init_db.py                  ✓ Database initialization
    └── ...other scripts
```

## 🔄 Data Flow

### Real-time Prediction Pipeline

1. **Data Fetch** (Every 60s - background_fetcher.py)
   ```
   Belediye API → KayseriAPIClient.fetch_region()
   → Returns: {junction_id: [{"edge_direction": "A", "0": 25, "1": 28, ...}, ...], ...}
   ```

2. **Normalization**
   ```
   Raw counts → Normalize with SCALER_MEAN=28.53, SCALER_STD=38.72
   → Shape: [batch_size, num_nodes, lag, 1]
   ```

3. **AFDGCN Prediction**
   ```
   Historical data + Graph Attention + Temporal Convolution
   → Output: [batch_size, num_nodes, horizon, 1]
   → Returns next 10-min vehicle count forecast
   ```

4. **Phase Calculation** (Webster's Method)
   ```
   Vehicle counts per arm
   → Calculate load per lane
   → Determine cycle time (60-120s based on load)
   → Distribute green time proportionally
   → Fix yellow (3s) & protection (6s)
   → Output: {cycle: 65, "A": {green: 18, yellow: 3, red: 44}, ...}
   ```

5. **Cache & Broadcast**
   ```
   Store in memory cache (300s TTL)
   → WebSocket broadcast to all connected clients
   → HTTP API for on-demand requests
   ```

## 🔐 Security Features

- ✅ JWT token authentication (HS256)
- ✅ CORS protection
- ✅ Rate limiting (via slowapi)
- ✅ Async context managers for resource cleanup
- ✅ SQL injection prevention (SQLAlchemy ORM)
- ✅ Input validation (Pydantic)
- ✅ Error handling with proper HTTP status codes
- ✅ Logging for security events

## 📈 Performance Optimization

- **Caching**: 300s in-memory cache for predictions
- **Background Fetching**: Async data polling (60s interval)
- **WebSocket Broadcasting**: Reduced latency vs polling (600s interval)
- **Database Indexing**: timestamp, region, junction_id on PhasePrediction table
- **Lazy Model Loading**: AFDGCN loads only once on startup
- **Fallback Mechanism**: Moving average when model unavailable
- **Connection Pooling**: AsyncPG connection pool

## 🧪 Testing

```bash
# Backend tests
cd backend
pytest tests/ -v --cov

# Frontend tests
cd frontend
npm test

# Integration test (manual)
1. Start backend: uvicorn backend.app.main:app --port 9001
2. Start Belediye API: python kayseri_api.py --port 9000
3. Login: POST /auth/login with demo/demo123
4. Prediction: POST /api/v1/predict/region/ildem
5. WebSocket: Connect to ws://localhost:9001/ws/live?token=<JWT>
```

## 📝 Database Schema

### phase_predictions table
```sql
CREATE TABLE phase_predictions (
    id UUID PRIMARY KEY,
    created_at TIMESTAMP WITH TIMEZONE DEFAULT now(),
    region VARCHAR(50) NOT NULL,
    city VARCHAR(50) DEFAULT 'kayseri',
    time_label VARCHAR(10),
    minute_index INTEGER,
    prediction_source VARCHAR(30),
    kayseri_api_status VARCHAR(20),
    junction_count INTEGER,
    total_vehicles INTEGER,
    payload JSON,
    
    INDEXES: (created_at, region, minute_index)
);

CREATE TABLE model_events (
    id UUID PRIMARY KEY,
    created_at TIMESTAMP WITH TIMEZONE DEFAULT now(),
    event_type VARCHAR(30) NOT NULL,
    model_path VARCHAR(255),
    num_nodes INTEGER,
    lag INTEGER,
    details VARCHAR(500),
    
    INDEXES: (created_at, event_type)
);
```

## 🔧 Configuration

All settings via environment variables (.env):

```env
# API Configuration
KAYSERI_API_URL=http://localhost:9000
KAYSERI_USERNAME=demo
KAYSERI_PASSWORD=demo123

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/dbname

# JWT
JWT_SECRET_KEY=your-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# WebSocket
WS_BROADCAST_INTERVAL=600

# Logging
LOG_LEVEL=INFO

# API Version
API_VERSION=1.0.0
```

## 🐛 Troubleshooting

### Model Load Error
```
❌ AFDGCN model yükleme başarısız
→ Check: saved_models/ folder exists
→ Check: Correct node count (34 for İldem)
→ Check: PyTorch version compatibility
→ Fallback: System uses moving average
```

### API Connection Error
```
⚠️ Kayseri API ulaşılamıyor
→ Check: Belediye API running on port 9000
→ Check: Network connectivity
→ Fallback: Uses cached data
```

### WebSocket Timeout
```
🔌 WS bağlantı kesildi
→ Frontend should auto-reconnect
→ Check: JWT token validity
→ Check: Firewall/proxy settings
```

## 📚 API Documentation

Swagger UI: http://localhost:9001/docs
ReDoc: http://localhost:9001/redoc

## 🤝 Contributing

1. Create feature branch: `git checkout -b feature/my-feature`
2. Commit changes: `git commit -m "feat: my feature"`
3. Push: `git push origin feature/my-feature`
4. Create PR

## 📄 License

MIT License - see LICENSE file for details

---

**Last Updated**: May 5, 2026
**System Status**: ✅ Production Ready
**Model**: AFDGCN (Garnoldi Algorithm)
**Regions**: İldem (9), Tuna (7), Kızılırmak (3)
