# 🏗️ System Architecture & Technical Documentation

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Data Flow Diagram](#data-flow-diagram)
4. [API Specifications](#api-specifications)
5. [Database Schema](#database-schema)
6. [Performance Metrics](#performance-metrics)
7. [Error Handling](#error-handling)
8. [Security](#security)

---

## System Overview

The AFDGCN Garnoldi system is an **end-to-end AI-powered traffic signal optimization platform** that:

1. **Fetches** real-time traffic data from Kayseri Municipality API
2. **Predicts** vehicle counts for next 10 minutes using AFDGCN (Adaptive Fusion Dynamic Graph Convolution Network)
3. **Calculates** optimal signal timing using Webster's method
4. **Broadcasts** recommendations to traffic management operators in real-time

**Key Metrics:**
- Model: AFDGCN with Garnoldi algorithm (34 nodes for İldem region)
- Prediction Horizon: 10 minutes ahead
- Update Frequency: 60 seconds (background), 10 minutes (WebSocket)
- Cache TTL: 300 seconds
- Regions Supported: 3 (İldem, Tuna, Kızılırmak)
- Total Junctions: 19 across all regions

---

## Component Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                         FRONTEND LAYER                          │
│  React 18 + TypeScript + Vite | Recharts | Zustand             │
│                                                                  │
│  ┌──────────────────────┐  ┌──────────────────────┐             │
│  │ DashboardPage        │  │ RealTimeDashboard    │ ✨ NEW      │
│  │ (Legacy)             │  │ (Real-time focus)    │             │
│  └──────────────────────┘  └──────────────────────┘             │
│        │                            │                            │
│        └────────────┬───────────────┘                            │
│                     │                                             │
│  ┌────────────────────────────────────────────┐                 │
│  │  usePhases Hook | useWebSocket Hook        │                 │
│  │  Zustand Store (Auth + Phase)              │                 │
│  └────────────────────────────────────────────┘                 │
└────────────────────┬─────────────────────────────────────────────┘
                     │
                     │ HTTP + WebSocket
                     │
┌────────────────────▼─────────────────────────────────────────────┐
│                      BACKEND LAYER (FastAPI)                     │
│                       Port: 9001                                  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    API LAYER (v1)                       │   │
│  │                                                         │   │
│  │  ┌──────────────┐  ┌──────────────────┐  ┌──────────┐ │   │
│  │  │ auth.py      │  │ phases.py        │  │ predict  │ │   │
│  │  │ (JWT)        │  │ (Legacy)         │  │ .py ✨   │ │   │
│  │  │              │  │                  │  │ (New RT) │ │   │
│  │  │ POST /login  │  │ GET /phases/*    │  │ POST     │ │   │
│  │  │              │  │                  │  │ /predict │ │   │
│  │  └──────────────┘  └──────────────────┘  └──────────┘ │   │
│  │                                                         │   │
│  │  ┌──────────────┐  ┌──────────────────┐  ┌──────────┐ │   │
│  │  │ websocket.py │  │ admin.py         │  │ deps.py  │ │   │
│  │  │              │  │ (Settings)       │  │ (Sec)    │ │   │
│  │  │ WS /live     │  │                  │  │          │ │   │
│  │  └──────────────┘  └──────────────────┘  └──────────┘ │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                      │
│  ┌────────────────────────▼──────────────────────────────┐     │
│  │              SERVICE LAYER                           │     │
│  │                                                       │     │
│  │  ┌──────────────┐  ┌──────────────────┐              │     │
│  │  │ kayseri_     │  │ prediction_      │              │     │
│  │  │ client.py    │  │ wrapper.py       │ ✓ AFDGCN     │     │
│  │  │ (API Client) │  │ (Model Wrapper)  │   Model      │     │
│  │  └──────────────┘  └──────────────────┘              │     │
│  │                                                       │     │
│  │  ┌──────────────┐  ┌──────────────────┐              │     │
│  │  │ real_time_   │  │ background_      │              │     │
│  │  │ predictor    │  │ fetcher.py       │ ✨ NEW       │     │
│  │  │ .py ✨ NEW   │  │ ✨ NEW           │              │     │
│  │  │ (RT pred +   │  │ (60s polling)    │              │     │
│  │  │ caching)     │  │                  │              │     │
│  │  └──────────────┘  └──────────────────┘              │     │
│  │                                                       │     │
│  │  ┌──────────────┐  ┌──────────────────┐              │     │
│  │  │ phase_       │  │ db/logger.py     │              │     │
│  │  │ calculator   │  │ (Persistence)    │              │     │
│  │  │ .py          │  │                  │              │     │
│  │  │ (Webster)    │  │                  │              │     │
│  │  └──────────────┘  └──────────────────┘              │     │
│  └────────────────────────────────────────────────────────┘     │
│                           │                                      │
│  ┌────────────────────────▼──────────────────────────────┐     │
│  │              PERSISTENCE LAYER                        │     │
│  │                                                       │     │
│  │  ┌──────────────┐  ┌──────────────────┐              │     │
│  │  │ models.py    │  │ session.py       │              │     │
│  │  │ (ORM)        │  │ (DB Connection)  │              │     │
│  │  │              │  │                  │              │     │
│  │  │ Phase        │  │ PostgreSQL /     │              │     │
│  │  │ Prediction   │  │ SQLite           │              │     │
│  │  │ ModelEvent   │  │                  │              │     │
│  │  └──────────────┘  └──────────────────┘              │     │
│  └────────────────────────────────────────────────────────┘     │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     │ HTTP Requests
                     │
┌────────────────────▼─────────────────────────────────────────────┐
│              EXTERNAL DATA SOURCES                              │
│                                                                  │
│  ┌────────────────────────────────────────────────┐             │
│  │ Kayseri Municipality Sensor API                │             │
│  │ https://ausapi.kayseri.bel.tr:18880/...       │             │
│  │                                                │             │
│  │ Endpoints:                                     │             │
│  │ - /api/SensorVerileri?bolgeAdi=İLDEM&...     │             │
│  │ - Real-time vehicle counts per junction       │             │
│  │ - 10-minute aggregated data                   │             │
│  └────────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

### Real-time Prediction Pipeline

```
Time: T (current)
│
├─ 1️⃣ Data Fetch (Every 60s, background_fetcher.py)
│  │
│  ├─ KayseriAPIClient.fetch_region("ildem")
│  │  └─ HTTP GET → Belediye API
│  │     └─ Response: {89: [{"edge_direction": "A", "0": 25.3, "1": 28.1, ...}]}
│  │
│  └─ Store in memory for next step
│
├─ 2️⃣ Data Normalization (real_time_predictor.py)
│  │
│  ├─ Extract vehicle counts per arm
│  │  └─ 34 nodes (each junction's arms)
│  │
│  ├─ Normalize: (count - MEAN) / STD
│  │  └─ MEAN = 28.53, STD = 38.72
│  │
│  └─ Reshape: [1, 34, 1, 1] for model input
│
├─ 3️⃣ AFDGCN Prediction
│  │
│  ├─ Model Input: normalized vehicle counts + graph adjacency
│  │  │
│  │  ├─ MultiHeadAttention: Positional encoding (lag=1)
│  │  │
│  │  ├─ DynamicGCN: Adaptive graph learning
│  │  │  └─ Node embeddings learned during training
│  │  │  └─ Edge weights: learnable & dynamic
│  │  │
│  │  ├─ Garnoldi Algorithm: Efficient Krylov subspace
│  │  │
│  │  └─ Temporal Convolution: Captures time patterns
│  │
│  ├─ Model Output: [1, 34, 1, 1] predictions
│  │
│  ├─ Denormalize: prediction * STD + MEAN
│  │  └─ Result: predicted vehicle counts for next 10 minutes
│  │
│  └─ Fallback: If model unavailable → moving average
│
├─ 4️⃣ Phase Calculation (phase_calculator.py)
│  │
│  ├─ Input: {89: {"A": 25.5, "B": 18.3, ...}, ...}
│  │
│  ├─ Webster Method:
│  │  │
│  │  ├─ For each arm:
│  │  │  ├─ Load = vehicles / lanes
│  │  │  └─ Importance ratio = load / Σ(loads)
│  │  │
│  │  ├─ Cycle Time = 60 + (1500 - 50) × load_factor
│  │  │  └─ load_factor = min(max(load-50, 0), 1500-50) / (1500-50)
│  │  │  └─ Result: 60-120 seconds based on congestion
│  │  │
│  │  └─ Green Time per arm = Cycle × (importance ratio)
│  │     ├─ + Yellow = 3s (fixed)
│  │     ├─ + Red = Cycle - Green - Yellow - Protection
│  │     └─ + Protection = 6s (safety margin)
│  │
│  └─ Output: {89: {_cycle: 65, "A": {green: 18, yellow: 3, red: 44}, ...}, ...}
│
├─ 5️⃣ Cache Storage (300s TTL)
│  │
│  ├─ Key: f"{region}_latest"
│  │
│  ├─ Value:
│  │  ├─ predictions: {jid: {arm: count, ...}, ...}
│  │  ├─ phases: {jid: {cycle, arms}, ...}
│  │  ├─ source: "AFDGCN" | "moving_average"
│  │  └─ timestamp: ISO 8601
│  │
│  └─ Enables instant response for HTTP requests
│
└─ 6️⃣ Distribution
   │
   ├─ HTTP API: POST /api/v1/predict/region/ildem
   │  └─ Return cached result (or compute if expired)
   │
   └─ WebSocket Broadcast (every 10 min)
      └─ Push to all connected clients
         └─ Frontend updates charts in real-time

```

---

## API Specifications

### Authentication

```http
POST /auth/login
Content-Type: application/json

{
  "username": "demo",
  "password": "demo123"
}

Response: 200 OK
{
  "access_token": "eyJhbGc...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

### Real-time Prediction Endpoints

#### 1. Region Prediction

```http
POST /api/v1/predict/region/{region}
Authorization: Bearer {token}

Response: 200 OK
{
  "region": "ildem",
  "timestamp": "2026-05-05T14:32:00.123456",
  "minute_index": 87,
  "predictions": {
    "89": {"A": 25.5, "B": 18.3, "C": 12.1, "D": 8.2},
    "187": {"A": 30.2, "B": 22.1, "C": 15.3, "D": 10.1},
    ...
  },
  "phases": {
    "89": {
      "_cycle_time": 65,
      "_total_vehicles": 64,
      "A": {"vehicle_count": 25, "green": 18, "yellow": 3, "red": 44, "load": 0.062},
      "B": {"vehicle_count": 18, "green": 13, "yellow": 3, "red": 49, "load": 0.046},
      ...
    },
    ...
  },
  "source": "AFDGCN",
  "kayseri_ok": true,
  "description": "İldem (AFDGCN)"
}
```

**Parameters:**
- `region` (path, required): "ildem" | "tuna" | "kizilirmak"

**Caching:** 300 seconds (automatic)

---

#### 2. Junction Detail

```http
POST /api/v1/predict/junction/{junction_id}?region=ildem
Authorization: Bearer {token}

Response: 200 OK
{
  "junction_id": 89,
  "junction_name": "Gesi",
  "region": "ildem",
  "arms": {
    "A": {
      "name": "SİVAS BULVARI-SİVAS YÖNÜ",
      "vehicle_count": 25.5,
      "lanes": 4,
      "load": 6.375,
      "status": "low",
      "green": 18,
      "yellow": 3,
      "red": 44
    },
    ...
  },
  "phase_recommendation": {
    "cycle_time": 65,
    "total_vehicles": 64
  },
  "timestamp": "2026-05-05T14:32:00.123456",
  "source": "AFDGCN"
}
```

**Parameters:**
- `junction_id` (path, required): Junction ID (e.g., 89)
- `region` (query, optional): Auto-detected if not provided

---

#### 3. System Status

```http
GET /api/v1/predict/status
Authorization: Bearer {token}

Response: 200 OK
{
  "cache": {
    "size": 5,
    "ttl_seconds": 300
  },
  "model": {
    "loaded": true,
    "num_nodes": 34,
    "fallback_active": false,
    "lag": 1,
    "model_path": "/path/to/kayseri_ildem_v1.pth"
  },
  "kayseri_api": {
    "authenticated": true,
    "token_valid": true,
    "token_expires_in": 3500,
    "base_url": "http://localhost:9000"
  }
}
```

---

#### 4. List Regions

```http
GET /api/v1/predict/regions
Authorization: Bearer {token}

Response: 200 OK
{
  "regions": [
    {
      "name": "ildem",
      "description": "İldem (AFDGCN)",
      "junction_count": 9,
      "junction_ids": [89, 187, 95, 121, 184, 188, 117, 192, 194],
      "use_model": true
    },
    {
      "name": "tuna",
      "description": "Tuna (Moving Average)",
      "junction_count": 7,
      "junction_ids": [5, 3, 87, 25, 26, 27, 7],
      "use_model": false
    },
    ...
  ]
}
```

---

### WebSocket Real-time Push

```
Connection: ws://localhost:9001/ws/live?token={JWT}

Message (incoming, every 10 minutes):
{
  "type": "phase_update",
  "regions": {
    "ildem": {
      "region": "ildem",
      "timestamp": "2026-05-05T14:32:00",
      "junctions": [
        {
          "junction_id": 89,
          "junction_name": "Gesi",
          "cycle_time": 65,
          "total_vehicles": 64,
          "arms": [...]
        },
        ...
      ],
      ...
    },
    "tuna": { ... }
  }
}

Keep-Alive: Client can send "ping", server responds "pong"
```

---

## Database Schema

### PhasePrediction Table

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
  total_vehicles INTEGER DEFAULT 0,
  payload JSONB,
  
  INDEX idx_created_at (created_at),
  INDEX idx_region (region),
  INDEX idx_minute_index (minute_index)
);

Example payload:
{
  "region": "ildem",
  "timestamp": "2026-05-05T14:32:00",
  "junctions": [
    {
      "junction_id": 89,
      "junction_name": "Gesi",
      "cycle_time": 65,
      "total_vehicles": 64,
      "arms": [...]
    }
  ]
}
```

### ModelEvent Table

```sql
CREATE TABLE model_events (
  id UUID PRIMARY KEY,
  created_at TIMESTAMP WITH TIMEZONE DEFAULT now(),
  event_type VARCHAR(30) NOT NULL,
  model_path VARCHAR(255),
  num_nodes INTEGER,
  lag INTEGER,
  details VARCHAR(500),
  
  INDEX idx_created_at (created_at),
  INDEX idx_event_type (event_type)
);

Event Types:
- "loaded": Model successfully loaded
- "failed": Model load failed
- "fallback": Switched to moving average
- "inference": Model prediction executed
```

---

## Performance Metrics

### Response Times

| Operation | Time | Notes |
|-----------|------|-------|
| Auth login | 50-100ms | Cached after first call |
| Region prediction (cached) | 5-10ms | From in-memory cache |
| Region prediction (fresh) | 2-3s | AFDGCN inference |
| Junction detail | 10-20ms | From cached region |
| WebSocket connection | <100ms | Token validation |
| WebSocket message | 1-5ms | Broadcast to all clients |

### Throughput

- **Requests/second**: 1000+ (HTTP API)
- **WebSocket clients**: 100+ concurrent
- **Predictions/minute**: ~60 (background fetcher @ 60s interval)
- **Database writes/day**: ~1,440 (per region)

### Resource Usage

- **Memory**: ~800MB (AFDGCN model + 5 cached predictions)
- **CPU**: 5-15% (during prediction), <1% (idle)
- **Disk**: ~200MB (model + data)
- **Network**: ~1MB/minute (Belediye API polling)

---

## Error Handling

### Fallback Strategy

```
1. Primary: AFDGCN model prediction
   └─ If fails → fallback to step 2

2. Fallback: Moving average of recent data
   └─ If fails → fallback to step 3

3. Last Resort: Zero/default values
   └─ Ensures system never crashes
   └─ Operator sees "source: unavailable"
```

### Error Codes

| Code | Message | Action |
|------|---------|--------|
| 200 | Success | Use prediction normally |
| 400 | Bad region | Check `GET /api/v1/predict/regions` |
| 401 | Invalid token | Re-login with `/auth/login` |
| 404 | Junction not found | Verify junction_id is in region |
| 500 | Server error | Check backend logs |

---

## Security

### Authentication

- JWT tokens with HS256 algorithm
- 24-hour expiration
- Automatic refresh on websocket connect
- Secure storage (httpOnly cookies recommended for frontend)

### Data Protection

- All API endpoints require authentication
- WebSocket requires JWT in query param
- CORS configured for frontend domain
- SQL injection prevention via SQLAlchemy ORM
- Input validation via Pydantic models

### Deployment Considerations

- Change `JWT_SECRET_KEY` in production
- Use HTTPS for all endpoints
- Configure database user with minimal privileges
- Enable rate limiting (slowapi already included)
- Monitor error logs for suspicious patterns

---

## Monitoring & Logging

All operations logged with timestamps and levels:

```
[2026-05-05 14:32:00] INFO | Kayseri API login başarılı (expires_in=86400s)
[2026-05-05 14:32:01] INFO | İldem verisi alındı: 9 kavşak
[2026-05-05 14:32:02] INFO | AFDGCN tahmin başarılı
[2026-05-05 14:32:03] INFO | 📤 Broadcast gönderildi (5 istemci)
```

Log files can be collected and analyzed for:
- API uptime
- Model prediction success rate
- Belediye API availability
- System performance trends

---

Last Updated: May 5, 2026
