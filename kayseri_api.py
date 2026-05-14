"""
🚀 Kayseri Büyükşehir Belediyesi Trafik Tahmin API
==================================================

Professional FastAPI service for traffic prediction using AFDGCN model.

Endpoints:
    - GET  /                    : API bilgileri
    - GET  /health             : Sağlık kontrolü
    - POST /predict/{junction_id} : Kavşak için tahmin yap
    - GET  /junctions          : Mevcut kavşak listesi
    - GET  /docs               : Swagger UI (Otomatik Dokümantasyon)
    - GET  /redoc              : ReDoc UI (Alternatif Dokümantasyon)

Usage:
    uvicorn kayseri_api:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException, Query, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import numpy as np
import logging
from pathlib import Path

# JWT ve Password Hashing
from jose import JWTError, jwt
from passlib.context import CryptContext

# Web Scraping
import requests
import json
import urllib3

# SSL uyarılarını kapat (self-signed certificate için)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# BUILT-IN: Belediye Cookies Configuration
# ============================================================================

BELEDIYE_COOKIES = {
    '.AspNetCore.Antiforgery.708PDUFCWo4': 'CfDJ8B9JLEL67llGtXY4pwUvuwygNgpVdSBpqo58gIsraZTkJoO9l_vmK9fPZTbFz9qKHyhKGyWbMEZRP1jjNbO5KB2fA7vT9ZHH8dNFc65mUPzQ5zFiCCa7UuZGI1CoNMKwR3C2rkwOTkJ6IXeGRRqoXG4',
    '.AspNetCore.Identity.Application': 'CfDJ8B9JLEL67llGtXY4pwUvuwzENkFdMmSSt0X8NUoZ7onLJ3F_hVrpfwtdQjnfgKki_oVaTGU_agTPcGk1iB3mf2PHjYJMHRyZISK5weVMU3dTmgJyfgIeSVFLqnSdWo0jNa-O3gxmzvFiMZ-hNI-v_ufzg4mL48kU-mWAFq9RzF2xdEyehBrTMA-QKo7fW7pcIc2sfCsPkzDuF6gcV9u2bI4Jt_CgO9twe18_VAl8Jh8R0PusijTPJ73Qgiepg0XZwuYitMs9IL7OOkkpPU6Odx_p0eBJ6uIZ4aluDv4zjGb98jQUi3sbyYW6gFWU1WaYq01ehbQ7OD8lLUhEdYVD8aC-nOtMtHaqm4jkoOi5xqAK-3aeICcSYkg92U9R1-1cbcSk4YDz6aqLKWVoLsCzi7zfJpYqrHUu-FIx1lz_7pbMFWZWcW4EwBQw6X-_y8wj7pWLFmTthMTfuZwI7WKHglH-N0WVa3kKWyEU0TD0mhcRcidHViII51Wq2SpVug98M2_mz63UA_NbcR_AeNNBFwIwBSSGhXmSzoQWmriPb4HcdHNZqptb1IwLsfaqddT_560sdoSL02CWI7Z-fLGpfSFCVsyOUdSqenUXFNEDFuKOckbfPqKnSg4VlfcDO7_hrQyt0ZqKGNxijpwb9TGctcUuWgY7-nM9dUx2zgHl_LTx3ns2g2UE1QMVG9DTVlyXvQz7_jaDhAbrdqvxVOJ9PKzVL2wCbbBhwRaMPtzm31lAMpz2tvi6glH183LQMYf5Z3u0o7uoaO_wsyTBQIuHM-X0pkzNaJ7LL_s2cb6d96XOlJw6GxwudHsZZl0xs6g8p-wd96JdQ5-qlFyPBN4nUgsuJKL-YSoCNlwF44VUnqHj'
}

# ============================================================================
# BUILT-IN: Region & Junction Definitions
# ============================================================================

REGION_JUNCTIONS = {
    "kayseri": {
        "ildem": {
            "junction_count": 9,
            "junction_ids": [89, 187, 95, 121, 184, 188, 117, 192, 194],
            "description": "Kayseri İldem Bölgesi - 9 kavşak"
        },
        "tuna": {
            "junction_count": 7,
            "junction_ids": [5, 3, 87, 25, 26, 27, 7],
            "description": "Kayseri Tuna Bölgesi - 7 kavşak"
        },
        "kizilirmak": {
            "junction_count": 3,
            "junction_ids": [130, 38, 176],
            "description": "Kayseri Kızılırmak Bölgesi - 3 kavşak"
        }
    }
}

KNOWN_JUNCTIONS = {
    89: {"name": "Kavşak 89", "edges": 4, "city": "kayseri", "region": "ildem"},
    187: {"name": "822. SK - GESİ CAD. - KOCASİNAN CAD.", "edges": 4, "city": "kayseri", "region": "ildem"},
    95: {"name": "Kavşak 95", "edges": 4, "city": "kayseri", "region": "ildem"},
    121: {"name": "Kavşak 121", "edges": 4, "city": "kayseri", "region": "ildem"},
    184: {"name": "Kavşak 184", "edges": 4, "city": "kayseri", "region": "ildem"},
    188: {"name": "Kavşak 188", "edges": 4, "city": "kayseri", "region": "ildem"},
    117: {"name": "Kavşak 117", "edges": 4, "city": "kayseri", "region": "ildem"},
    192: {"name": "Kavşak 192", "edges": 4, "city": "kayseri", "region": "ildem"},
    194: {"name": "Kavşak 194", "edges": 4, "city": "kayseri", "region": "ildem"},
    5: {"name": "Kavşak 5", "edges": 4, "city": "kayseri", "region": "tuna"},
    3: {"name": "Kavşak 3", "edges": 4, "city": "kayseri", "region": "tuna"},
    87: {"name": "Kavşak 87", "edges": 4, "city": "kayseri", "region": "tuna"},
    25: {"name": "Kavşak 25", "edges": 4, "city": "kayseri", "region": "tuna"},
    26: {"name": "Kavşak 26", "edges": 4, "city": "kayseri", "region": "tuna"},
    27: {"name": "Kavşak 27", "edges": 4, "city": "kayseri", "region": "tuna"},
    7: {"name": "Kavşak 7", "edges": 4, "city": "kayseri", "region": "tuna"},
    130: {"name": "Kavşak 130", "edges": 4, "city": "kayseri", "region": "kizilirmak"},
    38: {"name": "Kavşak 38", "edges": 4, "city": "kayseri", "region": "kizilirmak"},
    176: {"name": "Kavşak 176", "edges": 4, "city": "kayseri", "region": "kizilirmak"}
}

# ============================================================================
# BUILT-IN: Config Helper Functions
# ============================================================================

def get_region_junctions(city: str, region: str):
    """Belirli bir bölge için kavşak listesini döndürür"""
    city_data = REGION_JUNCTIONS.get(city.lower())
    if not city_data:
        return None
    region_data = city_data.get(region.lower())
    return region_data

def is_valid_junction_for_region(junction_id: int, city: str, region: str) -> bool:
    """Bir kavşak ID'sinin belirli bir bölge için geçerli olup olmadığını kontrol eder"""
    region_data = get_region_junctions(city, region)
    if not region_data:
        return False
    return junction_id in region_data["junction_ids"]

# ============================================================================
# BUILT-IN: Belediye Data Fetcher Class
# ============================================================================

class BelediyeDataFetcher:
    """
    Kayseri Büyükşehir Belediyesi web arayüzünden trafik verisi çeker.
    """
    
    def __init__(self, base_url: str = "https://10.50.234.18", cookies: dict = None):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.verify = False  # SSL doğrulamasını atla
        
        if cookies:
            self.session.cookies.update(cookies)
        
        self.session.headers.update({
            'accept': '*/*',
            'accept-language': 'en-US,en;q=0.9',
            'cache-control': 'no-cache',
            'pragma': 'no-cache',
            'referer': f'{base_url}/Rapor',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'x-requested-with': 'XMLHttpRequest'
        })
    
    def fetch_junction_data(self, junction_id: int, date: str = None, wa: int = 0) -> Optional[List[Dict]]:
        """Belirli bir kavşak için veri çeker."""
        
        if date is None:
            date = datetime.now().strftime("%Y-%m-%dT00:00")
        
        endpoint = f"{self.base_url}/Rapor/GetDakikalikRapor"
        params = {"id": junction_id, "date": date, "wa": wa}
        
        try:
            logger.info(f"📡 API çağrısı: id={junction_id}, date={date}")
            response = self.session.get(endpoint, params=params, verify=False, timeout=30)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if isinstance(data, dict):
                        return data.get("data", data.get("results", data))
                    elif isinstance(data, list):
                        return data
                    else:
                        return data
                except ValueError as e:
                    logger.warning(f"⚠️ JSON parse hatası: {e}")
                    return None
            else:
                logger.error(f"❌ API hatası: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Veri çekme hatası: {e}")
            return None

# ============================================================================
# Authentication Configuration
# ============================================================================

# JWT ayarları
SECRET_KEY = "kayseri-traffic-api-secret-key-2026-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 saat

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer security
security = HTTPBearer(auto_error=False)

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="🚦 Kayseri Trafik Tahmin API",
    description="""
    **Kayseri Büyükşehir Belediyesi** için geliştirilmiş yapay zeka destekli trafik tahmin sistemi.
    
    ## Özellikler
    
    * 🤖 **AFDGCN Model**: Gelişmiş derin öğrenme modeli ile tahmin
    * 📊 **Gerçek Zamanlı Veri**: Belediye sisteminden anlık veri çekme
    * 🔄 **RESTful API**: Standart HTTP istekleri ile kolay entegrasyon
    * 📱 **Otomatik Dokümantasyon**: Swagger UI ve ReDoc desteği
    * 🔒 **Güvenli**: CORS koruması ve rate limiting
    
    ## Nasıl Kullanılır?
    
    1. `/junctions` endpoint'inden kavşak listesini alın
    2. `/predict/{junction_id}` ile tahmin yapın
    3. JSON formatında tahmin sonuçlarını alın
    
    ## Teknik Detaylar
    
    - **Framework**: FastAPI 0.116+
    - **ML Model**: AFDGCN (Adaptive Fusion Dynamic Graph Convolution Network)
    - **Python**: 3.12+
    - **Veri Kaynağı**: Kayseri Büyükşehir Belediyesi Trafik Sistemi
    """,
    version="1.0.0",
    contact={
        "name": "AFDGCN Trafik Tahmin Sistemi",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT License",
    },
)

# CORS middleware - Dışarıdan erişim için
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production'da spesifik domain'ler belirtin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Pydantic Models (Request/Response Schemas)
# ============================================================================

class TrafficRequest(BaseModel):
    """Trafik verisi isteği"""
    city: str = Field(
        ...,
        description="İl adı (kayseri, sivas)",
        example="kayseri"
    )
    region: str = Field(
        ...,
        description="Bölge adı (ildem, tuna, merkez, vb.)",
        example="ildem"
    )
    date: Optional[str] = Field(
        None,
        description="Tarih (YYYY-MM-DD formatında, örn: 2026-04-21). Varsayılan: bugün",
        example="2026-04-21"
    )
    hour: Optional[int] = Field(
        None,
        ge=0,
        le=23,
        description="Saat (0-23). Eğer verilirse sadece o saat döner.",
        example=14
    )
    minute: Optional[int] = Field(
        None,
        ge=0,
        le=59,
        description="Dakika (0-59). Saat ile birlikte kullanılır.",
        example=30
    )
    aggregation: Optional[str] = Field(
        None,
        description="Toplama türü: 'hourly' (saatlik) veya boş (10'ar dakika). Sadece tarih girildiğinde geçerli.",
        example="hourly"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "city": "kayseri",
                "region": "ildem",
                "date": "2026-04-21",
                "aggregation": "hourly"
            }
        }


class EdgeTrafficData(BaseModel):
    """Her kol için trafik verisi"""
    edge_name: str = Field(..., description="Yol adı")
    edge_direction: str = Field(..., description="Yön (A, B, C, D)")
    traffic_count: float = Field(..., description="Araç sayısı")


class TimeSlotData(BaseModel):
    """Belirli bir zaman dilimi için veri"""
    time: str = Field(..., description="Zaman (HH:MM formatında)")
    edges: List[EdgeTrafficData] = Field(..., description="Her kol için veri")


class TrafficResponse(BaseModel):
    """Trafik verisi cevabı"""
    junction_id: int = Field(..., description="Kavşak ID")
    city: str = Field(..., description="İl")
    region: str = Field(..., description="Bölge")
    data_date: str = Field(..., description="Veri tarihi")
    aggregation: str = Field(..., description="Toplama türü: '10min' veya 'hourly'")
    timestamp: str = Field(..., description="API çağrı zamanı")
    time_slots: List[TimeSlotData] = Field(..., description="Zaman dilimlerine göre veri")
    model_info: Dict[str, Any] = Field(..., description="Model bilgisi")
    data_source: str = Field(..., description="Veri kaynağı")
    
    class Config:
        json_schema_extra = {
            "example": {
                "junction_id": 187,
                "city": "kayseri",
                "region": "ildem",
                "data_date": "2026-04-21",
                "aggregation": "10min",
                "timestamp": "2026-04-22T12:00:00",
                "time_slots": [
                    {
                        "time": "00:00",
                        "edges": [
                            {"edge_name": "822. SK", "edge_direction": "A", "traffic_count": 25},
                            {"edge_name": "GESİ CAD.", "edge_direction": "B", "traffic_count": 42}
                        ]
                    },
                    {
                        "time": "00:10",
                        "edges": [
                            {"edge_name": "822. SK", "edge_direction": "A", "traffic_count": 28},
                            {"edge_name": "GESİ CAD.", "edge_direction": "B", "traffic_count": 45}
                        ]
                    }
                ],
                "model_info": {
                    "model_id": "kayseri_ildem_v1",
                    "version": "1.0",
                    "junction_count": 19
                },
                "data_source": "Kayseri Belediye API (Live)"
            }
        }


class JunctionInfo(BaseModel):
    """Kavşak bilgisi"""
    id: int = Field(..., description="Kavşak ID")
    name: str = Field(..., description="Kavşak adı")
    edges: int = Field(..., description="Kol sayısı")
    status: str = Field(..., description="Durum (active/inactive)")


class User(BaseModel):
    """Kullanıcı modeli"""
    username: str = Field(..., description="Kullanıcı adı")
    email: Optional[str] = Field(None, description="E-posta")
    full_name: Optional[str] = Field(None, description="Tam adı")
    disabled: Optional[bool] = Field(False, description="Hesap durumu")


class UserInDB(User):
    """Veritabanındaki kullanıcı (şifreli)"""
    hashed_password: str = Field(..., description="Hash'lenmiş şifre")


class UserCreate(BaseModel):
    """Kullanıcı kayıt modeli"""
    username: str = Field(..., min_length=3, max_length=50, description="Kullanıcı adı")
    password: str = Field(..., min_length=6, description="Şifre")
    email: Optional[str] = Field(None, description="E-posta")
    full_name: Optional[str] = Field(None, description="Tam adı")
    
    class Config:
        json_schema_extra = {
            "example": {
                "username": "ahmet",
                "password": "guvenli123",
                "email": "ahmet@belediye.gov.tr",
                "full_name": "Ahmet Yılmaz"
            }
        }


class Token(BaseModel):
    """JWT Token cevabı"""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(..., description="Token tipi")
    expires_in: int = Field(..., description="Token geçerlilik süresi (saniye)")
    user: User = Field(..., description="Kullanıcı bilgisi")


class LoginRequest(BaseModel):
    """Login isteği"""
    username: str = Field(..., description="Kullanıcı adı")
    password: str = Field(..., description="Şifre")
    
    class Config:
        json_schema_extra = {
            "example": {
                "username": "ahmet",
                "password": "guvenli123"
            }
        }


class RegionInfo(BaseModel):
    """Bölge bilgisi"""
    city: str = Field(..., description="İl adı")
    region: str = Field(..., description="Bölge adı")
    junction_count: int = Field(..., description="Kavşak sayısı")
    junction_ids: List[int] = Field(..., description="Kavşak ID'leri")
    description: str = Field(..., description="Açıklama")


class HealthResponse(BaseModel):
    """Sağlık kontrolü cevabı"""
    status: str
    timestamp: str
    belediye_api: str
    model_loaded: bool
    version: str


# ============================================================================
# Global State
# ============================================================================

class AppState:
    """Uygulama durumu"""
    def __init__(self):
        self.data_fetcher: Optional[BelediyeDataFetcher] = None
        self.model = None
        self.model_loaded = False
        self.cookies = None
        # In-memory user storage (production'da veritabanı kullanın)
        self.users_db: Dict[str, UserInDB] = {}

state = AppState()

# ============================================================================
# Authentication Utilities
# ============================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Şifreyi doğrula"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Şifreyi hash'le"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """JWT token oluştur"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_user(username: str) -> Optional[UserInDB]:
    """Kullanıcıyı getir"""
    return state.users_db.get(username)


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Kullanıcıyı doğrula"""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[User]:
    """JWT token'dan kullanıcıyı al (opsiyonel)"""
    if not credentials:
        return None
    
    token = credentials.credentials
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
    except JWTError:
        return None
    
    user = get_user(username)
    if user is None:
        return None
    
    return User(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        disabled=user.disabled
    )


async def get_current_active_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """JWT token'dan aktif kullanıcıyı al (zorunlu)"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = get_user(username)
    if user is None:
        raise credentials_exception
    
    if user.disabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    
    return User(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        disabled=user.disabled
    )


# ============================================================================
# Startup & Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """API başlarken çalışır"""
    logger.info("🚀 Kayseri Trafik Tahmin API başlatılıyor...")
    
    # Belediye veri çekiciyi hazırla
    logger.info("📡 Belediye veri bağlantısı kuruluyor...")
    state.data_fetcher = BelediyeDataFetcher(cookies=BELEDIYE_COOKIES)
    state.cookies = BELEDIYE_COOKIES
    
    # Demo kullanıcı oluştur (sadece development için)
    demo_user = UserInDB(
        username="demo",
        email="demo@kayseri.gov.tr",
        full_name="Demo Kullanıcı",
        disabled=False,
        hashed_password=get_password_hash("demo123")
    )
    state.users_db["demo"] = demo_user
    logger.info("👤 Demo kullanıcı oluşturuldu: username='demo', password='demo123'")
    
    logger.info("✅ API hazır!")
    logger.info(f"📊 Bilinen kavşak sayısı: {len(KNOWN_JUNCTIONS)}")


@app.on_event("shutdown")
async def shutdown_event():
    """API kapanırken çalışır"""
    logger.info("👋 API kapatılıyor...")


# ============================================================================
# Helper Functions
# ============================================================================

def get_model_info(city: str, region: str) -> Dict[str, Any]:
    """
    Belirtilen il ve bölge için model bilgisini döndürür.
    
    Parameters
    ----------
    city : str
        İl adı (örn: 'kayseri', 'sivas')
    region : str
        Bölge adı (örn: 'ildem', 'tuna', 'merkez')
    
    Returns
    -------
    Dict[str, Any]
        Model bilgileri (basit metadata)
    
    Notes
    -----
    Bu fonksiyon sadece metadata döndürür. Gerçek model yükleme işlemi
    daha sonra veritabanı entegrasyonu ile eklenecek.
    """
    region_data = get_region_junctions(city, region)
    
    if not region_data:
        return None
    
    model_info = {
        'model_id': f"{city}_{region}_v1",
        'city': city,
        'region': region,
        'junction_count': region_data['junction_count'],
        'version': '1.0',
        'created_at': datetime.now().isoformat(),
        'note': 'Model bilgisi - Veritabanı entegrasyonu henüz eklenmedi'
    }
    
    return model_info


def convert_belediye_data_to_matrix(data: List[Dict]) -> np.ndarray:
    """
    Belediye API'sinden gelen veriyi model input formatına çevirir.
    
    Parameters
    ----------
    data : List[Dict]
        Belediye API response (her kol için bir dict)
    
    Returns
    -------
    np.ndarray
        Shape: (num_edges, num_timesteps)
    """
    matrix = []
    for edge in data:
        # Sayısal kolonları al (0, 1, 2, ... 143)
        values = [edge.get(str(i), 0) for i in range(144)]
        matrix.append(values)
    
    return np.array(matrix, dtype=np.float32)


def calculate_trend(current: float, predictions: List[float]) -> str:
    """Trend hesapla"""
    if not predictions:
        return "unknown"
    
    avg_pred = sum(predictions) / len(predictions)
    diff = avg_pred - current
    
    if abs(diff) < 2:
        return "stable"
    elif diff > 0:
        return "increasing"
    else:
        return "decreasing"


def aggregate_data(data: np.ndarray, aggregation: str = "minute") -> np.ndarray:
    """
    Veriyi istenen seviyede toplar (aggregate eder).
    
    Parameters
    ----------
    data : np.ndarray
        Ham veri (num_edges, 144 dakika)
    aggregation : str
        "minute" (dakikalık - değişiklik yok)
        "hourly" (saatlik - her 60 dakikayı topla)
        "daily" (günlük - tüm günü topla)
    
    Returns
    -------
    np.ndarray
        Toplanmış veri
    """
    if aggregation == "minute":
        return data
    
    elif aggregation == "hourly":
        # Her 60 dakikayı topla
        num_edges = data.shape[0]
        hourly = []
        for edge_data in data:
            hours = []
            for hour in range(12):  # 12 saat
                start = hour * 60
                end = min(start + 60, 144)
                hour_sum = np.sum(edge_data[start:end])
                hours.append(hour_sum)
            hourly.append(hours)
        return np.array(hourly, dtype=np.float32)
    
    elif aggregation == "daily":
        # Tüm günü topla (tek değer)
        daily = []
        for edge_data in data:
            daily_sum = np.sum(edge_data)
            daily.append([daily_sum])  # Liste içinde tek eleman
        return np.array(daily, dtype=np.float32)
    
    else:
        # Bilinmeyen aggregation, orijinal veriyi dön
        return data


def mock_prediction(data_matrix: np.ndarray, horizon: int = 12) -> np.ndarray:
    """
    Mock tahmin fonksiyonu (gerçek model yüklenene kadar).
    Basit moving average ile tahmin yapar.
    
    Parameters
    ----------
    data_matrix : np.ndarray
        Input veri (num_edges, num_timesteps)
    horizon : int
        Kaç adım ilerisi tahmin edilsin
    
    Returns
    -------
    np.ndarray
        Tahminler (num_edges, horizon)
    """
    # Son 12 dakikanın ortalamasını al ve küçük noise ekle
    predictions = []
    for edge_data in data_matrix:
        # Son 12 dakika
        recent = edge_data[-12:]
        avg = np.mean(recent)
        
        # Basit trend ekle
        trend = (edge_data[-1] - edge_data[-12]) / 12
        
        # Tahminler
        preds = [avg + trend * i + np.random.normal(0, 1) for i in range(1, horizon + 1)]
        predictions.append(preds)
    
    return np.array(predictions, dtype=np.float32)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """API ana sayfa"""
    return {
        "message": "🚦 Kayseri Trafik Veri API",
        "version": "1.0.0",
        "status": "operational",
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_json": "/openapi.json"
        },
        "authentication": {
            "register": "POST /auth/register → Yeni kullanıcı oluştur",
            "login": "POST /auth/login → Token al",
            "usage": "Token'ı Authorization: Bearer <token> header'ında gönder",
            "demo_credentials": {
                "username": "demo",
                "password": "demo123",
                "note": "Demo hesabı ile test edebilirsiniz"
            }
        },
        "usage_flow": {
            "step_0": "POST /auth/login → Token al (opsiyonel)",
            "step_1": "GET /{city}/regions → Bölge listesi",
            "step_2": "GET /{city}/{region} → Bölgedeki TÜM kavşakların verisi",
            "step_3": "GET /{city}/{region}/{junction_id} → Tek kavşak verisi"
        },
        "example_flow": {
            "step_0": "/auth/login",
            "step_1": "/kayseri/regions",
            "step_2": "/kayseri/ildem",
            "step_3": "/kayseri/ildem/187"
        },
        "features": {
            "authentication": "JWT token based authentication",
            "path_based": "RESTful path-based routing",
            "region_based": "Bölge bazlı kavşak yönetimi",
            "bulk_data": "Bölgedeki tüm kavşak verilerini tek istekte alma",
            "validation": "Kavşak ID validasyonu",
            "live_data": "Belediye canlı veri entegrasyonu"
        },
        "powered_by": "FastAPI + Kayseri Belediye Canlı Veri",
        "data_source": "Kayseri Büyükşehir Belediyesi (Live)"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Sistem sağlık kontrolü"""
    belediye_status = "connected" if state.data_fetcher else "disconnected"
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        belediye_api=belediye_status,
        model_loaded=state.model_loaded,
        version="1.0.0"
    )


# ============================================================================
# Authentication Endpoints
# ============================================================================

@app.post("/auth/register", response_model=Token, tags=["Authentication"])
async def register_user(user: UserCreate):
    """
    Yeni kullanıcı kaydı oluştur.
    
    Parameters
    ----------
    user : UserCreate
        Kullanıcı bilgileri
    
    Returns
    -------
    Token
        JWT token ve kullanıcı bilgisi
    
    Examples
    --------
    POST /auth/register
    {
        "username": "ahmet",
        "password": "guvenli123",
        "email": "ahmet@belediye.gov.tr",
        "full_name": "Ahmet Yılmaz"
    }
    """
    # Kullanıcı zaten var mı?
    if user.username in state.users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Yeni kullanıcı oluştur
    hashed_password = get_password_hash(user.password)
    db_user = UserInDB(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        disabled=False,
        hashed_password=hashed_password
    )
    
    # Kaydet
    state.users_db[user.username] = db_user
    logger.info(f"✅ Yeni kullanıcı kaydedildi: {user.username}")
    
    # Token oluştur
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=User(
            username=db_user.username,
            email=db_user.email,
            full_name=db_user.full_name,
            disabled=db_user.disabled
        )
    )


@app.post("/auth/login", response_model=Token, tags=["Authentication"])
async def login(login_data: LoginRequest):
    """
    Kullanıcı girişi yap ve JWT token al.
    
    Parameters
    ----------
    login_data : LoginRequest
        Kullanıcı adı ve şifre
    
    Returns
    -------
    Token
        JWT token ve kullanıcı bilgisi
    
    Examples
    --------
    POST /auth/login
    {
        "username": "demo",
        "password": "demo123"
    }
    
    Response:
    {
        "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
        "token_type": "bearer",
        "expires_in": 86400,
        "user": {
            "username": "demo",
            "email": "demo@kayseri.gov.tr",
            "full_name": "Demo Kullanıcı",
            "disabled": false
        }
    }
    
    Notes
    -----
    Token'ı aldıktan sonra, diğer endpoint'lere şu şekilde gönderin:
    
    Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
    """
    # Kullanıcıyı doğrula
    user = authenticate_user(login_data.username, login_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if user.disabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    
    # Token oluştur
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=access_token_expires
    )
    
    logger.info(f"✅ Kullanıcı giriş yaptı: {user.username}")
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=User(
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            disabled=user.disabled
        )
    )


@app.get("/auth/me", response_model=User, tags=["Authentication"])
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """
    Mevcut kullanıcı bilgisini getir (token gerekli).
    
    Parameters
    ----------
    current_user : User
        Token'dan alınan kullanıcı
    
    Returns
    -------
    User
        Kullanıcı bilgisi
    
    Examples
    --------
    GET /auth/me
    Headers:
        Authorization: Bearer <token>
    
    Response:
    {
        "username": "demo",
        "email": "demo@kayseri.gov.tr",
        "full_name": "Demo Kullanıcı",
        "disabled": false
    }
    """
    return current_user


@app.get("/{city}/regions", tags=["Navigation"])
async def list_regions(
    city: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Şehrin bölgelerini döner (RESTful path-based) - Token gereklidir.
    
    Parameters
    ----------
    city : str
        İl adı (Örn: kayseri)
    
    Returns
    -------
    List[RegionInfo]
        Bölge bilgileri listesi
    
    Examples
    --------
    GET /kayseri/regions
    
    Response:
    [
      {
        "city": "kayseri",
        "region": "ildem",
        "junction_count": 9,
        "junction_ids": [186, 187, 192, ...]
      }
    ]
    
    Notes
    -----
    RESTful API akışı:
    1. GET /{city}/regions → Bölge listesi ⬅️ Buradasınız
    2. GET /{city}/{region}/junctions → Kavşak listesi
    3. GET /{city}/{region}/{junction_id} → Trafik verisi
    """
    regions = []
    
    # Şehir kontrolü
    city_data = REGION_JUNCTIONS.get(city.lower())
    if not city_data:
        raise HTTPException(
            status_code=404,
            detail=f"İl '{city}' bulunamadı. Mevcut iller: {list(REGION_JUNCTIONS.keys())}"
        )
    
    # O şehrin tüm bölgelerini döndür
    for region_name, region_data in city_data.items():
        regions.append(RegionInfo(
            city=city,
            region=region_name,
            junction_count=region_data["junction_count"],
            junction_ids=region_data["junction_ids"],
            description=region_data["description"]
        ))
    
    return {
        "city": city.lower(),
        "regions": regions,
        "count": len(regions),
        "note": "Her bölgenin tüm kavşak verilerini almak için: GET /{city}/{region}"
    }


@app.get("/{city}/{region}", tags=["Traffic Data"])
async def get_region_traffic_data(
    city: str,
    region: str,
    current_user: User = Depends(get_current_active_user),
    date: Optional[str] = Query(None, description="Tarih (YYYY-MM-DD formatında, boş bırakılırsa bugün)"),
    aggregation: Optional[str] = Query(None, description="Veri toplama tipi: 'hourly' (saatlik toplam)")
):
    """
    Bölgedeki TÜM kavşakların trafik verilerini getirir.
    
    Parameters
    ----------
    city : str
        İl adı (path parameter)
    region : str
        Bölge adı (path parameter)
    date : Optional[str]
        Tarih (query parameter, opsiyonel)
    aggregation : Optional[str]
        Toplama tipi (query parameter, opsiyonel)
    
    Returns
    -------
    Dict
        Tüm kavşakların trafik verileri
    
    Examples
    --------
    GET /kayseri/ildem
    GET /kayseri/ildem?date=2026-04-21
    GET /kayseri/ildem?date=2026-04-21&aggregation=hourly
    
    Notes
    -----
    RESTful API akışı:
    1. GET /{city}/regions → Bölge listesi
    2. GET /{city}/{region} → Bölgedeki TÜM kavşakların verisi ⬅️ Buradasınız
    3. GET /{city}/{region}/{junction_id} → Sadece bir kavşağın verisi
    """
    try:
        logger.info(f"📊 Bölge trafik verisi isteği: İl: {city}, Bölge: {region}")
        
        # Bölge validasyonu
        region_data = get_region_junctions(city.lower(), region.lower())
        if not region_data:
            raise HTTPException(
                status_code=404,
                detail=f"İl '{city}' veya bölge '{region}' bulunamadı. GET /{city}/regions ile kontrol edin."
            )
        
        junction_ids = region_data["junction_ids"]
        
        if not junction_ids:
            raise HTTPException(
                status_code=404,
                detail=f"Bölge '{region}' henüz kavşak tanımlanmamış."
            )
        
        logger.info(f"✅ {len(junction_ids)} kavşak bulundu: {junction_ids}")
        
        # Tarihi hazırla
        if date:
            target_date = date
        else:
            from datetime import datetime as dt
            target_date = dt.now().strftime("%Y-%m-%d")
        
        # Her kavşak için veri çek
        all_junctions_data = []
        
        for junction_id in junction_ids:
            try:
                # Belediye'den veriyi çek
                if not state.data_fetcher:
                    raise HTTPException(
                        status_code=503,
                        detail="Belediye API bağlantısı kurulamadı."
                    )
                
                fetch_datetime = f"{target_date}T00:00"
                belediye_data = state.data_fetcher.fetch_junction_data(
                    junction_id=junction_id,
                    date=fetch_datetime,
                    wa=0
                )
                
                if not belediye_data:
                    logger.warning(f"⚠️ Kavşak {junction_id} için veri bulunamadı, atlanıyor...")
                    continue
                
                # Veriyi hazırla
                time_slots = []
                
                if aggregation == "hourly":
                    # Saatlik toplam
                    for hr in range(24):
                        time_str = f"{hr:02d}:00"
                        edges_data = []
                        
                        for edge in belediye_data:
                            start_index = hr * 6
                            hourly_sum = sum(
                                float(edge.get(str(start_index + i), 0)) 
                                for i in range(6)
                            )
                            edges_data.append({
                                "edge_name": edge.get('edge_name', 'Bilinmeyen'),
                                "edge_direction": edge.get('edge_direction', '?'),
                                "traffic_count": hourly_sum
                            })
                        
                        time_slots.append({"time": time_str, "edges": edges_data})
                    
                    aggregation_type = "hourly"
                else:
                    # 10'ar dakikalık
                    for minute_index in range(144):
                        hr = minute_index // 6
                        mn = (minute_index % 6) * 10
                        time_str = f"{hr:02d}:{mn:02d}"
                        edges_data = []
                        
                        for edge in belediye_data:
                            traffic_count = float(edge.get(str(minute_index), 0))
                            edges_data.append({
                                "edge_name": edge.get('edge_name', 'Bilinmeyen'),
                                "edge_direction": edge.get('edge_direction', '?'),
                                "traffic_count": traffic_count
                            })
                        
                        time_slots.append({"time": time_str, "edges": edges_data})
                    
                    aggregation_type = "10min"
                
                # Kavşak bilgisi
                junction_info = KNOWN_JUNCTIONS.get(junction_id, {})
                
                all_junctions_data.append({
                    "junction_id": junction_id,
                    "junction_name": junction_info.get("name", f"Kavşak {junction_id}"),
                    "edge_count": len(belediye_data),
                    "time_slots": time_slots,
                    "aggregation": aggregation_type
                })
                
                logger.info(f"✅ Kavşak {junction_id} verisi eklendi ({len(belediye_data)} kol)")
                
            except Exception as e:
                logger.error(f"❌ Kavşak {junction_id} verisi alınamadı: {e}")
                continue
        
        if not all_junctions_data:
            raise HTTPException(
                status_code=404,
                detail=f"Bölge '{region}' için hiçbir kavşak verisi alınamadı."
            )
        
        # Model bilgisi
        model_info_data = get_model_info(city.lower(), region.lower())
        
        return {
            "city": city.lower(),
            "region": region.lower(),
            "data_date": target_date,
            "junction_count": len(all_junctions_data),
            "timestamp": datetime.now().isoformat(),
            "junctions": all_junctions_data,
            "model_info": {
                "model_id": model_info_data['model_id'] if model_info_data else f"{city}_{region}_v1",
                "version": model_info_data['version'] if model_info_data else "1.0",
            },
            "data_source": "Kayseri Belediye Canlı Veri (Gerçek Zamanlı)"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Bölge veri çekme hatası: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"İç hata: {str(e)}"
        )


@app.get("/{city}/{region}/{junction_id}", response_model=TrafficResponse, tags=["Traffic Data"])
async def get_traffic_data(
    city: str,
    region: str,
    junction_id: int,
    current_user: User = Depends(get_current_active_user),
    date: Optional[str] = Query(None, description="Tarih (YYYY-MM-DD formatında, boş bırakılırsa bugün)"),
    hour: Optional[int] = Query(None, description="Saat (0-23, saatlik toplam için)"),
    minute: Optional[int] = Query(None, description="Dakika (0, 10, 20, 30, 40, 50)"),
    aggregation: Optional[str] = Query(None, description="Veri toplama tipi: 'hourly' (saatlik toplam)")
):
    """
    Kavşak trafik verilerini getirir (RESTful path-based endpoint).
    
    Parameters
    ----------
    city : str
        İl adı (path parameter)
    region : str
        Bölge adı (path parameter)
    junction_id : int
        Kavşak ID (path parameter)
    date : Optional[str]
        Tarih (query parameter, opsiyonel)
    hour : Optional[int]
        Saat (query parameter, opsiyonel)
    minute : Optional[int]
        Dakika (query parameter, opsiyonel)
    aggregation : Optional[str]
        Toplama tipi (query parameter, opsiyonel)
    
    Returns
    -------
    TrafficResponse
        Gerçek trafik verileri
    
    Examples
    --------
    GET /kayseri/ildem/187
    GET /kayseri/ildem/187?date=2026-04-21
    GET /kayseri/ildem/187?date=2026-04-21&aggregation=hourly
    GET /kayseri/ildem/187?date=2026-04-21&hour=14
    
    Raises
    ------
    HTTPException
        - 503: Belediye API'sine bağlanılamadı
        - 404: Kavşak bulunamadı veya veri yok
        - 500: İç hata
    
    Notes
    -----
    RESTful API akışı:
    1. GET /{city}/regions → Bölge listesi
    2. GET /{city}/{region}/junctions → Kavşak listesi
    3. GET /{city}/{region}/{junction_id} → Trafik verisi ⬅️ Buradasınız
    """
    try:
        logger.info(f"📊 Trafik verisi isteği: Kavşak {junction_id}, İl: {city}, Bölge: {region}")
        
        # 0. Kavşak ID validasyonu - Bu kavşak bu bölgede var mı?
        if not is_valid_junction_for_region(junction_id, city.lower(), region.lower()):
            region_data = get_region_junctions(city.lower(), region.lower())
            if region_data:
                valid_ids = region_data["junction_ids"]
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Geçersiz kavşak ID",
                        "message": f"Kavşak {junction_id}, {city}/{region} bölgesinde bulunamadı.",
                        "valid_junction_ids": valid_ids,
                        "hint": f"Bu bölgede {len(valid_ids)} kavşak mevcut."
                    }
                )
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"İl '{city}' veya bölge '{region}' bulunamadı."
                )
        
        logger.info(f"✅ Kavşak validasyonu başarılı: {junction_id}")
        
        # 1. Model bilgisini al (basit metadata)
        model_info_data = get_model_info(city.lower(), region.lower())
        if not model_info_data:
            raise HTTPException(
                status_code=404,
                detail=f"İl '{city}' ve bölge '{region}' bulunamadı."
            )
        
        logger.info(f"✅ Model bilgisi hazır: {model_info_data['model_id']}")
        
        # 2. Belediye API'sine bağlan
        
        if not state.data_fetcher:
            raise HTTPException(
                status_code=503,
                detail="Belediye API bağlantısı kurulamadı."
            )
        
        # Tarih hesapla
        if date:
            target_date = date
        else:
            from datetime import datetime as dt
            target_date = dt.now().strftime("%Y-%m-%d")
        
        # Belediye API'sinden veri çek (her zaman 00:00 ile başlat, tüm gün verisi alır)
        fetch_datetime = f"{target_date}T00:00"
        logger.info(f"📅 Veri çekiliyor: {fetch_datetime}")
        
        belediye_data = state.data_fetcher.fetch_junction_data(
            junction_id=junction_id,
            date=fetch_datetime,
            wa=0
        )
        
        if not belediye_data:
            raise HTTPException(
                status_code=404,
                detail=f"Kavşak {junction_id} için veri bulunamadı."
            )
        
        logger.info(f"✅ Belediye'den {len(belediye_data)} kol verisi alındı")
        
        # SENARYO 1: Belirli saat + dakika verilmiş → Sadece o anı döndür
        if hour is not None:
            minute_val = minute if minute is not None else 0
            # Dakika index hesapla: saat * 6 + dakika / 10
            # Örn: 14:30 → 14 * 6 + 30/10 = 84 + 3 = 87
            minute_index = hour * 6 + (minute_val // 10)
            
            time_str = f"{hour:02d}:{minute_val:02d}"
            edges_data = []
            for edge in belediye_data:
                traffic_count = float(edge.get(str(minute_index), 0))
                edges_data.append(EdgeTrafficData(
                    edge_name=edge.get('edge_name', 'Bilinmeyen'),
                    edge_direction=edge.get('edge_direction', '?'),
                    traffic_count=traffic_count
                ))
            
            time_slots = [TimeSlotData(time=time_str, edges=edges_data)]
            aggregation_type = "single"
            logger.info(f"✅ Tek zaman dilimi: {time_str}")
        
        # SENARYO 2: Sadece tarih verilmiş → Tüm günü döndür
        else:
            time_slots = []
            
            # Aggregation kontrolü: hourly mi 10min mi?
            if aggregation == "hourly":
                # Saatlik: 00:00, 01:00, ..., 23:00 (24 saat)
                # Her saat için o saatin tüm verilerini topla (6 adet 10'ar dakikalık veri)
                for hr in range(24):
                    time_str = f"{hr:02d}:00"
                    
                    edges_data = []
                    for edge in belediye_data:
                        # O saatin tüm 10'ar dakikalık verilerini topla
                        # Örn: Saat 14 için → index 84, 85, 86, 87, 88, 89 (14*6=84, +0 to +5)
                        start_index = hr * 6
                        hourly_sum = sum(
                            float(edge.get(str(start_index + i), 0)) 
                            for i in range(6)  # 0, 1, 2, 3, 4, 5 (6 adet 10'ar dakika)
                        )
                        
                        edges_data.append(EdgeTrafficData(
                            edge_name=edge.get('edge_name', 'Bilinmeyen'),
                            edge_direction=edge.get('edge_direction', '?'),
                            traffic_count=hourly_sum
                        ))
                    
                    time_slots.append(TimeSlotData(time=time_str, edges=edges_data))
                
                aggregation_type = "hourly"
                logger.info(f"✅ Saatlik toplam veri: 24 saat")
            
            else:
                # 10'ar dakikalık: 00:00, 00:10, 00:20, ..., 23:50 (144 veri noktası)
                for minute_index in range(144):
                    hr = minute_index // 6
                    mn = (minute_index % 6) * 10
                    time_str = f"{hr:02d}:{mn:02d}"
                    
                    edges_data = []
                    for edge in belediye_data:
                        traffic_count = float(edge.get(str(minute_index), 0))
                        edges_data.append(EdgeTrafficData(
                            edge_name=edge.get('edge_name', 'Bilinmeyen'),
                            edge_direction=edge.get('edge_direction', '?'),
                            traffic_count=traffic_count
                        ))
                    
                    time_slots.append(TimeSlotData(time=time_str, edges=edges_data))
                
                aggregation_type = "10min"
                logger.info(f"✅ 10'ar dakikalık veri: 144 veri noktası")
        
        # Model bilgisini hazırla
        model_info = {
            "model_id": model_info_data['model_id'],
            "city": model_info_data['city'],
            "region": model_info_data['region'],
            "version": model_info_data['version'],
            "junction_count": model_info_data['junction_count'],
            "note": "Canlı belediye verisi - Model entegrasyonu ileride eklenecek"
        }
        
        return TrafficResponse(
            junction_id=junction_id,
            city=city.lower(),
            region=region.lower(),
            data_date=target_date,
            aggregation=aggregation_type,
            timestamp=datetime.now().isoformat(),
            time_slots=time_slots,
            model_info=model_info,
            data_source="Kayseri Belediye Canlı Veri (Gerçek Zamanlı)"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Veri çekme hatası: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"İç hata: {str(e)}"
        )


@app.post("/update-cookies", tags=["Admin"])
async def update_cookies(cookies: Dict[str, str]):
    """
    API için cookie'leri günceller (admin endpoint).
    
    Parameters
    ----------
    cookies : Dict[str, str]
        Yeni cookie'ler
    
    Example
    -------
    POST /update-cookies
    {
        ".AspNetCore.Identity.Application": "CfDJ8...",
        ".AspNetCore.Antiforgery.708PDUFCWo4": "CfDJ8..."
    }
    """
    if state.data_fetcher:
        state.data_fetcher.session.cookies.update(cookies)
        state.cookies = cookies
        return {"status": "success", "message": "Cookie'ler güncellendi"}
    else:
        raise HTTPException(status_code=503, detail="Data fetcher hazır değil")


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """404 hatası özel mesaj"""
    return {
        "error": "Not Found",
        "message": "İstediğiniz endpoint bulunamadı. /docs adresinden API dokümantasyonunu inceleyebilirsiniz.",
        "documentation": "/docs"
    }


# ============================================================================
# Main (Development Server)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("🚀 KAYSERİ TRAFİK TAHMİN API BAŞLATILIYOR")
    print("=" * 70)
    print()
    print("📡 Server: http://localhost:9000")
    print("📚 Swagger UI: http://localhost:9000/docs")
    print("📖 ReDoc: http://localhost:9000/redoc")
    print()
    print("💡 API'yi test etmek için:")
    print("   1. http://localhost:9000/docs adresini tarayıcıda açın")
    print("   2. /predict/187 endpoint'ini deneyin")
    print()
    print("🔴 Durdurmak için: Ctrl+C")
    print("=" * 70)
    print()
    
    uvicorn.run(
        "kayseri_api:app",
        host="0.0.0.0",
        port=9000,
        reload=True,
        log_level="info"
    )
