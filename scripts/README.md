# 📝 Scripts - Production Utilities

Production ortamı için gerekli yönetim scriptleri.

---

## 🗃️ `init_db.py`

**Amaç:** Veritabanını başlatır ve seed verilerini ekler.

**Kullanım:**
```bash
python scripts/init_db.py
```

**Ne Yapar:**
1. PostgreSQL tablolarını oluşturur (regions, junctions, model_artifacts)
2. Bölge verilerini ekler (Kayseri, Konya, Sivas)
3. Kavşak verilerini ekler (node sayıları ile birlikte)

**Ne Zaman Kullanılır:**
- İlk kurulumda (bir kez)
- Veritabanı sıfırlandığında
- Yeni bir ortama deployment yapılırken

**Gereksinimler:**
- `.env` dosyasında `DATABASE_URL` tanımlanmış olmalı
- PostgreSQL çalışıyor olmalı

---

## 📤 `upload_model.py`

**Amaç:** Eğitilmiş model ağırlıklarını (checkpoint) veritabanına yükler.

**Kullanım:**

**Örnek 1: StandardScaler ile**
```bash
python scripts/upload_model.py \
    --junction-code kayseri_ildem \
    --weights-path saved_models/kayseri_ildem_v3.pth \
    --scaler-mean 12.4 \
    --scaler-std 5.3
```

**Örnek 2: MinMaxScaler ile**
```bash
python scripts/upload_model.py \
    --junction-code sivas \
    --weights-path saved_models/sivas_best.pth \
    --scaler-type max01 \
    --scaler-min 0.0 \
    --scaler-max 120.0
```

**Parametreler:**
- `--junction-code`: Kavşak kodu (DB'de tanımlı olmalı)
- `--weights-path`: .pth checkpoint dosyası yolu
- `--scaler-type`: `standardize` (default) veya `max01`
- `--scaler-mean`: StandardScaler için ortalama
- `--scaler-std`: StandardScaler için std sapma
- `--scaler-min`: MinMaxScaler için min değer
- `--scaler-max`: MinMaxScaler için max değer
- `--algo`: Model türü (default: `Garnoldi`)
- `--api-key`: Admin API anahtarı (varsayılan: `.env` dosyasından)

**Ne Yapar:**
1. Junction ID'yi kontrol eder
2. Checkpoint'i BLOB olarak okur
3. Model config'i checkpoint'ten çıkarır
4. Eski artifact'ı devre dışı bırakır (`is_active=False`)
5. Yeni artifact'ı yükler ve aktif eder
6. Version otomatik artırılır

**Ne Zaman Kullanılır:**
- Yeni bir model eğitildikten sonra
- Model güncellemesi yapılırken
- Farklı bir kavşak için model eklenirken

**Önemli Notlar:**
- ⚠️ Bu script eski modeli devre dışı bırakır (rollback için DB backup alın)
- ✅ Checkpoint'te `register_buffer('adj', ...)` olmalı (graph embedding)
- ✅ Model config otomatik olarak state_dict'ten okunur

---

## 🗑️ Silinen Scripts (Artık Gereksiz)

### Migration Scripts (Tek Seferlik)
- ❌ `resave_checkpoint.py` - Checkpoint'lere buffer ekleme (tamamlandı)
- ❌ `fix_junctions.py` - DB junction kayıtlarını düzeltme (tamamlandı)
- ❌ `sync_model_config.py` - Model config senkronizasyonu (tamamlandı)
- ❌ `migrate_edges_to_db.py` - CSV edges → JSONB migration (tamamlandı)

### Test Scripts (Geçici)
- ❌ `_tmp_check.py` - CSV format test
- ❌ `_tmp_test.py` - API endpoint test

---

## 🔄 Workflow: Yeni Model Deployment

1. **Model eğit:**
   ```bash
   python train.py --config conf/Kayseri_AFDGCN.conf
   ```

2. **Scaler değerlerini kaydet** (training loglarından al)

3. **Model yükle:**
   ```bash
   python scripts/upload_model.py \
       --junction-code kayseri_ildem \
       --weights-path saved_models/model_epoch_100.pth \
       --scaler-mean 12.4 \
       --scaler-std 5.3
   ```

4. **API'den test et:**
   ```bash
   curl -X POST http://localhost:8000/api/v1/predict \
       -H "Content-Type: application/json" \
       -d '{"junction_code": "kayseri_ildem", "input_data": [...]}'
   ```

5. **Model version'ı kontrol et:**
   ```bash
   # Response'da model_version artmış olmalı
   {"junction_code": "kayseri_ildem", "model_version": 4, ...}
   ```

---

## 📂 İlgili Dosyalar

- `.env` - Environment variables (DATABASE_URL, ADMIN_API_KEY)
- `api/crud.py` - Database CRUD operations
- `api/models.py` - SQLAlchemy ORM models
- `saved_models/` - Checkpoint dosyaları (.pth)
- `conf/` - Training configuration files

---

**Son Güncelleme:** Nisan 2026  
**Statü:** ✅ Production Ready
