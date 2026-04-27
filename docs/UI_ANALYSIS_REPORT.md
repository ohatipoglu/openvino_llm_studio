# UI Analiz Raporu - 3 Farklı Arayüz Karşılaştırması

**Tarih:** 2026-04-27  
**Proje:** OpenVINO LLM Studio

---

## 1. Genel Bakış

Projede **3 farklı Gradio tabanlı kullanıcı arayüzü** bulunmaktadır:

| UI | Dosya | Port | Durum |
|----|-------|------|-------|
| **Klasik UI** | `ui/app.py` | 7860 | Stabil |
| **Modern UI** | `ui/app_modern.py` | 7861 | Önerilen |
| **Workspace UI** | `ui/app_workspace.py` | 7862 | Deneysel |

---

## 2. Her UI'nin Özellikleri

### 2.1 Klasik UI (`app.py`)

**Mimari:**
- Tek ekran, sekme tabanlı düzen
- Tüm ayarlar aynı anda görünür
- Sol panel: Model yükleme, backend seçimi, Ollama pull
- Orta panel: Chat arayüzü
- Sağ panel: Loglar ve istatistikler

**Özellikler:**
- Backend seçimi (OpenVINO, Ollama, IPEX/llama-cpp)
- Detaylı GPU/CPU ayarları (GPU_MAX_ALLOC_PERCENT, KV_CACHE_PRECISION, NUM_STREAMS, etc.)
- Model katalogları (OpenVINO, Ollama, GGUF)
- Web arama ve DSPy zenginleştirme
- Log görüntüleme (Search, DSPy, LLM, Error, General)
- İstatistik paneli

**Kullanılan Orchestrator Metodları:**
```python
- get_model_choices()
- load_model()
- get_backend_status()
- run_pipeline()
- get_logs()
- clear_logs()
- get_stats()
- get_openvino_catalog()
- get_ollama_catalog()
- get_ipex_catalog()
- download_openvino_model()
- download_gguf_model()
- pull_ollama_model()
- ipex.start_worker()
- ipex.stop_worker()
- ipex.scan_local_gguf()
```

---

### 2.2 Modern UI (`app_modern.py`)

**Mimari:**
- 3 adımlı workflow (Model → Ayarlar → Chat)
- Kullanıcı modları (Basit/Orta/Expert)
- Progressive disclosure (kademeli bilgi gösterimi)
- Real-time CPU/RAM dashboard
- Preset profiller (Hızlı/Dengeli/Kaliteli)

**Özellikler:**
- Adımlı navigasyon (Step 1: Model, Step 2: Ayarlar, Step 3: Chat)
- Kullanıcı moduna göre UI elemanları gizleme/gösterme
- HTML tabanlı durum göstergeleri
- Sistem prompt desteği
- Preset selector ile hızlı ayar uygulama

**Kullanılan Orchestrator Metodları:**
```python
- get_model_choices()
- load_model()
- get_backend_status()
- run_pipeline()
- get_logs()
- clear_logs()
- get_openvino_catalog()
- get_ollama_catalog()
- get_ipex_catalog()
- download_gguf_model()
- pull_ollama_model()
```

**NOT:** `ipex.start_worker()` ve `ipex.stop_worker()` metodları **KULLANILMIYOR**.

---

### 2.3 Workspace UI (`app_workspace.py`)

**Mimari:**
- Sidebar + Ana Kanvas düzeni
- Minimal header, bilgilendirici footer
- Sürekli erişilebilir ayarlar (Sidebar)
- Merkezde sohbet (Ana Kanvas)
- Accordion'lar ile organize edilmiş ayarlar

**Özellikler:**
- Kompakt tasarım
- Footer durum çubuğu (sabit pozisyon)
- LLM parametreleri accordion
- Arama & DSPy accordion
- Log modal (açılır pencere)
- Kullanıcı modu seçici (Basit/Orta/Expert)

**Kullanılan Orchestrator Metodları:**
```python
- get_model_choices()
- load_model()
- run_pipeline()
- get_logs()
- get_openvino_catalog()
- get_ollama_catalog()
- get_ipex_catalog()
- download_gguf_model()
- pull_ollama_model()
```

**NOT:** 
- `get_backend_status()` **KULLANILMIYOR** (sadece `get_system_status()` kullanılıyor)
- `ipex.start_worker()` ve `ipex.stop_worker()` **KULLANILMIYOR**
- `clear_logs()` **KULLANILMIYOR**

---

## 3. Tespit Edilen Çelişkiler ve Sorunlar

### 🔴 KRİTİK: Singleton Orchestrator Paylaşımı

**Sorun:**
```python
# Tüm UI'larda aynı pattern:
_orch_lock = threading.Lock()
_orch = None

def _get_orch() -> Orchestrator:
    global _orch
    with _orch_lock:
        if _orch is None:
            _orch = Orchestrator()
        return _orch
```

**Problem:**
- Her UI **ayrı bir global `_orch` değişkeni** kullanıyor
- Ancak `Orchestrator` sınıfı **stateful** (durumlu):
  - `_active_backend` (örn: "openvino", "ollama", "ipex")
  - `session_id` (örn: "sess_1234567890")
  - `_ov_models` (cache'lenmiş model listesi)
  - `searcher` (arama cache'i)
  - `enricher` (DSPy state)

**Senaryo:**
1. Kullanıcı Klasik UI'da (port 7860) backend'i "ollama" yapar
2. Aynı anda Modern UI'da (port 7861) "openvino" seçer
3. **ÇAKIŞMA:** Orchestrator'un `_active_backend` değeri son yazana göre belirlenir
4. Her iki UI da yanlış backend durumunu gösterebilir

**Etki:**
- Backend durum tutarsızlığı
- Model yükleme hataları
- Session karışıklığı (farklı UI'larda farklı session_id)

---

### 🟡 ORTA: Farklı Log Dosyaları

**Durum:**
```python
# app.py
logging.FileHandler(LOG_DIR / "studio.log")

# app_modern.py
logging.FileHandler(LOG_DIR / "studio_modern.log")

# app_workspace.py
logging.FileHandler(LOG_DIR / "studio_workspace.log")
```

**Analiz:**
- ✅ **Avantaj:** Her UI'nin logları ayrı, debug kolay
- ⚠️ **Dezavantaj:** Merkezi log analizi zor
- ⚠️ **Dezavantaj:** Database'deki loglar ile dosya logları senkronize değil

**Öneri:**
- Tek merkezi log dosyası + UI identifier
- Veya log aggregator kullanımı

---

### 🟡 ORTA: Session Yönetimi

**Durum:**
```python
# Orchestrator.__init__()
self.session_id = f"sess_{int(time.time())}"

# new_session() metodu var ama sadece Modern UI kullanıyor olabilir
```

**Problem:**
- Session ID Orchestrator instance'ında saklanıyor
- Her UI farklı Orchestrator instance'ı kullanıyorsa → farklı session_id
- Database logları session_id ile filtreleniyor
- Kullanıcı bir UI'da yaptığı sohbeti başka UI'da göremez

**Etki:**
- Log izolasyonu (aslında feature olabilir)
- Cross-UI state paylaşımı yok

---

### 🟢 DÜŞÜK: GPU Ayarları Tutarsızlığı

**Durum:**

| UI | GPU Ayarları | Varsayılan Değerler |
|----|--------------|---------------------|
| Klasik | GPU_MAX_ALLOC_PERCENT, KV_CACHE_PRECISION, NUM_STREAMS, PERF_HINT, CACHE_DIR | 75%, u8, 1, LATENCY, CACHE_DIR |
| Modern | Aynı + daha fazla tooltip | Aynı |
| Workspace | Aynı, ama Expert modda görünür | Aynı |

**Problem:**
- `CACHE_DIR` parametresi Workspace UI'da **KULLANILMIYOR**
- `load_model_action()` imzası farklı:
  - Klasik: `load_model_action(model_choice, device, gpu_max_alloc, kv_cache_prec, num_streams, perf_hint, cache_dir)`
  - Modern: `load_model_action(model_choice, device, gpu_max_alloc, kv_cache_prec, num_streams, perf_hint, cache_dir)`
  - Workspace: `load_model_action(model_choice, device, gpu_max_alloc, kv_cache_prec, num_streams, perf_hint)` ← **cache_dir eksik!**

**Etki:**
- Workspace UI'da OpenVINO cache'i yapılandırılamıyor
- Performans etkisi (cache disabled)

---

### 🟢 DÜŞÜK: Pipeline Parametre Farklılıkları

**Durum:**

```python
# Klasik UI - run_inference
orch.run_pipeline(
    prompt=prompt,
    params=params,
    enable_search=enable_search,
    enable_dspy=enable_dspy,
    num_search_results=int(num_search_results),
    system_prompt=system_prompt,
    history=chat_history  # <-- history var
)

# Modern UI - run_inference
orch.run_pipeline(
    prompt=prompt,
    params=params,
    enable_search=enable_search,
    enable_dspy=enable_dspy,
    num_search_results=int(num_search_results),
    system_prompt=system_prompt,
    # history YOK!
)

# Workspace UI - run_inference
orch.run_pipeline(
    prompt=prompt,
    params=params,
    enable_search=enable_search,
    enable_dspy=enable_dspy,
    num_search_results=int(num_search_results),
    # system_prompt YOK!
    # history YOK!
)
```

**Problem:**
- Workspace UI'da `system_prompt` parametresi **KULLANILMIYOR**
- Modern ve Workspace UI'larda chat history **GÖNDERİLMİYOR**
- Orchestrator `run_pipeline()` metodu history parametresini destekliyor ama tüm UI'lar kullanmıyor

**Etki:**
- Chat bağlamı kayboluyor (multi-turn conversations çalışmıyor)
- Sistem prompt özelleştirmesi Workspace UI'da yok

---

### 🟢 DÜŞÜK: Backend Durum Gösterimi

**Durum:**

| UI | Backend Status | Güncelleme |
|----|----------------|------------|
| Klasik | `get_backend_status()` | Manuel refresh |
| Modern | `get_backend_status_display()` | Manuel refresh |
| Workspace | **YOK** (sadece footer'da backend adı) | Footer otomatik güncelleniyor |

**Problem:**
- Workspace UI'da backend detaylı durumu yok (loaded/available)
- Kullanıcı backend'in hazır mı yoksa yüklü mü olduğunu ayırt edemiyor

---

### 🟢 DÜŞÜK: Model Kataloğu Refresh Stratejisi

**Durum:**

```python
# Klasik UI
refresh_ov_catalog(search, force=False)  # 30 sn timeout
refresh_ollama_catalog(search, force=False)
refresh_gguf_catalog(search, force=False)

# Modern UI
refresh_models()  # Timeout yok
# Kataloglar ayrı ayrı yenilenmiyor

# Workspace UI
refresh_models()  # Timeout yok
```

**Problem:**
- Klasik UI'da catalog refresh için 30 saniye timeout var
- Modern ve Workspace UI'larda timeout yok → uzun bekleme
- Force refresh seçeneği sadece Klasik UI'da var

---

### 🟢 DÜŞÜK: Log Temizleme

**Durum:**

| UI | Log Temizleme | Tablo Seçimi |
|----|---------------|--------------|
| Klasik | `clear_logs_action(table_choice)` | ✅ (Tümü, Arama, DSPy, LLM, Hata, Genel) |
| Modern | `clear_logs_action(table_choice)` | ✅ (Aynı) |
| Workspace | **YOK** | ❌ |

**Problem:**
- Workspace UI'da log temizleme fonksiyonu yok
- Log modal var ama temizleme butonu yok

---

### 🟢 DÜŞÜK: llama-cpp-python Yönetimi

**Durum:**

```python
# Klasik UI
check_llamacpp()      # ipex.start_worker()
unload_llamacpp()     # ipex.stop_worker()

# Modern UI
# YOK

# Workspace UI
# YOK
```

**Problem:**
- Sadece Klasik UI'da llama-cpp-python process yönetimi var
- Modern ve Workspace UI'larda kullanıcı IPEXWorkerClient'ı başlatamıyor/durduramıyor
- IPEX backend kullanımı Klasik UI ile sınırlı

---

## 4. Orchestrator State Çelişkisi Detaylı Analiz

### State Değişkenleri

```python
class Orchestrator:
    def __init__(self):
        self._active_backend = "openvino"  # ← STATE
        self.session_id = f"sess_{...}"    # ← STATE
        self._ov_models = []                # ← STATE (cache)
        
        # Backend instances (paylaşılan)
        self.ov_loader = ModelLoader(...)
        self.ipex = IPEXWorkerClient(...)
        self.ollama = OllamaBackend(...)
```

### Çakışma Senaryoları

#### Senaryo 1: Backend Değiştirme Çakışması

```
Zaman    | Klasik UI (Thread A)      | Modern UI (Thread B)
---------|---------------------------|------------------------
T0       | _active_backend = openvino| _active_backend = openvino
T1       | set_backend("ollama")     |
T2       | _active_backend = ollama  | set_backend("ipex")
T3       |                           | _active_backend = ipex
T4       | get_model_choices() →     | get_model_choices() →
         | Ollama modelleri bekler   | GGUF modelleri döner
         | ama ipex döner! ❌        |
```

#### Senaryo 2: Session ID Karışıklığı

```
Zaman    | Klasik UI (Thread A)      | Modern UI (Thread B)
---------|---------------------------|------------------------
T0       | session_id = sess_1000    | session_id = sess_1000
T1       | new_session()             |
T2       | session_id = sess_2000    | new_session()
T3       |                           | session_id = sess_3000
T4       | log_search(sess_2000)     | log_search(sess_3000)
T5       | get_logs(session_only=T)  | get_logs(session_only=T)
         | → sess_2000 logları       | → sess_3000 logları
         | (farklı UI, farklı log) ✅|
```

**NOT:** Bu aslında bir feature olabilir (UI izolasyonu). Ancak kullanıcı aynı session'ı paylaşmak isterse problem.

---

## 5. Öneriler

### 5.1 KRİTİK: Orchestrator State Yönetimi

**Seçenek A: UI-Bazlı Orchestrator İzolasyonu**
```python
# Her UI kendi Orchestrator instance'ını kullansın
# State çakışması olmaz
# Ancak memory overhead artar (3x DatabaseManager, 3x WebSearcher, etc.)
```

**Seçenek B: Merkezi State Yönetimi**
```python
# Orchestrator state'ini UI'dan bağımsız yap
# Backend seçimi UI-local olsun (Orchestrator'da saklanmasın)
# Session ID user-specific olsun (cookie/token ile)
```

**Seçenek C: Thread-Local State**
```python
# threading.local() ile her thread için ayrı state
# Ancak Gradio thread pool kullanıyorsa çalışmayabilir
```

**ÖNERİ:** **Seçenek B** - En temiz çözüm. Orchestrator stateless'e yakın çalışsın, UI state'i UI'da saklansın.

---

### 5.2 ORTA: Tutarlı API Kullanımı

**Pipeline Parametreleri:**
- Tüm UI'lar `run_pipeline()` metoduna **aynı parametreleri** göndermeli
- En azından: `prompt, params, enable_search, enable_dspy, num_search_results, system_prompt, history`
- Workspace UI güncellenmeli: `system_prompt` ve `history` eklenmeli

**GPU Ayarları:**
- Workspace UI'ya `cache_dir` parametresi eklenmeli
- Veya tüm UI'larda cache_dir varsayılan değeri kullanılmalı

---

### 5.3 DÜŞÜK: Özellik Eşitleme

**Eksik Özellikler:**

| Özellik | Klasik | Modern | Workspace | Eklenecek UI |
|---------|--------|--------|-----------|--------------|
| llama-cpp start/stop | ✅ | ❌ | ❌ | Modern, Workspace |
| Log temizleme | ✅ | ✅ | ❌ | Workspace |
| Backend detay durum | ✅ | ✅ | ❌ | Workspace |
| Force catalog refresh | ✅ | ❌ | ❌ | Modern, Workspace |
| System prompt | ✅ | ✅ | ❌ | Workspace |
| Chat history | ✅ | ❌ | ❌ | Modern, Workspace |

---

### 5.4 DÜŞÜK: Log Stratejisi Birleştirme

**Öneri:**
```python
# Tek log dosyası, UI identifier ile
LOG_DIR / f"studio_{ui_type}.log"  # yerine
LOG_DIR / "studio.log"  # + log formatına UI identifier ekle

# Format: "%(asctime)s [%(levelname)s] [UI:app] %(name)s: %(message)s"
```

---

## 6. Sonuç

### Çelişki Özeti

| Kategori | Sorun Sayısı | Kritiklik |
|----------|--------------|-----------|
| State Yönetimi | 1 | 🔴 KRİTİK |
| API Tutarlılığı | 2 | 🟡 ORTA |
| Özellik Parçalanması | 6 | 🟢 DÜŞÜK |
| Log Yönetimi | 1 | 🟢 DÜŞÜK |

### Genel Değerlendirme

**İyi Yanlar:**
- ✅ Her UI farklı kullanım senaryolarına hitap ediyor (klasik, modern, workspace)
- ✅ Orchestrator merkezi iş mantığını iyi soyutluyor
- ✅ DatabaseManager thread-safe
- ✅ Backend'ler (OpenVINO, Ollama, IPEX) iyi ayrılmış

**Kötü Yanlar:**
- ❌ Orchestrator stateful olması çoklu UI kullanımında çakışmaya yol açıyor
- ❌ UI'lar arasında özellik farkları var (feature parity yok)
- ❌ Chat history desteği tüm UI'larda yok
- ❌ Workspace UI eksik parametreler içeriyor

### Tavsiye

**Kısa Vadeli:**
1. Workspace UI'ya eksik parametreleri ekle (`system_prompt`, `cache_dir`, `history`)
2. Modern ve Workspace UI'lara chat history desteği ekle
3. Orchestrator state yönetimini dokümante et

**Orta Vadeli:**
1. Orchestrator'u stateless yap (backend seçimi UI-local olsun)
2. Session yönetimini kullanıcı bazına çevir (cookie/token)
3. Özellik paritesini sağla (tüm UI'lar aynı core özelliklere sahip olsun)

**Uzun Vadeli:**
1. UI'ları birleştir veya tek UI'ı resmi yap
2. Backend state yönetimini database'e taşı
3. Multi-user desteği ekle (authentication, user-specific sessions)

---

## 7. Ek: Orchestrator Metod Kullanım Matrisi

| Metod | Klasik | Modern | Workspace | Not |
|-------|--------|--------|-----------|-----|
| `get_model_choices()` | ✅ | ✅ | ✅ | |
| `load_model()` | ✅ | ✅ | ✅ | |
| `set_backend()` | ✅ | ✅ | ✅ | |
| `get_backend_status()` | ✅ | ✅ | ❌ | Workspace: sadece backend adı |
| `run_pipeline()` | ✅ | ✅ | ✅ | Parametreler farklı! |
| `get_logs()` | ✅ | ✅ | ✅ | |
| `clear_logs()` | ✅ | ✅ | ❌ | |
| `get_stats()` | ✅ | ❌ | ❌ | |
| `get_openvino_catalog()` | ✅ | ✅ | ✅ | |
| `get_ollama_catalog()` | ✅ | ✅ | ✅ | |
| `get_ipex_catalog()` | ✅ | ✅ | ✅ | |
| `download_openvino_model()` | ✅ | ❌ | ❌ | |
| `download_gguf_model()` | ✅ | ✅ | ✅ | |
| `pull_ollama_model()` | ✅ | ✅ | ✅ | |
| `new_session()` | ❌ | ✅ | ❌ | |
| `ipex.start_worker()` | ✅ | ❌ | ❌ | |
| `ipex.stop_worker()` | ✅ | ❌ | ❌ | |
| `ipex.scan_local_gguf()` | ✅ | ❌ | ✅ | |
