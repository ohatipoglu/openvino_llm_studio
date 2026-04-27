# ⚡ OpenVINO LLM Studio

Intel Arc iGPU üzerinde yerel LLM çalıştırma ortamı. **llama-server (SYCL)** + **OpenVINO** + **Ollama** backend'lerini tek arayüzde sunar.

**Özellikler:**
- 🔷 **3 Backend**: OpenVINO / Ollama / llama-cpp-python (GGUF/SYCL)
- 🌐 **Web Araması**: DuckDuckGo + BM25 + semantik sıralama
- 🧠 **DSPy**: Otomatik mod seçimi + prompt optimizasyonu
- 🤖 **ReAct Ajan**: Tool calling (fonksiyon çağırma) desteği
- 📊 **SQLite Loglama**: Tüm işlemler `logs/studio.db` içinde

---

## 🚀 Hızlı Başlangıç

### Tek Komutla Başlatma (ÖNERİLEN)

```bat
:: Interaktif başlatma (UI seçimi sorar)
start.bat

:: Direkt Unified UI (YENİ - ÖNERİLEN)
start.bat unified

:: Direkt Modern UI
start.bat modern

:: Direkt Workspace UI
start.bat workspace

:: Direkt Klasik UI
start.bat classic
```

**`start.bat` Neler Yapar:**
1. ✅ Ollama Vulkan başlatır (Arc GPU hızlandırma)
2. ✅ Intel oneAPI ortamını hazırlar
3. ✅ Conda 'openvino_studio' ortamını aktif eder
4. ✅ Seçilen UI'ı başlatır

### Manuel Başlatma

```bat
conda activate openvino_studio
python ui\app_unified.py    :: Unified UI (ÖNERİLEN)
python ui\app_modern.py     :: Modern UI
python ui\app_workspace.py  :: Workspace UI
python ui\app.py            :: Klasik UI
```

**Varsayılan Portlar:**
- Unified UI: `http://127.0.0.1:7860`
- Modern UI: `http://127.0.0.1:7861`
- Workspace UI: `http://127.0.0.1:7862`
- Klasik UI: `http://127.0.0.1:7863`

---

## 🎯 UI Seçenekleri

### 1. Unified UI (YENİ - ÖNERİLEN)

Tek UI'da 3 farklı görünüm modu:

| Mod | Açıklama |
|-----|----------|
| **🏠 Klasik** | Tek ekran, tab navigasyonu |
| **✨ Modern** | 3 adımlı workflow (Model → Ayarlar → Chat) |
| **💼 Workspace** | Sidebar + Ana Kanvas düzeni |

**Özellikler:**
- ✅ 3 mod seçeneği tek UI'da
- ✅ Chat history desteği
- ✅ Log temizleme
- ✅ Backend detay durum göstergesi
- ✅ System prompt desteği

### 2. Modern UI

**Özellikler:**
- 🎯 3 kullanıcı modu (Basit/Orta/Expert)
- 📋 3 adımlı workflow
- ⚡ Preset profiller (Hızlı/Dengeli/Kaliteli)
- 📊 Real-time CPU/RAM dashboard
- 💡 Her slider'da tooltip

### 3. Workspace UI

**Özellikler:**
- 📱 Sidebar + Ana Kanvas düzeni
- 📋 Sürekli erişilebilir ayarlar
- 📊 Minimal header, bilgilendirici footer
- 🗂️ Accordion'lar ile organize ayarlar

### 4. Klasik UI

**Özellikler:**
- 🖥️ Tek ekran arayüz
- ⚙️ Tüm ayarlar görünür
- ✅ Stabil ve alışılmış

---

## 📋 Kullanım Kılavuzu

### 1. Model Seçimi ve Yükleme

1. **Backend seçin**: OpenVINO / Ollama / llama-cpp
2. **Model seçin**: Dropdown'dan seçin veya 🔄 ile yenileyin
3. **Cihaz seçin**: CPU / GPU / AUTO
4. **⚡ Modeli Yükle** butonuna tıklayın

**Ollama için:**
- Model otomatik yüklenir (ilk request'te)
- "Modeli Yükle" butonu ile önceden yüklenebilir
- `keep_alive=-1` ile bellekte tutulur

### 2. Ayarlar

**Preset Seçimi (Hızlı):**
| Preset | Temperature | Max Tokens | Kullanım |
|--------|-------------|------------|----------|
| 🚀 Hızlı | 0.5 | 256 | Düşük kalite, hızlı |
| ⚖️ Dengeli | 0.7 | 512 | Önerilen |
| 🎨 Kaliteli | 0.9 | 1024 | Yüksek kalite, yavaş |

**Manuel Ayarlar:**
- **Temperature**: 0.7 (önerilen) - Düşük=odaklı, Yüksek=yaratıcı
- **Max Tokens**: 512 - Maksimum yanıt uzunluğu
- **Top-P**: 0.9 - Nucleus sampling
- **Top-K**: 50 - En iyi k token'dan örneklem
- **Repetition Penalty**: 1.1 - Tekrarları önle

**Search & DSPy:**
- 🌐 **Web Araması**: DuckDuckGo ile güncel bilgi ara
- 🧠 **DSPy**: Otomatik mod seçimi ve prompt optimizasyonu

### 3. Sohbet

1. Sorunuzu yazın
2. **📤 Gönder** butonuna tıklayın
3. Yanıtı real-time izleyin

**Chat History:**
- Geçmiş sohbetler otomatik saklanır
- "🗑️ Temizle" ile geçmişi silebilirsiniz

---

## 🔧 Yapılandırma

### Port Ayarı

```bat
set GRADIO_SERVER_PORT=7861
python ui\app_unified.py
```

### Host Ayarı

```bat
set GRADIO_SERVER_NAME=0.0.0.0
python ui\app_unified.py
```

### Ollama Timeout (Büyük Modeller)

Timeout varsayılan olarak **300 saniye (5 dakika)**. Daha büyük modeller için:

```python
# modules/ipex_backend.py içinde
timeout=600  # 10 dakika
```

---

## 📁 Proje Yapısı

```
openvino_llm_studio/
├── ui/
│   ├── app.py                 # Klasik UI
│   ├── app_modern.py          # Modern UI
│   ├── app_workspace.py       # Workspace UI
│   ├── app_unified.py         # ✨ Unified UI (YENİ)
│   └── components/
│       └── monitoring.py
├── core/
│   ├── orchestrator.py        # Merkezi yönetim
│   ├── state_manager.py       # ✨ UI-bazlı state yönetimi
│   ├── config.py
│   ├── constants.py
│   └── schema.py
├── modules/
│   ├── database.py            # SQLite loglama
│   ├── search_engine.py       # DuckDuckGo + BM25
│   ├── dspy_enricher.py       # DSPy optimizasyon
│   ├── model_manager.py       # OpenVINO yönetimi
│   ├── ipex_backend.py        # Ollama + llama-cpp
│   ├── ipex_worker_client.py  # llama-server client
│   ├── hf_catalog.py          # HuggingFace katalog
│   └── tools.py               # ReAct araçları
├── logs/                      # SQLite DB
├── start.bat                  # ✨ Tek komut başlatma
├── setup_and_run.bat          # İlk kurulum
└── run.bat                    # Günlük çalıştırma
```

---

## 🏗️ Mimari Akışı

```
[Kullanıcı Girdisi (UI)]
         |
         v
[Orchestrator]
         |
         ├────> [Web Araması] ────> [BM25 + Semantik]
         |
         ├────> [DSPy] ────────────> [Mod Seçimi + Template]
         |
         v
[Backend: OpenVINO / Ollama / llama-cpp]
         |
         v
[LLM Yanıtı] ───> [UI Dashboard]

(Tüm adımlar SQLite'da loglanır)
```

---

## 📋 Sistem Gereksinimleri

| Gereksinim | Detay |
|---|---|
| **OS** | Windows 11 |
| **CPU** | Intel 12. nesil+ (önerilir) |
| **GPU** | Intel Arc iGPU (veya CPU modu) |
| **RAM** | 16 GB+ (32 GB önerilir) |
| **Disk** | 20 GB+ (modeller hariç) |
| **Yazılım** | Anaconda, Intel oneAPI (opsiyonel) |

---

## 📂 Model Dizinleri

**GGUF (llama-cpp-python):**
```
C:\OpenVINO_LLM\gguf\
```

**OpenVINO:**
```
C:\OpenVINO_LLM\
```

**Ollama:**
```
~/.ollama/models/
```

---

## 🔍 Sorun Giderme

### "Model yüklü değil" hatası

**Çözüm:**
1. Backend seçili mi kontrol edin
2. Model dropdown'dan model seçin
3. "⚡ Modeli Yükle" butonuna tıklayın

### Ollama timeout hatası

**Çözüm:**
- Büyük modeller için 5 dakika timeout yeterli olmayabilir
- `modules/ipex_backend.py` içinde `timeout=600` yapın (10 dakika)

### llama-cpp-python bulunamadı

**Çözüm:**
```bat
pip install llama-cpp-python
```

SYCL/XPU desteği için:
```bat
set CMAKE_ARGS=-DGGML_SYCL=ON
pip install llama-cpp-python --no-binary llama-cpp-python
```

### DuckDuckGo rate limit

**Çözüm:**
- Biraz bekleyin
- Arama sonuç sayısını azaltın (5 → 3)

### GPU bellek yetersiz

**Çözüm:**
- GPU Max Alloc % değerini düşürün (75% → 60%)
- KV Cache Precision: u8 kullanın
- Daha küçük model deneyin

---

## 📊 Performans İpuçları

| İyileştirme | Açıklama |
|-------------|----------|
| **GPU Max Alloc %** | 75% önerilir (crash önleme) |
| **KV Cache Precision** | u8 = 2x daha az VRAM |
| **Num Streams** | Büyük modellerde 1 tutun |
| **Performance Hint** | LATENCY = tek kullanıcı |

---

## 🧪 Test

```bat
:: Pytest (test dosyası eklenmeli)
pip install pytest
pytest tests/ -v
```

---

## 📚 Daha Fazla Bilgi

- **UI Analiz Raporu**: `docs/UI_ANALYSIS_REPORT.md`
- **State Management**: `core/state_manager.py`

---

## 📝 License

MIT License
