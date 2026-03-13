# ⚡ OpenVINO LLM Studio

Intel Arc iGPU üzerinde yerel LLM çalıştırma ortamı.  
**llama-server (SYCL/Arc GPU)** + **OpenVINO** + **Ollama** üç backend'i tek arayüzde sunar.  
DuckDuckGo web araması, DSPy prompt zenginleştirme ve SQLite loglama dahildir.

---

## 🖥️ Sistem Gereksinimleri

| Gereksinim | Detay |
|---|---|
| İşletim Sistemi | Windows 11 |
| CPU | Intel (12. nesil+ önerilir) |
| GPU | Intel Arc iGPU (Arc Graphics / A-serisi) |
| RAM | 16 GB+ |
| Disk | 20 GB+ (modeller için) |
| Yazılım | Anaconda, Intel oneAPI Base Toolkit 2025+ |

---

## 🗂️ Proje Yapısı

```
openvino_llm_studio/
│
├── ui/
│   └── app.py                    # Gradio arayüzü
│
├── core/
│   └── orchestrator.py           # Tüm backend'leri koordine eden merkez
│
├── modules/
│   ├── database.py               # SQLite loglama (SQLAlchemy)
│   ├── model_manager.py          # OpenVINO model tarama + yükleme
│   ├── search_engine.py          # DuckDuckGo + BM25 + semantik sıralama
│   ├── dspy_enricher.py          # DSPy prompt zenginleştirme
│   ├── ipex_backend.py           # OllamaBackend + LlamaCppBackend + LlamaServerBackend
│   ├── ipex_worker_client.py     # Backend sarmalayıcı (orchestrator ↔ backend köprüsü)
│   └── hf_catalog.py             # HuggingFace model kataloğu
│
├── logs/                         # SQLite DB (otomatik oluşur)
│
├── requirements.txt
├── setup_and_run.bat             # İlk kurulum
└── run.bat                       # Günlük çalıştırma
```

---

## 🚀 Kurulum

### 1. Intel oneAPI Base Toolkit
[https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)

Kurulum yolu: `C:\Program Files (x86)\Intel\oneAPI\`

### 2. llama-server (SYCL/Arc GPU)
En güncel SYCL destekli Windows binary'sini indirin:
```
https://github.com/ggml-org/llama.cpp/releases
```
`llama-bXXXX-bin-win-sycl-x64.zip` dosyasını indirip şuraya açın:
```
C:\OpenVINO_LLM\llama-server\
```

> **Not:** b8281 ve üzeri Qwen3, Qwen3.5, Gemma3 gibi yeni mimarileri destekler.

### 3. Conda Ortamı
```bat
setup_and_run.bat
```
Bu script:
1. `openvino_studio` conda ortamı oluşturur (Python 3.11)
2. Tüm bağımlılıkları kurar
3. Uygulamayı başlatır

### 4. Günlük Kullanım
```bat
run.bat
```

---

## 📁 Model Dizinleri

### GGUF Modeller (llama-server / Arc GPU)
```
C:\OpenVINO_LLM\gguf\
├── Qwen3-1.7B.Q4_K_M.gguf
├── Qwen3.5-9B-Q4_K_M.gguf
├── Phi-3.5-mini-instruct.Q4_K_M.gguf
└── ...
```

GGUF modelleri HuggingFace'den indirebilirsiniz:
- [Unsloth](https://huggingface.co/unsloth) — optimize edilmiş GGUF koleksiyonu
- [bartowski](https://huggingface.co/bartowski) — geniş model yelpazesi

### OpenVINO Modeller
```
C:\OpenVINO_LLM\
├── Qwen2.5-7B-int4-ov\
│   ├── openvino_model.xml
│   ├── openvino_model.bin
│   └── config.json
└── ...
```

HuggingFace modelini OpenVINO formatına çevirme:
```bash
optimum-cli export openvino \
  --model Qwen/Qwen2.5-7B-Instruct \
  --weight-format int4 \
  C:\OpenVINO_LLM\Qwen2.5-7B-int4-ov
```

---

## 🏗️ Mimari

### Backend'ler

| Backend | Motor | GPU Desteği | Model Formatı |
|---|---|---|---|
| **llama-server** | llama.cpp (C++) | ✅ Arc GPU (SYCL) | GGUF |
| **OpenVINO** | Intel OpenVINO | ✅ Arc GPU (AUTO) | OpenVINO XML/BIN |
| **Ollama** | Ollama / IPEX fork | ✅ Arc GPU (iGPU) | GGUF (Ollama) |

### llama-server Akışı
```
UI → Yükle butonu
    → IPEXWorkerClient.load()
    → LlamaServerBackend.start_server()
        → subprocess.Popen (CREATE_NEW_CONSOLE)
            → llama-server.exe -m model.gguf --device SYCL0
        → port 8080 açılana kadar bekle (max 180s)
    → HTTP bağlantısı kuruldu
    
UI → Mesaj gönder
    → LlamaServerBackend.generate()
    → POST http://127.0.0.1:8080/v1/chat/completions
    → Yanıt UI'a döner
```

### Pipeline Akışı
```
Kullanıcı Promptu
       │
       ▼
[1] Web Araması (DuckDuckGo)
    └─ BM25 + Semantik hibrit sıralama
       │
       ▼
[2] DSPy Zenginleştirme
    └─ Otomatik mod seçimi (ChainOfThought / ReAct / Predict / ...)
       │
       ▼
[3] LLM İnferens (seçili backend)
       │
       ▼
[4] SQLite Loglama
```

---

## ⚡ Performans (Intel Arc iGPU)

| Model | Boyut | Backend | Hız |
|---|---|---|---|
| Qwen3-1.7B Q4_K_M | 1.2 GB | llama-server SYCL | ~27 tok/s |
| Qwen3.5-9B Q4_K_M | 5.2 GB | llama-server SYCL | ~10 tok/s |
| Qwen3.5-27B Q4_K_S | 14.7 GB | llama-server SYCL | ~3 tok/s |

> Değerler Intel Arc Graphics (36 GB paylaşımlı bellek) üzerinde ölçülmüştür.

---

## 🧠 DSPy Modları

| Mod | Tetikleyici | Kullanım |
|---|---|---|
| `ChainOfThought` | "neden", "nasıl", "açıkla" | Karmaşık akıl yürütme |
| `ReAct` | "araştır", "güncel", "haberler" | Web araması ağırlıklı |
| `ProgramOfThought` | "hesapla", "kod", "algoritma" | Matematik / kod |
| `MultiChainComparison` | "vs", "karşılaştır" | Karşılaştırma |
| `Summarize` | "özetle", "kısaca" | Özetleme |
| `Predict` | Default | Basit olgusal sorular |

---

## ⚙️ LLM Parametreleri

| Parametre | Açıklama | Varsayılan |
|---|---|---|
| Temperature | 0=deterministik, 1=yaratıcı | 0.7 |
| Max Tokens | Üretilecek maksimum token | 512 |
| Top-P | Nucleus sampling | 0.9 |
| Top-K | Top-K sampling | 50 |
| Repetition Penalty | Tekrar cezası | 1.1 |

---

## 🐛 Sorun Giderme

**llama-server başlamıyor**
```
llama-server.exe --list-devices
```
`SYCL0: Intel Arc Graphics` görünüyor mu kontrol edin. oneAPI kurulu ve `setvars.bat` çalışmış olmalı.

**`unknown model architecture` hatası**
llama-server binary'niz çok eski. b8281+ indirin:
```
https://github.com/ggml-org/llama.cpp/releases
```

**`WinError 87` / subprocess hatası**
`CREATE_NEW_CONSOLE` flag sorunu. `run.bat`'ı Intel oneAPI Command Prompt'tan değil, normal CMD'den çalıştırın.

**Model bulunamıyor (OpenVINO)**
`C:\OpenVINO_LLM\` dizininde `.xml` + `.bin` dosyası içeren alt dizin olmalı.

**DuckDuckGo rate limit**
Birkaç saniye bekleyin veya `pip install ddgs` ile güncelleyin.

---

## 📊 Loglama

Tüm işlemler `logs/studio.db` SQLite veritabanına kaydedilir:

| Tablo | İçerik |
|---|---|
| `sessions` | Oturum bilgileri |
| `search_logs` | Arama sorguları ve sonuçlar |
| `dspy_logs` | DSPy mod seçimi ve zenginleştirme |
| `llm_logs` | Prompt, yanıt, token sayısı, süre |
| `error_logs` | Hatalar ve stack trace |
| `general_logs` | Uygulama olayları |

---

## 📦 Bağımlılıklar

```
gradio              # Web arayüzü
openvino            # Intel OpenVINO runtime
optimum[openvino]   # HuggingFace → OpenVINO dönüşümü
dspy-ai             # Prompt optimizasyonu
duckduckgo-search   # Web araması
sentence-transformers # Semantik sıralama
rank-bm25           # BM25 sıralama
sqlalchemy          # SQLite ORM
requests            # HTTP (llama-server API)
psutil              # Sistem belleği izleme
```

---

## 🤝 Katkı

Pull request ve issue'lar memnuniyetle karşılanır.

---

## 📄 Lisans

MIT License
