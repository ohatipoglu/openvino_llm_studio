# ⚡ OpenVINO LLM Studio

> **TR / EN README**: First Turkish, then English.

Intel Arc iGPU üzerinde yerel LLM çalıştırma ortamı.  
**llama-server (SYCL/Arc GPU)** + **OpenVINO** + **Ollama** backend’lerini tek arayüzde sunar.  
DuckDuckGo web araması, DSPy prompt zenginleştirme ve SQLite loglama dahildir.

---

## 🇹🇷 Türkçe

### Öne çıkanlar
- **Tek UI, 3 backend**: OpenVINO / Ollama / llama-server (llama.cpp)
- **Web araması**: DuckDuckGo + BM25 + semantik sıralama
- **DSPy zenginleştirme**: otomatik mod seçimi + template
- **Loglama**: `logs/studio.db` (SQLite)

### Hızlı başlangıç
1) Intel oneAPI Base Toolkit kur (Windows + Arc/SYCL için gerekli).  
2) SYCL destekli `llama-server` binary’sini indirip `C:\OpenVINO_LLM\llama-server\` altına çıkar.  
3) Ortamı kur ve başlat:

```bat
setup_and_run.bat
```

Sonraki çalıştırmalar:

```bat
run.bat
```

UI varsayılan olarak `http://127.0.0.1:7860` ile açılır.  
Port doluysa uygulama **7860–7870** aralığında boş bir port seçer.

### Port ayarı (Gradio)
Port’u sabitlemek istersen:

```bat
set GRADIO_SERVER_PORT=7861
run.bat
```

Host değiştirmek istersen:

```bat
set GRADIO_SERVER_NAME=0.0.0.0
set GRADIO_SERVER_PORT=7861
run.bat
```

### Sistem gereksinimleri
| Gereksinim | Detay |
|---|---|
| İşletim Sistemi | Windows 11 |
| CPU | Intel (12. nesil+ önerilir) |
| GPU | Intel Arc iGPU (Arc Graphics / A-serisi) |
| RAM | 16 GB+ |
| Disk | 20 GB+ (modeller için) |
| Yazılım | Anaconda, Intel oneAPI Base Toolkit 2025+ |

### Proje yapısı
```
openvino_llm_studio/
├── ui/
│   └── app.py                    # Gradio arayüzü
├── core/
│   └── orchestrator.py           # Backend koordinasyonu
├── modules/
│   ├── database.py               # SQLite loglama (SQLAlchemy)
│   ├── model_manager.py          # OpenVINO model tarama + yükleme
│   ├── search_engine.py          # DuckDuckGo + BM25 + semantik sıralama
│   ├── dspy_enricher.py          # DSPy prompt zenginleştirme
│   ├── ipex_backend.py           # OllamaBackend + LlamaCppBackend + LlamaServerBackend
│   ├── ipex_worker_client.py     # Orchestrator ↔ backend köprüsü
│   └── hf_catalog.py             # HuggingFace model kataloğu
├── logs/                         # SQLite DB / log (otomatik oluşur)
├── requirements.txt
├── setup_and_run.bat             # İlk kurulum
└── run.bat                       # Günlük çalıştırma
```

### Mimari Akışı
```
[Kullanıcı Girdisi (UI)]
         |
         v
[Orchestrator (core/orchestrator.py)]
         |
         +-----> [DSPyEnricher (modules/dspy_enricher.py)] --(LLM ile)--> [Arama Sorgusu Üretimi]
         |                                                                       |
         |                                                                       v
         +-----------------------------------------> [WebSearcher (modules/search_engine.py)]
         |                                                                       | (DuckDuckGo + Sıralama)
         |                                                                       v
         |<-------------------------------------------------------------- [Arama Sonuçları]
         |
         +-----> [DSPyEnricher (modules/dspy_enricher.py)] --(LLM ile)--> [Mod Seçimi & Prompt Şablonu]
         |                                                                       |
         |                                                                       v
         |<-------------------------------------------------------------- [Zenginleştirilmiş Prompt]
         |
         v
[Aktif Backend'e Yönlendirme]
         |
         +-----> [OpenVINO (modules/model_manager.py)]
         |
         +-----> [llama-server (modules/ipex_worker_client.py)]
         |
         +-----> [Ollama (modules/ipex_backend.py)]
         |
         v
[LLM Yanıtı] -> [UI]

(Tüm adımlar `database.py` ile loglanır)
```

### Model dizinleri
**GGUF (llama-server / Arc GPU)**:
```
C:\OpenVINO_LLM\gguf\
```

**OpenVINO**:
```
C:\OpenVINO_LLM\
```

HuggingFace modelini OpenVINO formatına çevirme örneği:

```bash
optimum-cli export openvino ^
  --model Qwen/Qwen2.5-7B-Instruct ^
  --weight-format int4 ^
  C:\OpenVINO_LLM\Qwen2.5-7B-int4-ov
```

### Sorun giderme (kısa)
- **llama-server başlamıyor**:

```bat
llama-server.exe --list-devices
```

`SYCL0: Intel Arc Graphics` görünmüyorsa oneAPI / driver / SYCL kurulumu eksik olabilir.

- **`unknown model architecture`**: `llama-server` çok eski olabilir. Güncel SYCL binary indir: `https://github.com/ggml-org/llama.cpp/releases`
- **DuckDuckGo rate limit**: Biraz bekle veya bağımlılıkları güncelle (`ddgs`).

---

## 🇺🇸 English

### Highlights
- **One UI, 3 backends**: OpenVINO / Ollama / llama-server (llama.cpp)
- **Web search**: DuckDuckGo + BM25 + semantic reranking
- **DSPy enrichment**: automatic mode selection + templates
- **Logging**: `logs/studio.db` (SQLite)

### Quickstart
1) Install Intel oneAPI Base Toolkit (recommended for Arc/SYCL on Windows).  
2) Download a SYCL-enabled `llama-server` release and extract to `C:\OpenVINO_LLM\llama-server\`.  
3) Setup and run:

```bat
setup_and_run.bat
```

Daily run:

```bat
run.bat
```

The UI usually opens at `http://127.0.0.1:7860`.  
If the port is already taken, the app will **auto-pick a free port in 7860–7870**.

### Gradio server configuration
Force a specific port:

```bat
set GRADIO_SERVER_PORT=7861
run.bat
```

Bind to a different host:

```bat
set GRADIO_SERVER_NAME=0.0.0.0
set GRADIO_SERVER_PORT=7861
run.bat
```

### System requirements
| Requirement | Details |
|---|---|
| OS | Windows 11 |
| CPU | Intel (12th gen+ recommended) |
| GPU | Intel Arc iGPU (Arc Graphics / A-series) |
| RAM | 16 GB+ |
| Disk | 20 GB+ (models) |
| Software | Anaconda, Intel oneAPI Base Toolkit 2025+ |

### Architectural Flow
```
[User Input (UI)]
      |
      v
[Orchestrator (core/orchestrator.py)]
      |
      +-----> [DSPyEnricher (modules/dspy_enricher.py)] --(via LLM)--> [Search Query Generation]
      |                                                                    |
      |                                                                    v
      +-----------------------------------------> [WebSearcher (modules/search_engine.py)]
      |                                                                    | (DuckDuckGo + Reranking)
      |                                                                    v
      |<-------------------------------------------------------------- [Search Results]
      |
      +-----> [DSPyEnricher (modules/dspy_enricher.py)] --(via LLM)--> [Mode Selection & Prompt Template]
      |                                                                    |
      |                                                                    v
      |<-------------------------------------------------------------- [Enriched Prompt]
      |
      v
[Route to Active Backend]
      |
      +-----> [OpenVINO (modules/model_manager.py)]
      |
      +-----> [llama-server (modules/ipex_worker_client.py)]
      |
      +-----> [Ollama (modules/ipex_backend.py)]
      |
      v
[LLM Response] -> [UI]

(All steps are logged via `database.py`)
```

### License
MIT License
