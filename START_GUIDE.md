# 🚀 OpenVINO LLM Studio - Başlatma Kılavuzu

## 📋 Tek Komutla Başlatma

### ✨ Yeni: `start.bat` Kullanımı

Artık tüm sistemi tek bir komutla başlatabilirsiniz!

```bat
:: Interaktif mod (UI seçimi sorar)
start.bat

:: Direkt Modern UI başlat
start.bat modern

:: Direkt Klasik UI başlat
start.bat classic

:: Sadece Ollama başlat (UI yok)
start.bat ollama

:: Kısayol: 1=Modern, 2=Klasik, 3=Ollama
start.bat 1
start.bat 2
start.bat 3
```

---

## 🔄 Eski Yöntemler (Hala Çalışır)

### Yöntem 1: Ayrı Ayrı Başlatma

**1. PowerShell'de Ollama Vulkan:**
```bat
start_ollama_vulkan.bat
```

**2. Anaconda Prompt'ta OpenVINO Studio:**
```bat
conda activate openvino_studio
run.bat
```

### Yöntem 2: Kurulum + Başlatma
```bat
setup_and_run.bat
```

---

## 📊 `start.bat` Akış Diyagramı

```
┌─────────────────────────────────────────────────────────┐
│  start.bat Başlatılıyor...                              │
├─────────────────────────────────────────────────────────┤
│  [1/4] Ollama Vulkan Başlatılıyor...                    │
│         ├─ Ollama zaten çalışıyor mu? → Kontrol         │
│         ├─ OLLAMA_VULKAN=1 (Arc GPU hizlandırma)        │
│         ├─ OLLAMA_KEEP_ALIVE=-1 (Modeller RAM'de)       │
│         ├─ OLLAMA_NUM_PARALLEL=2 (2 model aynı anda)    │
│         └─ Arka planda 'ollama serve'                   │
├─────────────────────────────────────────────────────────┤
│  [2/4] Intel oneAPI Ortamı                              │
│         └─ setvars.bat → SYCL/Arc GPU desteği           │
├─────────────────────────────────────────────────────────┤
│  [3/4] Conda Ortamı                                     │
│         └─ 'openvino_studio' aktif                      │
├─────────────────────────────────────────────────────────┤
│  [4/4] UI Seçimi                                        │
│         ├─ [1] Modern UI (7861) ← ÖNERİLEN              │
│         ├─ [2] Klasik UI (7860)                         │
│         └─ [3] Sadece Ollama                            │
└─────────────────────────────────────────────────────────┘
```

---

## 🎯 Kullanım Senaryoları

### Senaryo 1: Hızlı Başlangıç (Önerilen)
```bat
start.bat modern
```
→ Ollama + Modern UI otomatik başlar

### Senaryo 2: Sadece Ollama Backend
```bat
start.bat ollama
```
→ Ollama arka planda çalışır, başka uygulamalar kullanabilir

### Senaryo 3: Geliştirme Modu
```bat
:: Terminal 1: Ollama
start.bat ollama

:: Terminal 2: UI
start.bat classic
```

---

## 🔧 Ortam Değişkenleri

`start.bat` otomatik olarak şu değişkenleri ayarlar:

| Değişken | Değer | Açıklama |
|----------|-------|----------|
| `OLLAMA_VULKAN` | 1 | Intel Arc GPU hardware hızlandırma |
| `OLLAMA_KEEP_ALIVE` | -1 | Modelleri RAM'de sürekli tut |
| `OLLAMA_NUM_PARALLEL` | 2 | Aynı anda 2 model çalıştır |
| `OLLAMA_HOST` | 127.0.0.1:11434 | Ollama dinleme adresi |

---

## 🛠️ Sorun Giderme

### Ollama Başlamıyor
```bat
:: Ollama servisini kontrol et
ollama --version

:: Port kullanımda mı?
netstat -ano | findstr :11434

:: Ollama'yı tamamen durdur ve yeniden başlat
taskkill /F /IM ollama.exe
start.bat ollama
```

### UI Başlamıyor
```bat
:: Conda ortamını kontrol et
conda env list

:: Port kullanımda mı?
netstat -ano | findstr :7861

:: Logları kontrol et
dir logs\
```

### Intel oneAPI Bulunamadı
```
[UYARI] Intel oneAPI bulunamadı.

Çözüm: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html
```

---

## 📁 Dosya Özeti

| Dosya | Açıklama |
|-------|----------|
| `start.bat` | ✨ Tek komutla başlatma (ÖNERİLEN) |
| `start_ollama_vulkan.bat` | Ollama Vulkan başlatma (eski yöntem) |
| `run.bat` | UI başlatma (Conda ortamı gerekli) |
| `setup_and_run.bat` | Kurulum + başlatma (ilk kullanım) |

---

## 💡 İpuçları

### 1. Ollama'yı Sistem Tepsisinden Durdur
```
Sistem tepsisi → Ollama simgesi → Sağ tık → Quit
```

### 2. Ollama'yı Komutla Durdur
```bat
taskkill /F /IM ollama.exe
```

### 3. Port Değiştirme
```bat
:: Modern UI portunu değiştir
set GRADIO_SERVER_PORT=7862
start.bat modern

:: Ollama portunu değiştir
set OLLAMA_HOST=127.0.0.1:11435
start.bat ollama
```

### 4. Otomatik UI Seçimi (Batch Script İçinde)
```bat
@echo off
:: Direkt Modern UI ile başlat
call start.bat modern
```

---

## 📊 Performans İpuçları

### Ollama Keep-Alive
```bat
:: Modelleri RAM'de tut (hızlı yanıt)
set OLLAMA_KEEP_ALIVE=-1

:: Modelleri hemen boşalt (RAM tasarrufu)
set OLLAMA_KEEP_ALIVE=5m
```

### Parallel Model Loading
```bat
:: 2 model aynı anda (Continue.dev için)
set OLLAMA_NUM_PARALLEL=2

:: Tek model (düşük RAM)
set OLLAMA_NUM_PARALLEL=1
```

---

## 🎉 Başarılı Başlatma Örneği

```
C:\OpenVINO_LLM\openvino_llm_studio\openvino_llm_studio>start.bat modern

╔═══════════════════════════════════════════════════════════════╗
║     OpenVINO LLM Studio - Başlatılıyor...                     ║
╚═══════════════════════════════════════════════════════════════╝

[1/4] Ollama Vulkan başlatılıyor...
[OK] Ollama zaten çalışıyor.

[2/4] Intel oneAPI ortamı hazırlanıyor...
[OK] Intel oneAPI başlatıldı.

[3/4] Conda ortamı aktif ediliyor...
[OK] Conda 'openvino_studio' ortamı aktif.

[4/4] Uygulama başlatılıyor...

═══════════════════════════════════════════════════════════════

    Modern UI Başlatılıyor...
═══════════════════════════════════════════════════════════════

Tarayıcı: http://127.0.0.1:7861

Özellikler:
  ✓ 3 kullanıcı modu (Basit/Orta/Expert)
  ✓ 3 adımlı workflow
  ✓ Preset profiller (Hızlı/Dengeli/Kaliteli)
  ✓ Real-time CPU/RAM monitoring
  ✓ Contextual help tooltips

Başlatılıyor... Ctrl+C ile durdurabilirsiniz.

* Running on local URL:  http://127.0.0.1:7861
```

---

## 📝 Notlar

- `start.bat` **yönetici yetkisi gerektirmez**
- Ollama arka planda çalışmaya devam eder (UI kapansa bile)
- İlk başlatmada model yükleme 10-30 saniye sürebilir
- Intel Arc GPU için `OLLAMA_VULKAN=1` şart

---

**Son Güncelleme:** 2026-04-27  
**Versiyon:** 2.0 (Tek Komut Başlatma)
