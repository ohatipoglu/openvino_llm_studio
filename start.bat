@echo off
setlocal EnableDelayedExpansion

:: ╔═══════════════════════════════════════════════════════════════╗
:: ║     OpenVINO LLM Studio - Tek Komut Başlatma                  ║
:: ║     Ollama Vulkan + OpenVINO Studio                           ║
:: ╚═══════════════════════════════════════════════════════════════╝

:: Kullanım:
::   start.bat              -> UI seçimi sorar
::   start.bat unified      -> Direkt Unified UI başlatır (YENİ - ÖNERİLEN)
::   start.bat modern       -> Direkt Modern UI başlatır
::   start.bat workspace    -> Direkt Workspace UI başlatır
::   start.bat classic      -> Direkt Klasik UI başlatır
::   start.bat ollama       -> Sadece Ollama başlatır

:: ════════════════════════════════════════════════════════════════╗
::     OpenVINO LLM Studio - Başlatılıyor...                       ║
:: ════════════════════════════════════════════════════════════════╝

echo.
echo ╔═══════════════════════════════════════════════════════════════╗
echo ║     OpenVINO LLM Studio - Başlatılıyor...                     ║
echo ╚═══════════════════════════════════════════════════════════════╝
echo.

:: Komut satırı argümanını kontrol et
set UI_MODE=ask
if "%~1"=="" set UI_MODE=ask
if /i "%~1"=="unified" set UI_MODE=unified
if /i "%~1"=="modern" set UI_MODE=modern
if /i "%~1"=="workspace" set UI_MODE=workspace
if /i "%~1"=="classic" set UI_MODE=classic
if /i "%~1"=="ollama" set UI_MODE=ollama
if /i "%~1"=="1" set UI_MODE=unified
if /i "%~1"=="2" set UI_MODE=modern
if /i "%~1"=="3" set UI_MODE=workspace
if /i "%~1"=="4" set UI_MODE=classic
if /i "%~1"=="5" set UI_MODE=ollama

:: ─────────────────────────────────────────────────────────────────
:: 1. OLLAMA VULKAN BAŞLAT
:: ─────────────────────────────────────────────────────────────────
echo [1/4] Ollama Vulkan başlatılıyor...

:: Ollama servis durumu kontrolü
tasklist /FI "IMAGENAME eq ollama.exe" 2>nul | find "ollama.exe" >nul
if %errorlevel% equ 0 (
    echo [OK] Ollama zaten çalışıyor.
) else (
    echo [INFO] Ollama Vulkan başlatılıyor...

    :: Vulkan hardware acceleration (Intel Arc iGPU)
    set OLLAMA_VULKAN=1

    :: Modelleri RAM'de sürekli tut
    set OLLAMA_KEEP_ALIVE=-1

    :: Aynı anda 2 model çalıştırma izni (Continue.dev için)
    set OLLAMA_NUM_PARALLEL=2
    set OLLAMA_HOST=127.0.0.1:11434
    
    :: Model yükleme timeout - 5 dakika (büyük modeller için)
    set OLLAMA_MAX_LOADED_MODELS=1
    set OLLAMA_MAX_QUEUE=512

    :: Ollama'yı başlat (arka planda)
    start "Ollama Vulkan" /B ollama serve

    :: Ollama hazır olana kadar bekle (max 10 saniye)
    echo [INFO] Ollama servisi hazırlanıyor...
    timeout /t 5 /nobreak >nul

    :: Ollama kontrolü
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if %errorlevel% equ 0 (
        echo [OK] Ollama Vulkan başarıyla başlatıldı.
    ) else (
        echo [UYARI] Ollama başlatılamadı. Sadece OpenVINO backend kullanılabilir.
    )
)
echo.

:: ─────────────────────────────────────────────────────────────────
:: 2. INTEL ONEAPI ORTAMI
:: ─────────────────────────────────────────────────────────────────
echo [2/4] Intel oneAPI ortamı hazırlanıyor...

if exist "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" (
    call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force >nul 2>&1
    echo [OK] Intel oneAPI başlatıldı.
) else (
    echo [UYARI] Intel oneAPI bulunamadı. llama-server CPU modunda çalışacak.
)
echo.

:: ─────────────────────────────────────────────────────────────────
:: 3. CONDA ORTAMI
:: ─────────────────────────────────────────────────────────────────
echo [3/4] Conda ortamı aktif ediliyor...

call conda activate openvino_studio
if %errorlevel% equ 0 (
    echo [OK] Conda 'openvino_studio' ortamı aktif.
) else (
    echo [HATA] Conda ortamı bulunamadı!
    echo.
    echo Çözüm: setup_and_run.bat ile kurulum yapın.
    pause
    exit /b 1
)
echo.

:: ─────────────────────────────────────────────────────────────────
:: 4. UYGULAMAYI BAŞLAT
:: ─────────────────────────────────────────────────────────────────
echo [4/4] Uygulama başlatılıyor...
echo.

:: Logs klasörü
if not exist "logs" mkdir logs

:: UI seçimi (eğer komut satırından verilmemişse sor)
if "!UI_MODE!"=="ask" (
    echo Hangi arayüzü kullanmak istersiniz?
    echo.
    echo   [1] Unified UI   (YENİ - ÖNERİLEN)
    echo       - 3 mod seçeneği (Klasik/Modern/Workspace) tek UI'da
    echo       - Sidebar + Ana Kanvas düzeni
    echo       - Minimal header, bilgilendirici footer
    echo       - http://127.0.0.1:7860
    echo.
    echo   [2] Modern UI
    echo       - 3 adımlı workflow (Model → Ayarlar → Chat)
    echo       - Kullanıcı modları (Basit/Orta/Expert)
    echo       - Real-time CPU/RAM dashboard
    echo       - http://127.0.0.1:7861
    echo.
    echo   [3] Workspace UI
    echo       - Sidebar + Ana Kanvas düzeni
    echo       - Sürekli erişilebilir ayarlar
    echo       - Minimal header, bilgilendirici footer
    echo       - http://127.0.0.1:7862
    echo.
    echo   [4] Klasik UI
    echo       - Tek ekran
    echo       - Tüm ayarlar görünür
    echo       - http://127.0.0.1:7863
    echo.
    echo   [5] Sadece Ollama (UI yok)
    echo.

    set /p ui_choice="Seçiminiz (1/2/3/4/5, Varsayılan: 1): "
    if "!ui_choice!"=="" set ui_choice=1

    if "!ui_choice!"=="1" set UI_MODE=unified
    if "!ui_choice!"=="2" set UI_MODE=modern
    if "!ui_choice!"=="3" set UI_MODE=workspace
    if "!ui_choice!"=="4" set UI_MODE=classic
    if "!ui_choice!"=="5" set UI_MODE=ollama
)

echo.
echo ═══════════════════════════════════════════════════════════════
echo.

if "!UI_MODE!"=="unified" goto :UNIFIED_UI
if "!UI_MODE!"=="modern" goto :MODERN_UI
if "!UI_MODE!"=="workspace" goto :WORKSPACE_UI
if "!UI_MODE!"=="classic" goto :CLASSIC_UI
if "!UI_MODE!"=="ollama" goto :OLLAMA_ONLY

:: Geçersiz seçim → Unified UI
goto :UNIFIED_UI

:UNIFIED_UI
echo    Unified UI Başlatılıyor... (YENİ - ÖNERİLEN)
echo ═══════════════════════════════════════════════════════════════
echo.
echo Tarayıcı: http://127.0.0.1:7860
echo.
echo Özellikler:
echo   ✓ 3 mod seçeneği (Klasik/Modern/Workspace) tek UI'da
echo   ✓ Sidebar + Ana Kanvas düzeni
echo   ✓ Sürekli erişilebilir ayarlar
echo   ✓ Minimal header, bilgilendirici footer
echo   ✓ Chat history desteği
echo   ✓ Log temizleme
echo   ✓ Backend detay durum göstergesi
echo.
echo Başlatılıyor... Ctrl+C ile durdurabilirsiniz.
echo.
python ui\app_unified.py
goto :END

:MODERN_UI
echo    Modern UI Başlatılıyor...
echo ═══════════════════════════════════════════════════════════════
echo.
echo Tarayıcı: http://127.0.0.1:7861
echo.
echo Özellikler:
echo   ✓ 3 kullanıcı modu (Basit/Orta/Expert)
echo   ✓ 3 adımlı workflow
echo   ✓ Preset profiller (Hızlı/Dengeli/Kaliteli)
echo   ✓ Real-time CPU/RAM monitoring
echo   ✓ Contextual help tooltips
echo   ✓ llama-cpp start/stop kontrolü
echo.
echo Başlatılıyor... Ctrl+C ile durdurabilirsiniz.
echo.
python ui\app_modern.py
goto :END

:WORKSPACE_UI
echo    Workspace UI Başlatılıyor...
echo ═══════════════════════════════════════════════════════════════
echo.
echo Tarayıcı: http://127.0.0.1:7862
echo.
echo Özellikler:
echo   ✓ Sidebar + Ana Kanvas düzeni
echo   ✓ Sürekli erişilebilir ayarlar
echo   ✓ Minimal header, bilgilendirici footer
echo   ✓ Accordion'lar ile organize edilmiş ayarlar
echo   ✓ LM Studio / Ollama WebUI tarzı
echo   ✓ Chat history desteği
echo   ✓ Log temizleme
echo   ✓ System prompt desteği
echo.
echo Başlatılıyor... Ctrl+C ile durdurabilirsiniz.
echo.
python ui\app_workspace.py
goto :END

:CLASSIC_UI
echo    Klasik UI Başlatılıyor...
echo ═══════════════════════════════════════════════════════════════
echo.
echo Tarayıcı: http://127.0.0.1:7863
echo.
echo Özellikler:
echo   ✓ Tek ekran arayüz
echo   ✓ Tüm ayarlar görünür
echo   ✓ Stabil ve alışılmış
echo   ✓ Tüm backend'ler destekleniyor
echo.
echo Başlatılıyor... Ctrl+C ile durdurabilirsiniz.
echo.
python ui\app.py
goto :END

:OLLAMA_ONLY
echo    Sadece Ollama Modu
echo ═══════════════════════════════════════════════════════════════
echo.
echo Ollama zaten çalışıyor.
echo Model yüklemek için: ollama pull <model-name>
echo.
echo Örnek:
echo   ollama pull llama3.2
echo   ollama pull qwen2.5:7b
echo.
echo Ollama'yı durdurmak için görev yöneticisinden "ollama.exe" yi bitirin.
echo.
pause
goto :END

:END
echo.
echo ═══════════════════════════════════════════════════════════════
echo Uygulama kapatıldı.
echo ═══════════════════════════════════════════════════════════════
echo.
echo Ollama hala arka planda çalışıyor olabilir.
echo.
echo Ollama'yı durdurmak için:
echo   taskkill /F /IM ollama.exe
echo.
echo Veya sistem tepsisindeki Ollama simgesine sağ tıklayıp "Quit" seçin.
echo.
pause
