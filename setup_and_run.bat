@echo off
:: OpenVINO LLM Studio - Kurulum ve Çalıştırma
:: Windows 11 + Intel Arc iGPU + Anaconda

setlocal EnableDelayedExpansion

set ENV_NAME=openvino_studio
set PYTHON_VERSION=3.11
set LLAMA_SERVER_DIR=C:\OpenVINO_LLM\llama-server
set LLAMA_RELEASE_URL=https://github.com/ggml-org/llama.cpp/releases

echo.
echo ╔══════════════════════════════════════════════╗
echo ║      OpenVINO LLM Studio - Setup ^& Run      ║
echo ╚══════════════════════════════════════════════╝
echo.

:: ─────────────────────────────────────────────────────
:: 1. Intel oneAPI kontrolü (SYCL/Arc GPU için şart)
:: ─────────────────────────────────────────────────────
echo [1/5] Intel oneAPI kontrol ediliyor...
if exist "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" (
    call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force >nul 2>&1
    echo [OK] Intel oneAPI bulundu ve baslatildi.
) else (
    echo [UYARI] Intel oneAPI bulunamadi.
    echo         llama-server SYCL/Arc GPU destegi icin gerekli.
    echo         Indirme: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html
    echo         Devam ediliyor ^(sadece CPU modu kullanilabilir^)...
)
echo.

:: ─────────────────────────────────────────────────────
:: 2. Conda kontrolü
:: ─────────────────────────────────────────────────────
echo [2/5] Conda kontrol ediliyor...
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [HATA] Conda bulunamadi. Anaconda kurunuz:
    echo        https://www.anaconda.com/download
    pause
    exit /b 1
)
echo [OK] Conda bulundu.
echo.

:: ─────────────────────────────────────────────────────
:: 3. Conda ortamı
:: ─────────────────────────────────────────────────────
echo [3/5] Conda ortami kontrol ediliyor...
conda env list | findstr /C:"%ENV_NAME%" >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] '%ENV_NAME%' ortami zaten mevcut.
    goto :install_packages
)

echo [INFO] '%ENV_NAME%' ortami olusturuluyor ^(Python %PYTHON_VERSION%^)...
call conda create -n %ENV_NAME% python=%PYTHON_VERSION% -y
if %errorlevel% neq 0 (
    echo [HATA] Ortam olusturulamadi.
    pause
    exit /b 1
)
echo [OK] Ortam olusturuldu.

:: ─────────────────────────────────────────────────────
:: 4. Paket kurulumu
:: ─────────────────────────────────────────────────────
:install_packages
echo.
echo [4/5] Paketler kuruluyor...
call conda activate %ENV_NAME%

echo [INFO] Ana gereksinimler...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [HATA] requirements.txt kurulamadi.
    pause
    exit /b 1
)

echo [INFO] duckduckgo-search...
pip install duckduckgo-search --quiet

echo [INFO] llama-cpp-python ^(CPU fallback^)...
pip install llama-cpp-python --quiet

echo.
echo [OK] Paket kurulumu tamamlandi.

:: ─────────────────────────────────────────────────────
:: 5. llama-server kontrolü
:: ─────────────────────────────────────────────────────
echo.
echo [5/5] llama-server kontrol ediliyor...
if exist "%LLAMA_SERVER_DIR%\llama-server.exe" (
    echo [OK] llama-server bulundu: %LLAMA_SERVER_DIR%
    :: GPU listesi
    "%LLAMA_SERVER_DIR%\llama-ls-sycl-device.exe" 2>nul || echo [INFO] SYCL cihaz listesi alinamadi.
) else (
    echo [UYARI] llama-server bulunamadi: %LLAMA_SERVER_DIR%
    echo.
    echo         Arc GPU hizlandirmasi icin llama.cpp SYCL binary indirin:
    echo         %LLAMA_RELEASE_URL%
    echo.
    echo         Dosyayi su dizine acin:
    echo         %LLAMA_SERVER_DIR%\
    echo.
    echo         llama-server olmadan sadece OpenVINO ve Ollama backend kullanilabilir.
)

:: ─────────────────────────────────────────────────────
:: Uygulamayı başlat
:: ─────────────────────────────────────────────────────
echo.
echo ════════════════════════════════════════════════
echo  Kurulum tamamlandi! Studio baslatiliyor...
echo  Tarayicida: http://127.0.0.1:7860
echo ════════════════════════════════════════════════
echo.

call conda activate %ENV_NAME%
if not exist "logs" mkdir logs
python ui/app.py

if %errorlevel% neq 0 (
    echo.
    echo [HATA] Uygulama baslatılamadi.
    echo        logs\ klasorunu kontrol edin.
)
pause
endlocal
