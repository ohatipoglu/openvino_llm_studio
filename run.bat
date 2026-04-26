@echo off
setlocal enabledelayedexpansion

:: Intel oneAPI ortamini ayarla
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

:: Conda ortamini aktif et
call conda activate openvino_studio

:: Logs klasorunu olustur
if not exist "logs" mkdir logs

:: Baslik
echo ================================================
echo    OpenVINO LLM Studio - Baslatma Ekrani
echo ================================================
echo.

:: Kullaniciya UI secimi sor
echo Hangi arayuzu kullanmak istersiniz?
echo.
echo   [1] Modern UI  (YENI - onerilen)
echo       - 3 adimli workflow (Model -> Ayarlar -> Chat)
echo       - Kullanici modlari (Basit/Orta/Expert)
echo       - Real-time dashboard
echo       - Preset profiller
echo.
echo   [2] Klasik UI (Eski - stabil)
echo       - Tek ekran arayuz
echo       - Tum ayarlar gorunur
echo.

set /p ui_choice="Seciminiz (1 veya 2, Varsayilan: 1): "

:: Varsayilan deger 1
if "!ui_choice!"=="" set ui_choice=1

echo.
echo ================================================

if "!ui_choice!"=="1" goto :MODERN_UI
if "!ui_choice!"=="2" goto :CLASSIC_UI

:: Gecersiz secim
echo Gecersiz secim! Varsayilan olarak Modern UI baslatiliyor...
goto :MODERN_UI

:MODERN_UI
echo    Modern UI Baslatiliyor...
echo ================================================
echo.
echo Tarayici adresi: http://127.0.0.1:7861
echo (Port degistirmek icin: set GRADIO_SERVER_PORT=786x)
echo.
echo Modern UI ozellikleri:
echo   - 3 kullanici modu (Basit/Orta/Expert)
echo   - 3 adimli workflow
echo   - Preset profiller (Hizli/Dengeli/Kaliteli)
echo   - Real-time CPU/RAM dashboard
echo   - Contextual help tooltips
echo.
echo Baslatiliyor...
python ui\app_modern.py
goto :END

:CLASSIC_UI
echo    Klasik UI Baslatiliyor...
echo ================================================
echo.
echo Tarayici adresi: http://127.0.0.1:7860
echo.
echo Klasik UI ozellikleri:
echo   - Tek ekran arayuz
echo   - Tum ayarlar gorunur
echo   - Stabil ve alisilmis
echo.
echo Baslatiliyor...
python ui\app.py
goto :END

:END
echo.
echo ================================================
echo Uygulama kapatildi.
echo ================================================
pause
