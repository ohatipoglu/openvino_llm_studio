@echo off
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
:: Sadece çalıştırma (kurulum yapılmış ortam için)
call conda activate openvino_studio
if not exist "logs" mkdir logs
echo OpenVINO LLM Studio başlatılıyor...
echo Tarayıcıda: http://127.0.0.1:7860
python ui/app.py
pause
