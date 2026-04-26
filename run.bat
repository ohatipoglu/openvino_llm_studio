@echo off
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
rem Sadece calistirma (kurulum yapilmis ortam icin)
call conda activate openvino_studio
if not exist "logs" mkdir logs
echo OpenVINO LLM Studio baslatiliyor...
echo Tarayicida: http://127.0.0.1:7860
python ui/app.py
pause
