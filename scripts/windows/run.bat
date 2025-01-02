@echo off
echo Ötüken3D başlatılıyor...

:: Hugging Face uyarısını devre dışı bırak
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

:: Streamlit uygulamasını başlat
streamlit run app.py

pause 