Write-Host "Ötüken3D başlatılıyor..."

# Hugging Face uyarısını devre dışı bırak
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

# Streamlit uygulamasını başlat
streamlit run app.py 