@echo off
setlocal enabledelayedexpansion

:: Log dizini oluştur
mkdir logs 2>nul

:: Log dosyası başlat
echo %date% %time% - Kurulum başladı > logs\setup.log

:: Disk alanı kontrolü
for /f "tokens=3" %%a in ('dir /-c 2^>nul') do set "FREE_SPACE=%%a"
set /a "FREE_SPACE_GB=%FREE_SPACE:~0,-9%"
if %FREE_SPACE_GB% LSS 50 (
    echo UYARI: Önerilen minimum disk alanı 50GB, mevcut alan: %FREE_SPACE_GB%GB
    echo %date% %time% - Yetersiz disk alanı uyarısı >> logs\setup.log
    choice /M "Devam etmek istiyor musunuz"
    if errorlevel 2 goto :eof
)

:: Python versiyonu kontrolü
python -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)"
if errorlevel 1 (
    echo HATA: Python 3.8 veya üstü gerekli!
    echo %date% %time% - Python versiyon kontrolü başarısız >> logs\setup.log
    goto :error_exit
)

:: Paket versiyonlarını kontrol et
python scripts\check_dependencies.py
if errorlevel 1 (
    echo HATA: Paket uyumluluk kontrolü başarısız!
    goto :error_exit
)

:: Ana dizinleri oluştur
echo Dizin yapısı oluşturuluyor...
python scripts\create_structure.py
if errorlevel 1 (
    echo HATA: Dizin yapısı oluşturulamadı!
    goto :error_exit
)

:: Python paketlerini yükle
echo Python paketleri yükleniyor...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo HATA: Paket yüklemesi başarısız!
    goto :error_exit
)

:: Model indirme ve doğrulama
echo Modeller indiriliyor ve doğrulanıyor...
python scripts\download_models.py
if errorlevel 1 (
    echo UYARI: Bazı modeller indirilemedi veya doğrulanamadı
    echo %date% %time% - Model indirme/doğrulama hatası >> logs\setup.log
)

:: Veri seti yönetimi
echo.
echo Veri setleri yönetimi
echo -------------------
python scripts\dataset_manager.py

:: Kurulum sonrası doğrulama
python scripts\verify_installation.py
if errorlevel 1 (
    echo UYARI: Kurulum doğrulama hatası!
    echo %date% %time% - Kurulum doğrulama hatası >> logs\setup.log
)

echo Kurulum tamamlandı! Detaylar için logs\setup.log dosyasını kontrol edin.
pause
exit /b

:error_exit
echo %date% %time% - Kurulum hatası >> logs\setup.log
echo Kurulum başarısız oldu. Detaylar için logs\setup.log dosyasını kontrol edin.
pause
exit /b 1