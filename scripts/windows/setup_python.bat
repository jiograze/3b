@echo off
chcp 65001 > nul
setlocal EnableDelayedExpansion

echo Ötüken3D için Python kurulum kontrolü...

:: Python yüklü mü kontrol et
set "PYTHON_INSTALLED=0"
set "PYTHON_PATHS=C:\Python39\python.exe C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python39\python.exe C:\Program Files\Python39\python.exe C:\Program Files (x86)\Python39\python.exe"

for %%p in (%PYTHON_PATHS%) do (
    if exist "%%p" (
        set "PYTHON_PATH=%%p"
        set "PYTHON_INSTALLED=1"
        set "PATH=%%~dp0;%%~dp0Scripts;%PATH%"
        goto :found
    )
)

:found
if "%PYTHON_INSTALLED%"=="0" (
    echo Python bulunamadı! Python 3.9.16 yükleniyor...
    
    :: Python kurulum dosyasını indir
    curl -o "%TEMP%\python-3.9.16-amd64.exe" https://www.python.org/ftp/python/3.9.16/python-3.9.16-amd64.exe
    
    :: Sessiz kurulum yap ve PATH'e ekle
    "%TEMP%\python-3.9.16-amd64.exe" /quiet InstallAllUsers=1 PrependPath=1 Include_test=0 TargetDir=C:\Python39
    
    :: PATH'i güncelle
    set "PATH=C:\Python39;C:\Python39\Scripts;%PATH%"
    
    :: Kurulum dosyasını sil
    del "%TEMP%\python-3.9.16-amd64.exe"
    
    echo Python kurulumu tamamlandı!
    set "PYTHON_PATH=C:\Python39\python.exe"
)

:: Python sürümünü kontrol et
"%PYTHON_PATH%" --version
if errorlevel 1 (
    echo Python sürümü kontrol edilemedi!
    pause
    exit /b 1
)

:: pip'i güncelle
echo pip güncelleniyor...
"%PYTHON_PATH%" -m pip install --upgrade pip

:: Sanal ortam oluştur
echo Sanal ortam oluşturuluyor...
"%PYTHON_PATH%" -m pip install virtualenv
"%PYTHON_PATH%" -m venv .venv

:: Sanal ortamı etkinleştir
echo Sanal ortam etkinleştiriliyor...
call .venv\Scripts\activate.bat

:: Temel paketleri yükle
echo Temel paketler yükleniyor...
pip install -r requirements.txt

echo Python ortamı hazır!
echo Sanal ortam aktif: %VIRTUAL_ENV%
pause 