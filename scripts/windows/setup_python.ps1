# Yönetici hakları kontrolü
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "Bu script yönetici hakları gerektiriyor. Yönetici olarak yeniden başlatılıyor..." -ForegroundColor Yellow
    Start-Process powershell -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    exit
}

# UTF-8 karakter kodlamasını ayarla
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

function Test-PythonInstallation {
    try {
        $pythonVersion = & "C:\Python39\python.exe" --version 2>&1
        if ($pythonVersion -match "Python 3.9") {
            return $true
        }
    } catch {
        return $false
    }
    return $false
}

function Install-Python {
    Write-Host "Python 3.9.16 yükleniyor..." -ForegroundColor Yellow
    
    # Kurulum dosyasını indir
    $url = "https://www.python.org/ftp/python/3.9.16/python-3.9.16-amd64.exe"
    $installer = "$env:TEMP\python-3.9.16-amd64.exe"
    
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $url -OutFile $installer
        
        # Kurulumu başlat
        $arguments = "/quiet InstallAllUsers=1 PrependPath=1 Include_test=0 TargetDir=C:\Python39"
        $process = Start-Process -FilePath $installer -ArgumentList $arguments -Wait -PassThru
        
        if ($process.ExitCode -ne 0) {
            throw "Python kurulumu başarısız oldu (Exit code: $($process.ExitCode))"
        }
        
        # Kurulum dosyasını temizle
        Remove-Item $installer -Force -ErrorAction SilentlyContinue
        
        # PATH'i güncelle
        $env:Path = "C:\Python39;C:\Python39\Scripts;$env:Path"
        [System.Environment]::SetEnvironmentVariable(
            "Path",
            [System.Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableTarget]::Machine) + ";C:\Python39;C:\Python39\Scripts",
            [System.EnvironmentVariableTarget]::Machine
        )
        
        Write-Host "Python kurulumu tamamlandı!" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "Python kurulumu başarısız oldu: $_" -ForegroundColor Red
        return $false
    }
}

function Setup-VirtualEnvironment {
    Write-Host "Sanal ortam hazırlanıyor..." -ForegroundColor Yellow
    
    try {
        # pip'i güncelle
        & "C:\Python39\python.exe" -m pip install --upgrade pip
        
        # virtualenv'i yükle
        & "C:\Python39\python.exe" -m pip install virtualenv
        
        # Eğer varsa eski sanal ortamı kaldır
        if (Test-Path ".venv") {
            Remove-Item -Recurse -Force ".venv"
        }
        
        # Yeni sanal ortam oluştur
        & "C:\Python39\python.exe" -m venv .venv
        
        # Sanal ortamı etkinleştir
        $activateScript = ".\.venv\Scripts\Activate.ps1"
        if (Test-Path $activateScript) {
            . $activateScript
        } else {
            throw "Sanal ortam etkinleştirme scripti bulunamadı"
        }
        
        # Gerekli paketleri yükle
        pip install --upgrade pip
        pip install -r requirements.txt
        
        return $true
    }
    catch {
        Write-Host "Sanal ortam kurulumu başarısız oldu: $_" -ForegroundColor Red
        return $false
    }
}

# Ana işlem
try {
    Write-Host "Ötüken3D Python ortamı kuruluyor..." -ForegroundColor Cyan
    
    # Python kurulumunu kontrol et
    if (-not (Test-PythonInstallation)) {
        if (-not (Install-Python)) {
            throw "Python kurulumu başarısız oldu"
        }
    }
    
    # Python sürümünü kontrol et
    $version = & "C:\Python39\python.exe" -c "import sys; print(sys.version.split()[0])"
    Write-Host "Python sürümü: $version" -ForegroundColor Green
    
    # Sanal ortamı kur
    if (-not (Setup-VirtualEnvironment)) {
        throw "Sanal ortam kurulumu başarısız oldu"
    }
    
    Write-Host "`nKurulum başarıyla tamamlandı!" -ForegroundColor Green
    Write-Host "Sanal ortam: $env:VIRTUAL_ENV" -ForegroundColor Green
}
catch {
    Write-Host "`nHATA: $_" -ForegroundColor Red
    Write-Host "Kurulum başarısız oldu. Lütfen manuel olarak Python 3.9.16'yı yükleyin:" -ForegroundColor Red
    Write-Host "https://www.python.org/downloads/release/python-3916/" -ForegroundColor Yellow
}
finally {
    Write-Host "`nDevam etmek için bir tuşa basın..." -ForegroundColor Cyan
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}