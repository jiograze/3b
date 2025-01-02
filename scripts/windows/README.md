# Windows Scripts

Bu klasör, Ötüken3D projesinin Windows işletim sistemi için özel scriptlerini içerir.

## Batch Scripts (.bat)

- `install_dependencies.bat`: Gerekli bağımlılıkları yükler
- `run.bat`: Uygulamayı başlatır
- `setup.bat`: İlk kurulum işlemlerini gerçekleştirir
- `setup_python.bat`: Python ortamını hazırlar
- `train.bat`: Model eğitimini başlatır

## PowerShell Scripts (.ps1)

- `install_dependencies.ps1`: Bağımlılıkları PowerShell ile yükler
- `run.ps1`: Uygulamayı PowerShell ile başlatır
- `setup.ps1`: Kurulum işlemlerini PowerShell ile gerçekleştirir
- `setup_python.ps1`: Python ortamını PowerShell ile hazırlar
- `train.ps1`: Model eğitimini PowerShell ile başlatır

## Kullanım

1. PowerShell scriptleri için yönetici hakları gerekebilir
2. Batch scriptleri komut isteminden çalıştırılmalıdır
3. İlk kurulum için `setup.bat` veya `setup.ps1` kullanılmalıdır 