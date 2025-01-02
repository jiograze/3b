# Linux Scripts

Bu klasör, Ötüken3D projesinin Linux işletim sistemi için özel scriptlerini içerir.

## Shell Scripts (.sh)

- `create_structure.sh`: Proje dizin yapısını oluşturur
- `init_project.sh`: Projeyi başlatır ve gerekli kurulumları yapar
- `setup.sh`: İlk kurulum işlemlerini gerçekleştirir

## Kullanım

1. Scriptleri çalıştırmadan önce çalıştırma izni verilmelidir:
   ```bash
   chmod +x scripts/linux/*.sh
   ```

2. İlk kurulum için `setup.sh` kullanılmalıdır:
   ```bash
   ./scripts/linux/setup.sh
   ```

3. Proje başlatma için `init_project.sh` kullanılmalıdır:
   ```bash
   ./scripts/linux/init_project.sh
   ```

## Not

- Scriptler Bash shell için yazılmıştır
- Ubuntu, Debian ve türevleri için test edilmiştir
- Root hakları gerektirebilir (sudo) 