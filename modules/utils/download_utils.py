import os
import requests
import zipfile
import tarfile
from tqdm import tqdm
import shutil
import sys

# Bağımlılıkları kontrol et ve gerekirse yükle
required_packages = ['gdown', 'requests', 'tqdm']

def check_and_install_dependencies():
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"{package} yükleniyor...")
            os.system(f"{sys.executable} -m pip install {package}")

# Bağımlılıkları kontrol et
check_and_install_dependencies()

# Şimdi gerekli modülleri import et
import gdown

class DatasetDownloader:
    def __init__(self, dataset_name: str, base_path: str = "data"):
        self.dataset_name = dataset_name
        self.base_path = base_path
        self.dataset_path = os.path.join(base_path, dataset_name)
        
        # Dizin yoksa oluştur
        os.makedirs(self.dataset_path, exist_ok=True)
        
    def download_file(self, url: str, filename: str, cookies: dict = None, timeout: int = 30) -> str:
        """Dosyayı indir ve ilerleme çubuğu göster"""
        try:
            local_path = os.path.join(self.dataset_path, filename)
            
            if os.path.exists(local_path):
                print(f"{filename} zaten mevcut, atlanıyor...")
                return local_path
                
            print(f"{filename} indiriliyor...")
            response = requests.get(url, stream=True, cookies=cookies, timeout=timeout)
            response.raise_for_status()  # HTTP hatalarını kontrol et
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(local_path, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
                    
            return local_path
            
        except requests.Timeout:
            print(f"Zaman aşımı: {url} kaynağına bağlanılamadı")
            if os.path.exists(local_path):
                os.remove(local_path)
            raise
        except Exception as e:
            print(f"Hata: {filename} indirilirken bir sorun oluştu: {str(e)}")
            if os.path.exists(local_path):
                os.remove(local_path)
            raise
    
    def download_gdrive(self, file_id: str, filename: str) -> str:
        """Google Drive'dan dosya indir"""
        try:
            local_path = os.path.join(self.dataset_path, filename)
            
            if os.path.exists(local_path):
                print(f"{filename} zaten mevcut, atlanıyor...")
                return local_path
                
            print(f"{filename} Google Drive'dan indiriliyor...")
            gdown.download(id=file_id, output=local_path, quiet=False)
            
            if not os.path.exists(local_path):
                raise Exception("Dosya indirilemedi")
                
            return local_path
            
        except Exception as e:
            print(f"Hata: Google Drive'dan indirme başarısız: {str(e)}")
            if os.path.exists(local_path):
                os.remove(local_path)
            raise
    
    def extract_archive(self, archive_path: str, extract_path: str = None) -> str:
        """Zip veya tar arşivini çıkart"""
        try:
            if extract_path is None:
                extract_path = self.dataset_path
                
            print(f"Arşiv çıkartılıyor: {archive_path}")
            
            if archive_path.endswith('.zip'):
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
            elif archive_path.endswith(('.tar.gz', '.tgz')):
                with tarfile.open(archive_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(extract_path)
            elif archive_path.endswith('.tar'):
                with tarfile.open(archive_path, 'r:') as tar_ref:
                    tar_ref.extractall(extract_path)
                    
            print(f"Arşiv başarıyla çıkartıldı: {extract_path}")
            return extract_path
            
        except Exception as e:
            print(f"Hata: Arşiv çıkartılırken bir sorun oluştu: {str(e)}")
            raise
    
    def git_clone(self, repo_url: str, target_dir: str = None) -> str:
        """Git repository'sini klonla"""
        try:
            if target_dir is None:
                target_dir = self.dataset_path
                
            if os.path.exists(target_dir):
                print(f"{target_dir} zaten mevcut, atlanıyor...")
                return target_dir
                
            print(f"Git repository klonlanıyor: {repo_url}")
            os.system(f"git clone {repo_url} {target_dir}")
            
            if not os.path.exists(target_dir):
                raise Exception("Git clone başarısız oldu")
                
            return target_dir
            
        except Exception as e:
            print(f"Hata: Git clone işlemi başarısız: {str(e)}")
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            raise 