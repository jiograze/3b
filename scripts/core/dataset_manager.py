import os
import sys
import shutil
from pathlib import Path
from tqdm import tqdm
from modules.utils.download_utils import DatasetDownloader
from utils.logging.logger import get_logger
from utils.helpers.exceptions import DatasetError, DiskSpaceError, DownloadError

logger = get_logger('dataset')

class DatasetManager:
    def __init__(self):
        self.datasets = {
            'thingi10k': {
                'size': 1.2,  # GB
                'priority': 1,
                'urls': [
                    "https://ten-thousand-models.appspot.com/archive/Thingi10K.zip",
                    "https://databasearchive.org/Thingi10K/Thingi10K.zip",
                    "https://dl.thingiverse.com/archives/Thingi10K.zip"
                ]
            },
            'shapenet': {
                'size': 2.0,
                'priority': 2,
                'url': 'http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v2.zip'
            },
            'abc': {
                'size': 15.0,
                'priority': 3,
                'base_url': 'https://deep-geometry.github.io/abc-dataset/data/abc',
                'parts': 5
            },
            'google_scanned': {
                'size': 8.0,
                'priority': 3,
                'repo_url': 'https://github.com/google-research-datasets/Scanned-Objects.git'
            },
            'kitti360': {
                'size': 7.0,
                'priority': 2,
                'url': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_3d_raw.zip'
            },
            'modelnet': {
                'size': 1.5,
                'priority': 2,
                'url': 'http://modelnet.cs.princeton.edu/ModelNet40.zip'
            }
        }
        self.downloader = DatasetDownloader('datasets')
        
    def check_disk_space(self, required_space_gb):
        """Disk alanı kontrolü"""
        try:
            data_dir = Path('data/datasets')
            total, used, free = shutil.disk_usage(data_dir)
            free_gb = free // (2**30)
            if free_gb < required_space_gb:
                raise DiskSpaceError(
                    f"Yetersiz disk alanı. Gerekli: {required_space_gb}GB, Mevcut: {free_gb}GB"
                )
            logger.debug(f"Disk alanı yeterli. Gerekli: {required_space_gb}GB, Mevcut: {free_gb}GB")
            return True
        except DiskSpaceError:
            raise
        except Exception as e:
            raise DatasetError(f"Disk alanı kontrolü sırasında hata: {str(e)}")

    def download_thingi10k(self):
        """Thingi10K veri setini indir"""
        dataset = self.datasets['thingi10k']
        downloaded = False
        
        for url in dataset['urls']:
            try:
                logger.info(f"Thingi10K indiriliyor: {url}")
                archive_path = self.downloader.download_file(url, "Thingi10K.zip")
                self.downloader.extract_archive(archive_path)
                downloaded = True
                logger.info("Thingi10K başarıyla indirildi")
                break
            except Exception as e:
                logger.warning(f"İndirme hatası ({url}): {str(e)}")
                continue
        
        if not downloaded:
            raise DownloadError("Thingi10K hiçbir kaynaktan indirilemedi")
        return True

    def download_shapenet(self):
        """ShapeNet veri setini indir"""
        try:
            dataset = self.datasets['shapenet']
            logger.info("ShapeNet indiriliyor...")
            archive_path = self.downloader.download_file(dataset['url'], "ShapeNetCore.v2.zip")
            self.downloader.extract_archive(archive_path)
            logger.info("ShapeNet başarıyla indirildi")
            return True
        except Exception as e:
            raise DownloadError(f"ShapeNet indirme hatası: {str(e)}")

    def download_abc(self):
        """ABC Dataset'i indir"""
        try:
            dataset = self.datasets['abc']
            logger.info("ABC Dataset indiriliyor...")
            for i in range(1, dataset['parts'] + 1):
                url = f"{dataset['base_url']}/abc_{i}.zip"
                archive_path = self.downloader.download_file(url, f"abc_{i}.zip")
                self.downloader.extract_archive(archive_path)
            logger.info("ABC Dataset başarıyla indirildi")
            return True
        except Exception as e:
            raise DownloadError(f"ABC indirme hatası: {str(e)}")

    def download_google_scanned(self):
        """Google Scanned Objects veri setini indir"""
        try:
            dataset = self.datasets['google_scanned']
            logger.info("Google Scanned Objects indiriliyor...")
            self.downloader.git_clone(dataset['repo_url'])
            logger.info("Google Scanned Objects başarıyla indirildi")
            return True
        except Exception as e:
            raise DownloadError(f"Google Scanned Objects indirme hatası: {str(e)}")

    def download_kitti360(self):
        """KITTI-360 veri setini indir"""
        try:
            dataset = self.datasets['kitti360']
            logger.info("KITTI-360 indiriliyor...")
            archive_path = self.downloader.download_file(dataset['url'], "data_3d_raw.zip")
            self.downloader.extract_archive(archive_path)
            logger.info("KITTI-360 başarıyla indirildi")
            return True
        except Exception as e:
            raise DownloadError(f"KITTI-360 indirme hatası: {str(e)}")

    def download_modelnet(self):
        """ModelNet veri setini indir"""
        try:
            dataset = self.datasets['modelnet']
            logger.info("ModelNet indiriliyor...")
            archive_path = self.downloader.download_file(dataset['url'], "ModelNet40.zip")
            self.downloader.extract_archive(archive_path)
            logger.info("ModelNet başarıyla indirildi")
            return True
        except Exception as e:
            raise DownloadError(f"ModelNet indirme hatası: {str(e)}")

    def download_by_priority(self, priority_level):
        """Öncelik seviyesine göre veri setlerini indir"""
        total_size = sum(d['size'] for d in self.datasets.values() 
                        if d.get('priority', 999) <= priority_level)
        
        try:
            # Disk alanı kontrolü
            self.check_disk_space(total_size)
            
            # Veri setlerini indir
            for name, dataset in self.datasets.items():
                if dataset.get('priority', 999) <= priority_level:
                    try:
                        download_method = getattr(self, f"download_{name}")
                        download_method()
                        logger.info(f"{name} başarıyla indirildi")
                    except (DownloadError, Exception) as e:
                        logger.error(f"{name} indirilemedi: {str(e)}")
                        
        except DiskSpaceError as e:
            logger.error(str(e))
            raise
        except Exception as e:
            logger.error(f"Veri seti indirme hatası: {str(e)}")
            raise DatasetError(str(e))

    def verify_downloads(self):
        """İndirilen veri setlerinin doğruluğunu kontrol et"""
        missing_datasets = []
        for name in self.datasets:
            path = Path(f"data/datasets/{name}")
            if path.exists():
                logger.info(f"{name} veri seti mevcut")
            else:
                logger.warning(f"{name} veri seti bulunamadı")
                missing_datasets.append(name)
        
        if missing_datasets:
            logger.warning(f"Eksik veri setleri: {', '.join(missing_datasets)}")
        else:
            logger.info("Tüm veri setleri mevcut")

def main():
    manager = DatasetManager()
    
    try:
        # Öncelik 1 olan veri setlerini indir
        logger.info("Öncelikli veri setleri indiriliyor...")
        manager.download_by_priority(1)
        
        # Doğrulama yap
        logger.info("Veri setleri kontrol ediliyor...")
        manager.verify_downloads()
        
        return True
    except (DatasetError, DiskSpaceError) as e:
        logger.error(str(e))
        return False
    except Exception as e:
        logger.critical(f"Beklenmeyen hata: {str(e)}")
        return False

if __name__ == "__main__":
    exit(0 if main() else 1) 