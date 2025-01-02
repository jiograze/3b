import os
import json
import yaml
from pathlib import Path
from utils.logging.logger import get_logger
from utils.helpers.exceptions import StructureError, ValidationError

logger = get_logger('structure')

class ProjectStructure:
    def __init__(self):
        self.base_dir = Path('.')
        self.config = {
            'directories': [
                # Data directories
                'data/images',
                'data/3d_models',
                'data/text_prompts',
                'data/datasets/COCO',
                'data/datasets/ImageNet',
                'data/datasets/ShapeNet',
                'data/datasets/Pix3D',
                'data/feedback',
                
                # Model directories
                'models/checkpoints',
                'models/pretrained',
                'models/generated',
                'models/architecture',
                'models/weights',
                'models/configs',
                
                # Core modules
                'modules/core',
                'modules/data_management',
                'modules/nlp',
                'modules/image_processing',
                'modules/model_generation',
                'modules/training',
                'modules/evaluation',
                'modules/ui',
                'modules/security',
                'modules/deployment',
                
                # Utils and others
                'utils/helpers',
                'utils/config',
                'utils/logging',
                'tests/unit',
                'tests/integration',
                'tests/end_to_end',
                'docs/user_guide',
                'docs/api_docs',
                'docs/developer_guide',
                'config',
                'scripts',
                'logs'
            ],
            'files': {
                'config/model_config.yaml': {
                    'model': {
                        'name': 'otuken3d',
                        'version': '1.0.0',
                        'base_model': 'point-e',
                        'language': 'tr'
                    },
                    'architecture': {
                        'embedding_dim': 512,
                        'hidden_dim': 1024,
                        'num_layers': 12
                    },
                    'training': {
                        'batch_size': 32,
                        'learning_rate': 0.0001,
                        'max_epochs': 100
                    }
                },
                'config/cultural_config.json': {
                    'motifs': ['kıvrım', 'rumi', 'hatai', 'palmet'],
                    'styles': ['Göktürk', 'Uygur', 'Selçuklu', 'Osmanlı'],
                    'materials': ['ahşap', 'taş', 'metal', 'seramik']
                },
                'README.md': '''# Ötüken3D
Türk kültürüne özgü 3D model üretimi için özelleştirilmiş yapay zeka modeli.

## Dizin Yapısı
- data/: Veri setleri ve işlenmiş veriler
- models/: Model dosyaları ve ağırlıkları
- modules/: Temel modüller ve işlevler
- utils/: Yardımcı araçlar
- config/: Konfigürasyon dosyaları
- tests/: Test dosyaları
- docs/: Dokümantasyon
'''
            }
        }

    def create_structure(self):
        """Proje dizin yapısını oluştur"""
        try:
            # Ana dizinleri oluştur
            for dir_path in self.config['directories']:
                dir_path = self.base_dir / dir_path
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    if 'modules' in str(dir_path):
                        (dir_path / '__init__.py').touch()
                    logger.info(f"Dizin oluşturuldu: {dir_path}")
                except Exception as e:
                    raise StructureError(f"Dizin oluşturma hatası ({dir_path}): {str(e)}")

            # Konfigürasyon dosyalarını oluştur
            for file_path, content in self.config['files'].items():
                full_path = self.base_dir / file_path
                try:
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    if file_path.endswith('.json'):
                        with open(full_path, 'w', encoding='utf-8') as f:
                            json.dump(content, f, indent=4, ensure_ascii=False)
                    elif file_path.endswith('.yaml'):
                        with open(full_path, 'w', encoding='utf-8') as f:
                            yaml.dump(content, f, allow_unicode=True)
                    else:
                        full_path.write_text(content, encoding='utf-8')
                        
                    logger.info(f"Dosya oluşturuldu: {full_path}")
                except Exception as e:
                    raise StructureError(f"Dosya oluşturma hatası ({full_path}): {str(e)}")

            return True

        except StructureError as e:
            logger.error(str(e))
            return False
        except Exception as e:
            logger.error(f"Beklenmeyen hata: {str(e)}")
            return False

    def verify_structure(self):
        """Oluşturulan yapıyı doğrula"""
        try:
            # Dizinleri kontrol et
            for dir_path in self.config['directories']:
                dir_path = self.base_dir / dir_path
                if not dir_path.is_dir():
                    raise ValidationError(f"Dizin bulunamadı: {dir_path}")
                logger.debug(f"Dizin doğrulandı: {dir_path}")

            # Dosyaları kontrol et
            for file_path in self.config['files']:
                full_path = self.base_dir / file_path
                if not full_path.is_file():
                    raise ValidationError(f"Dosya bulunamadı: {full_path}")
                logger.debug(f"Dosya doğrulandı: {full_path}")

            logger.info("Tüm yapı başarıyla doğrulandı")
            return True

        except ValidationError as e:
            logger.error(str(e))
            return False
        except Exception as e:
            logger.error(f"Doğrulama sırasında beklenmeyen hata: {str(e)}")
            return False

def main():
    setup = ProjectStructure()
    
    try:
        if setup.create_structure():
            if setup.verify_structure():
                logger.info("✓ Proje yapısı başarıyla oluşturuldu ve doğrulandı.")
                return True
            else:
                logger.error("✗ Yapı doğrulama hatası!")
        else:
            logger.error("✗ Yapı oluşturma hatası!")
        
        return False
        
    except Exception as e:
        logger.critical(f"Kritik hata: {str(e)}")
        return False

if __name__ == "__main__":
    exit(0 if main() else 1) 