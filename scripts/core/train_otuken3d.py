import logging
from pathlib import Path
from modules.training.model_trainer import Otuken3DTrainer

def setup_logging():
    """Loglama sistemini kur"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training.log'),
            logging.StreamHandler()
        ]
    )

def main():
    # Loglama sistemini kur
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Eğitim yöneticisini başlat
        trainer = Otuken3DTrainer()
        
        # Modeli hazırla
        if not trainer.prepare_model():
            logger.error("Model hazırlama başarısız!")
            return False
            
        # Eğitimi başlat
        logger.info("Ötüken3D model eğitimi başlıyor...")
        success = trainer.train()
        
        if success:
            logger.info("Eğitim başarıyla tamamlandı!")
        else:
            logger.error("Eğitim sırasında hata oluştu!")
            
        return success
        
    except Exception as e:
        logger.error(f"Beklenmeyen hata: {str(e)}")
        return False

if __name__ == "__main__":
    exit(0 if main() else 1) 