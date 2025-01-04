"""
Özel Hata Sınıfları
"""

class BaseError(Exception):
    """Temel hata sınıfı"""
    def __init__(self, message: str = None):
        self.message = message or self.__class__.__doc__
        super().__init__(self.message)

class ModelError(BaseError):
    """Model yükleme veya çalıştırma hatası"""
    pass

class ProcessingError(BaseError):
    """İşleme hatası"""
    pass

class ValidationError(BaseError):
    """Doğrulama hatası"""
    pass

class FileError(BaseError):
    """Dosya işleme hatası"""
    pass

class ConfigError(BaseError):
    """Yapılandırma hatası"""
    pass

class APIError(BaseError):
    """API hatası"""
    pass

class ResourceError(BaseError):
    """Kaynak bulunamadı hatası"""
    pass

class SecurityError(BaseError):
    """Güvenlik hatası"""
    pass 