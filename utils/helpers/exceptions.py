class Otuken3DError(Exception):
    """Temel hata sınıfı"""
    pass

class StructureError(Otuken3DError):
    """Proje yapısı ile ilgili hatalar"""
    pass

class DatasetError(Otuken3DError):
    """Veri seti işlemleri ile ilgili hatalar"""
    pass

class DiskSpaceError(DatasetError):
    """Disk alanı yetersizliği hataları"""
    pass

class DownloadError(DatasetError):
    """İndirme işlemi hataları"""
    pass

class ValidationError(Otuken3DError):
    """Doğrulama hataları"""
    pass

class DependencyError(Otuken3DError):
    """Bağımlılık hataları"""
    pass 