"""
Sabit Değerler
"""

# Desteklenen dosya formatları
SUPPORTED_FORMATS = [
    'obj',  # Wavefront OBJ
    'stl',  # Stereolithography
    'ply',  # Stanford PLY
    'glb',  # GL Transmission Format Binary
    'gltf', # GL Transmission Format
    'fbx',  # Autodesk FBX
    'dae'   # COLLADA
]

# Varsayılan değerler
DEFAULT_RESOLUTION = 128
DEFAULT_FORMAT = 'obj'
DEFAULT_STYLE = ['Geleneksel']

# Dosya boyutu limitleri (bytes)
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
MAX_TEXTURE_SIZE = 16 * 1024 * 1024  # 16MB

# İşlem limitleri
MAX_VERTICES = 1_000_000
MAX_FACES = 500_000
MAX_BATCH_SIZE = 10

# API limitleri
MAX_REQUESTS_PER_MINUTE = 60
MAX_CONCURRENT_REQUESTS = 10 