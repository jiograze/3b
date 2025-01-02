"""Constants used throughout Ötüken3D."""

from enum import Enum
from pathlib import Path

# Model related constants
class ModelType(str, Enum):
    TEXT_TO_3D = "text_to_3d"
    IMAGE_TO_3D = "image_to_3d"
    MESH_PROCESSING = "mesh_processing"
    TEXTURE_GENERATION = "texture_generation"
    STYLE_TRANSFER = "style_transfer"

MODEL_TYPES = [model_type.value for model_type in ModelType]

# File format constants
class FileFormat(str, Enum):
    OBJ = "obj"
    STL = "stl"
    PLY = "ply"
    GLTF = "gltf"
    FBX = "fbx"
    USD = "usd"

SUPPORTED_FORMATS = [fmt.value for fmt in FileFormat]

# Path constants
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
CACHE_DIR = ROOT_DIR / "cache"
LOGS_DIR = ROOT_DIR / "logs"

# Training constants
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_NUM_EPOCHS = 100
MAX_SEQUENCE_LENGTH = 512

# Model architecture constants
EMBEDDING_DIM = 768
HIDDEN_DIM = 1024
NUM_ATTENTION_HEADS = 12
NUM_TRANSFORMER_LAYERS = 6

# Processing constants
MAX_POINTS = 100000
VOXEL_RESOLUTION = 128
MESH_SIMPLIFICATION_TARGET = 10000
TEXTURE_RESOLUTION = 1024

# API constants
DEFAULT_API_VERSION = "v1"
DEFAULT_TIMEOUT = 30  # seconds
MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB

# Security constants
TOKEN_EXPIRY = 24 * 60 * 60  # 24 hours
MAX_LOGIN_ATTEMPTS = 5
PASSWORD_MIN_LENGTH = 8

# Resource limits
MAX_GPU_MEMORY = 0.9  # 90% of available GPU memory
MAX_BATCH_MEMORY = 0.5  # 50% of available GPU memory per batch
CPU_THREADS = 4 