"""
Özel Tip Tanımlamaları
"""

from pathlib import Path
from typing import Union, TypeVar, Any, Dict, List, Tuple, Optional
import numpy as np

# Yol tipleri
PathLike = Union[str, Path]

# Numpy dizileri
NDArray = Union[np.ndarray, Any]

# Mesh veri yapısı
Vertices = NDArray  # (N, 3) şeklinde float dizisi
Faces = NDArray     # (M, 3) şeklinde int dizisi
UVs = NDArray       # (N, 2) şeklinde float dizisi
Normals = NDArray   # (N, 3) şeklinde float dizisi

# Doku tipleri
TextureMap = NDArray  # (H, W, C) şeklinde uint8 dizisi

# Konfigürasyon tipleri
Config = Dict[str, Any]
ConfigValue = Union[str, int, float, bool, List, Dict]

# Jenerik tipler
T = TypeVar('T')
U = TypeVar('U')

# Mesh metrikleri
MeshMetrics = Dict[str, Union[int, float]]

# Dönüşüm matrisi
TransformMatrix = NDArray  # (4, 4) şeklinde float dizisi

# Renk değerleri
Color = Union[Tuple[int, int, int], Tuple[int, int, int, int]]

# Materyal özellikleri
MaterialProperties = Dict[str, Union[float, Color, PathLike]]

# İşlem sonucu
ProcessingResult = Dict[str, Any]

# Hata mesajları
ErrorMessage = Dict[str, str]

# API yanıtları
APIResponse = Dict[str, Any]

# İlerleme bilgisi
ProgressInfo = Dict[str, Union[int, float, str]]

# Dosya bilgisi
FileInfo = Dict[str, Union[str, int, PathLike]]

# İşlem seçenekleri
ProcessingOptions = Dict[str, Any]

# Geometri primitive'leri
Point = Tuple[float, float, float]
Vector = Tuple[float, float, float]
BoundingBox = Tuple[Point, Point]

# Dönüşüm parametreleri
TransformParams = Dict[str, Union[float, Vector]]

# Optimizasyon parametreleri
OptimizationParams = Dict[str, Union[int, float, bool]]

# Materyal haritaları
MaterialMaps = Dict[str, TextureMap]

# UV koordinatları
UVCoordinates = Dict[str, UVs]

# Mesh topolojisi
MeshTopology = Dict[str, Union[Vertices, Faces, Normals, UVs]]

# Animasyon verileri
AnimationData = Dict[str, Union[List[float], List[TransformMatrix]]]

# Render ayarları
RenderSettings = Dict[str, Union[int, float, Color, bool]]

# Işık parametreleri
LightParams = Dict[str, Union[Color, Vector, float]]

# Kamera parametreleri
CameraParams = Dict[str, Union[Vector, float]]

# Sahne yapılandırması
SceneSetup = Dict[str, Union[CameraParams, List[LightParams], RenderSettings]] 