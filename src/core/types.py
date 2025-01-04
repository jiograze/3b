"""
Tip Tanımlamaları
"""

from os import PathLike as OSPathLike
from typing import Union, Dict, Any, TypeVar
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

# Yol tipleri
PathLike = Union[str, Path, OSPathLike]

# Mesh tipi
Mesh = TypeVar('Mesh')

# Doku haritası tipi (H, W, C) şeklinde uint8 dizisi
TextureMap = NDArray

# İşlem sonucu tipi
ProcessingResult = Dict[str, Any] 