
from enum import Enum


class BackendType(Enum):
    OPENCV = 1
    PYAV = 2

DEFAULT_BACKEND = BackendType.OPENCV

def open_video(path, backend=DEFAULT_BACKEND, allow_fallback=True):
    pass

