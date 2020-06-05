import os
from pathlib import Path


def root_dir() -> Path:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return Path(current_dir)


def ensure_dir(path: Path) -> Path:
    path = os.path.abspath(path)
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except FileExistsError:
            pass
    return Path(path)
