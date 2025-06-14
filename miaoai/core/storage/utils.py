import pickle
from pathlib import Path


def extract_block_number(file: Path) -> int:
    try:
        return int(file.stem.split("-")[1])
    except (IndexError, ValueError):
        return -1


def check_key(key):
    try:
        return str(key)
    except TypeError:
        raise TypeError(f"Key '{key}' cannot be converted to string.")


def dumps(obj) -> bytes:
    try:
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        raise Exception(f"Failed to pickle object: {e}")


def loads(blob: bytes):
    try:
        return pickle.loads(blob)
    except Exception as e:
        raise Exception(f"Failed to unpickle object: {e}")
