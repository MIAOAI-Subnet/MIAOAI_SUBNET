from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from miaoai.core.path_utils import PathUtils
import time

def load_public_key(pub_path: str):
    project_root = PathUtils.get_project_root()
    _pub_path = project_root / pub_path
    with open(_pub_path, "rb") as key_file:
        return serialization.load_pem_public_key(key_file.read())

def verify_signature(public_key, message: bytes, signature: bytes) -> bool:
    try:
        public_key.verify(
            signature,
            message,
            padding.PKCS1v15(),
            hashes.SHA256()
        )
        return True
    except Exception as e:
        return False


def load_private_key(priv_path: str, password: bytes = None):
    project_root = PathUtils.get_project_root()
    _priv_path = project_root / priv_path
    with open(_priv_path, "rb") as key_file:
        return serialization.load_pem_private_key(key_file.read(), password=password)


def sign_message(private_key, message: bytes) -> bytes:
    signature = private_key.sign(
        message,
        padding.PKCS1v15(),
        hashes.SHA256()
    )
    return signature


