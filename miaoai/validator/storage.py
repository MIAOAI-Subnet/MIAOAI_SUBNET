from typing import Optional, Union

from bittensor.core.config import Config
from bittensor.utils.btlogging import logging

from miaoai.core.storage import  BaseRedisStorage


class RedisValidatorStorage(BaseRedisStorage):
    def __init__(self, config: Optional["Config"] = None):
        super().__init__(config)
        self.validator_id = self.generate_user_id(config)

    def save_state(self, state: dict) -> None:
        prefix = f"{self.validator_id}_state"
        self.save_data(key="current", data=state, prefix=prefix)

    def load_latest_state(self) -> dict:
        prefix = f"{self.validator_id}_state"
        return self.load_data(key="current", prefix=prefix)


STORAGE_CLASSES = { "redis": RedisValidatorStorage}


def get_validator_storage(
    storage_type: str, config: "Config"
) -> Union["RedisValidatorStorage"]:
    if storage_type not in STORAGE_CLASSES:
        raise ValueError(f"Unknown storage type: {storage_type}")

    storage_class = STORAGE_CLASSES[storage_type]

    try:
        return storage_class(config)
    except Exception as e:
        message = f"Failed to initialize {storage_type} storage: {e}"
        logging.error(message)
        raise Exception(message)
