import json
import base64

from typing import Optional, Union

from bittensor.core.config import Config
from bittensor.utils.btlogging import logging

from miaoai.core.storage import  BaseRedisStorage

class RedisStorage(BaseRedisStorage):
    def __init__(self, config: Optional["Config"] = None):
        super().__init__(config)
        self.miner_id = self.generate_user_id(config)

    def save_pool_data(self, block_number: int, pool_mapping: dict) -> None:
        prefix = f"{self.miner_id}_pools"
        self.save_data(key=block_number, data=pool_mapping, prefix=prefix)

    def get_pool_info(self, block_number: int) -> Optional[dict]:
        prefix = f"{self.miner_id}_pools"
        return self.load_data(key=block_number, prefix=prefix)

    def get_latest_pool_info(self) -> Optional[dict]:
        prefix = f"{self.miner_id}_pools"
        return self.get_latest(prefix=prefix)

    def save_schedule(self, block_number: int, schedule_obj) -> None:
        prefix = f"{self.miner_id}_schedule"
        self.save_data(key=block_number, data=schedule_obj, prefix=prefix)

    def load_latest_schedule(self) -> Optional[dict]:
        prefix = f"{self.miner_id}_schedule"
        return self.get_latest(prefix=prefix)


STORAGE_CLASSES = { "redis": RedisStorage}


def get_miner_storage(
    storage_type: str, config: "Config"
) -> Union[ RedisStorage]:
    if storage_type not in STORAGE_CLASSES:
        raise ValueError(f"Unknown storage type: {storage_type}")

    storage_class = STORAGE_CLASSES[storage_type]

    try:
        return storage_class(config)
    except Exception as e:
        message = f"Failed to initialize {storage_type} storage: {e}"
        logging.error(message)
        raise Exception(message)
