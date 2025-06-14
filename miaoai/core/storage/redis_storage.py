import argparse
import os
from typing import cast, Any, Optional

import redis

from miaoai.core.storage.base_storage import BaseStorage
from miaoai.core.storage.utils import check_key, dumps, loads

REDIS_DEFAULT_HOST = "localhost"
REDIS_DEFAULT_PWD = ""
REDIS_DEFAULT_PORT = 6379
REDIS_DEFAULT_TTL = 7200
REDIS_DEFAULT_DB = 0


class BaseRedisStorage(BaseStorage):
    def __init__(self, config=None):
        self.config = config or self.get_config()
        self._port = config.redis_port or REDIS_DEFAULT_PORT
        self._host = config.redis_host or REDIS_DEFAULT_HOST
        self._password = config.redis_password or REDIS_DEFAULT_PWD
        self._db = config.redis_db or REDIS_DEFAULT_DB

        self.client = redis.Redis(host=self._host,password=self._password ,  port=self._port, db=self._db)

        self.ttl = config.redis_ttl or REDIS_DEFAULT_TTL
        self.update_redis_client()
        self.check_health()

    def update_redis_client(self):
        self.client.config_set(name="appendonly", value="yes")

    def check_health(self):
        try:
            self.client.ping()
        except redis.exceptions.ConnectionError:
            raise ConnectionError("Redis connection error")

    @classmethod
    def add_args(cls, parser: "argparse.ArgumentParser"):
        redis_group = parser.add_argument_group("redis storage")
        redis_group.add_argument(
            "--redis_host",
            type=str,
            default=os.getenv("REDIS_HOST", REDIS_DEFAULT_HOST),
            help="Redis host",
        )
        redis_group.add_argument(
            "--redis_password",
            type=str,
            default=os.getenv("REDIS_PASSWORD", REDIS_DEFAULT_PWD),
            help="Redis password",
        )
        redis_group.add_argument(
            "--redis_port",
            type=int,
            default=os.getenv("REDIS_PORT", REDIS_DEFAULT_PORT),
            help="Redis port",
        )
        redis_group.add_argument(
            "--redis_ttl",
            type=int,
            default=os.getenv("REDIS_TTL", REDIS_DEFAULT_TTL),
            help="TTL for pool data in seconds",
        )
        redis_group.add_argument(
            "--redis_db",
            type=int,
            default=os.getenv("REDIS_DB", REDIS_DEFAULT_DB),
            help="Redis database",
        )

    def save_data(self, key: Optional[Any], data: Any, prefix: str = "pools") -> None:

        check_key(key)

        dumped_data = dumps(data)

        pipe = self.client.pipeline()
        key_name = f"{prefix}:{key}" if key else prefix
        pipe.set(name=key_name, value=dumped_data, ex=self.ttl)
        pipe.set(name=f"{prefix}:latest_block", value=key)
        pipe.execute()

    def load_data(self, key: Optional[Any], prefix: str = "pools") -> Optional[dict]:

        check_key(key)

        key_name = f"{prefix}:{key}" if key else prefix
        dumped_data = cast(bytes, self.client.get(name=key_name))
        data = loads(dumped_data) if dumped_data else None
        return data if data else None

    def get_latest(self, prefix: str = "pools") -> Optional[Any]:

        latest_block = cast(str, self.client.get(f"{prefix}:latest_block"))
        if latest_block is None:
            return None

        data = cast(bytes, self.client.get(f"{prefix}:{int(latest_block)}"))
        return loads(data) if data else None

    def get(self, key: str, default: Any = None) -> Any:

        value = self.client.get(f"state:{key}")
        if value is None:
            return default
        return loads(value)
        
    def set(self, key: str, value: Any) -> None:

        dumped_data = dumps(value)
        self.client.set(f"state:{key}", dumped_data, ex=self.ttl)
