from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import Any, Optional
import hashlib

import bittensor


class BaseStorage(ABC):
    @classmethod
    @abstractmethod
    def add_args(cls, parser: "ArgumentParser"):
        pass

    @abstractmethod
    def save_data(self, key: Any, data: Any, prefix: Optional[Any]) -> None:
        pass

    @abstractmethod
    def load_data(self, key: Any, prefix: Optional[Any]) -> Any:
        pass

    @abstractmethod
    def get_latest(self, prefix: Optional[Any]) -> Any:
        pass

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        pass

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        pass

    def get_config(self):
        parser = ArgumentParser()
        self.add_args(parser)
        return bittensor.config(parser)

    @staticmethod
    def generate_user_id(config: "bittensor.config") -> str:
        if not config or not hasattr(config, "wallet"):
            return "default"
            
        wallet_name = getattr(config.wallet, "name", "")
        hotkey = getattr(config.wallet, "hotkey", "")
        netuid = getattr(config, "netuid", "")
        
        unique_id = f"{wallet_name}_{hotkey}_{netuid}"
        return hashlib.md5(unique_id.encode()).hexdigest()
