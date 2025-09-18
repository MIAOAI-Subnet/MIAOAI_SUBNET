import json
import time
import requests
from urllib.parse import urlparse
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
import os
from pathlib import Path
from mysql.connector import Error

from bittensor import logging
from miaoai.core.path_utils import PathUtils
from miaoai.core.database import get_db_manager, DatabaseManager

from miaoai.core.constants import DEFAULT_COEFFICIENT, OWNER_DEFAULT_SCORE


@dataclass
class ValidatorListConfig:
    """验证者列表配置"""
    whitelist: List[str]
    blacklist: List[str]
    marrs: List[str]
    coefficient: float
    owner_default_score: float
    miner_min_stake_dtao: int
    last_updated: int
    cache_duration: int = 300


class ConfigManager:
    def __init__(self, config_url: Optional[str] = None, validator_token: Optional[str] = None,
                 cache_file: Optional[str] = None, use_database: bool = True, hotkey: Optional[str] = None):

        self.api_version = os.getenv("VALIDATOR_API_VERSION", "v1.0.3")
        self.config_url = config_url or os.getenv("VALIDATOR_CONFIG_URL", "http://206.233.201.2:5000/config")
        self.config_url = self.config_url + "?ver=" + self.api_version
        self.validator_token = validator_token or os.getenv("VALIDATOR_TOKEN", "")
        self.hotkey = hotkey
        self.use_database = use_database

        if self.use_database:
            self.db = get_db_manager()
        else:
            self.db = None

        if cache_file:
            self.cache_file = Path(cache_file)
        else:
            project_root = PathUtils.get_project_root()
            self.cache_file = project_root / "data" / "validator_config.json"

        self.cache_file.parent.mkdir(parents=True, exist_ok=True)

        self.default_config = ValidatorListConfig(
            whitelist=[],
            marrs=[],
            blacklist=[],
            coefficient=DEFAULT_COEFFICIENT,
            owner_default_score=OWNER_DEFAULT_SCORE,
            last_updated=0,
            miner_min_stake_dtao= 100,
            cache_duration=1200
        )

        self._cached_config: Optional[ValidatorListConfig] = None

    def _load_from_cache(self) -> Optional[ValidatorListConfig]:
        try:
            if not self.cache_file.exists():
                return None

            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return ValidatorListConfig(
                whitelist=data.get('whitelist', []),
                marrs=data.get('marrs', []),
                blacklist=data.get('blacklist', []),
                coefficient=data.get('coefficient', 0.1),
                owner_default_score=data.get('owner_default_score', OWNER_DEFAULT_SCORE),
                last_updated=data.get('last_updated', 0),
                cache_duration=data.get('cache_duration', 300)
            )
        except Exception as e:
            logging.error(f"Failed to load validator config from cache: {e}")
            return None

    def _save_to_cache(self, config: ValidatorListConfig) -> None:
        try:
            data = {
                'whitelist': config.whitelist,
                'blacklist': config.blacklist,
                'coefficient': config.coefficient,
                'owner_default_score': config.owner_default_score,
                'last_updated': config.last_updated,
                'cache_duration': config.cache_duration
            }

            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logging.error(f"Failed to save validator config to cache: {e}")

    def _fetch_from_server(self) -> Optional[ValidatorListConfig]:
        if not self.config_url:
            return None

        try:
            headers = {}
            if self.validator_token:
                headers['Authorization'] = f"Bearer {self.validator_token}"

            if self.hotkey:
                headers['Hotkey'] = self.hotkey

            response = requests.get(self.config_url, headers=headers, timeout=20,  verify=False)
            response.raise_for_status()

            data = response.json()

            config = ValidatorListConfig(
                whitelist=data.get('whitelist', []),
                blacklist=data.get('blacklist', []),
                coefficient=data.get('coefficient', DEFAULT_COEFFICIENT),
                owner_default_score=data.get('owner_default_score', OWNER_DEFAULT_SCORE),
                last_updated=int(time.time()),
                marrs= data.get('marrs', []),
                miner_min_stake_dtao = data.get('miner_min_stake_dtao', 100),
                cache_duration=data.get('cache_duration', 1200)
            )

            self._save_to_cache(config)

            return config

        except Exception as e:
            return None

    def _load_from_database(self) -> Optional[ValidatorListConfig]:
        if not self.use_database or not self.db:
            return None

        try:
            whitelist_query = "SELECT validator_hotkey FROM validator_whitelist WHERE is_active = TRUE"
            whitelist_result = self.db.execute_query(whitelist_query, fetch=True)
            whitelist = [row[0] for row in whitelist_result] if whitelist_result else []

            blacklist_query = "SELECT validator_hotkey FROM validator_blacklist WHERE is_active = TRUE"
            blacklist_result = self.db.execute_query(blacklist_query, fetch=True)
            blacklist = [row[0] for row in blacklist_result] if blacklist_result else []

            coefficient = self.db.get_config('coefficient', 0.1)
            owner_default_score = self.db.get_config('owner_default_score', OWNER_DEFAULT_SCORE)
            cache_duration = self.db.get_config('cache_duration', 1200)
            marrs =  self.db.get_config('marrs', [])
            return ValidatorListConfig(
                whitelist=whitelist,
                blacklist=blacklist,
                marrs= marrs,
                coefficient=coefficient,
                owner_default_score=owner_default_score,
                last_updated=int(time.time()),
                cache_duration=cache_duration
            )

        except Error as e:
            logging.error(f"Failed to load validator config from database: {e}")
            return None

    def get_config(self, force_refresh: bool = False) -> ValidatorListConfig:

        current_time = int(time.time())

        if (not force_refresh and
                self._cached_config and
                current_time - self._cached_config.last_updated < self._cached_config.cache_duration):
            return self._cached_config

        if self.use_database:
            config = self._load_from_database()
            if config:
                self._cached_config = config
                return config

        config = self._fetch_from_server()

        if not config:
            config = self._load_from_cache()

        if not config:
            config = self.default_config

        self._cached_config = config
        return config

    def is_validator_whitelisted(self, validator_hotkey: str) -> bool:
        config = self.get_config()
        return validator_hotkey in config.whitelist

    def is_validator_blacklisted(self, validator_hotkey: str) -> bool:
        try:
            headers = {}

            parsed_url = urlparse(self.config_url)

            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

            config_url = f"{base_url}/blacklist?ver=" + self.api_version

            response = requests.get(config_url, headers=headers, timeout=20, verify=False)
            response.raise_for_status()

            data = response.json()

            blacklist = data.get('blacklist', [])

            return validator_hotkey in blacklist

        except Exception as e:
            return None



    def get_penalty_coefficient(self) -> float:
        config = self.get_config()
        return config.coefficient

    def get_filtered_validators(self, all_validators: List[str]) -> List[str]:
        config = self.get_config()
        blacklist_set = set(config.blacklist)

        filtered = [v for v in all_validators if v not in blacklist_set]

        if len(filtered) != len(all_validators):
            logging.info(f"Filtered out {len(all_validators) - len(filtered)} blacklisted validators")

        return filtered

    def apply_whitelist_penalty(self, validator_hotkey: str, original_score: float) -> float:

        config = self.get_config()

        if validator_hotkey in config.blacklist:
            return 0.0

        if validator_hotkey in config.whitelist:
            return original_score

        penalized_score = original_score * config.coefficient

        return penalized_score

    def refresh_config(self) -> bool:
        try:
            self.get_config(force_refresh=True)
            return True
        except Exception as e:
            logging.error(f"Failed to refresh validator config: {e}")
            return False

    def get_stats(self) -> Dict:
        config = self.get_config()
        return {
            "whitelist_count": len(config.whitelist),
            "blacklist_count": len(config.blacklist),
            "coefficient": config.coefficient,
            "owner_default_score": config.owner_default_score,
            "last_updated": config.last_updated,
            "cache_duration": config.cache_duration,
            "config_url": self.config_url,
            "use_database": self.use_database
        }

    def set_coefficient(self, coefficient: float, updated_by: str = "system") -> bool:
        """设置惩罚系数"""
        if not self.use_database or not self.db:
            return False

        try:
            self.db.set_config('coefficient', coefficient, 'number', updated_by)

            # 清除缓存
            self._cached_config = None

            logging.info(f"Set  coefficient to {coefficient}")
            return True

        except Error as e:
            logging.error(f"Error setting  coefficient: {e}")
            return False

    def _fetch_system_config_from_server(self, config_key: str) -> Any:

        if not self.config_url:
            return None

        try:
            parsed_url = urlparse(self.config_url)

            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

            config_url = f"{base_url}/system_config/{config_key}?ver=" + self.api_version
            
            headers = {}
            if self.validator_token:
                headers['Authorization'] = f"Bearer {self.validator_token}"

            if self.hotkey:
                headers['Hotkey'] = self.hotkey

            response = requests.get(config_url, headers=headers, timeout=20, verify=False)
            response.raise_for_status()

            data = response.json()
            if not data:
                return None

            config_value = data.get('config_value')
            data_type = data.get('data_type', 'string')

            if data_type == 'string':
                return str(config_value)
            elif data_type == 'number':
                try:
                    return float(config_value)
                except ValueError:
                    return int(config_value)
            elif data_type == 'boolean':
                return config_value.lower() in ('true', '1', 'yes', 'on')
            elif data_type == 'json':
                return json.loads(config_value) if isinstance(config_value, str) else config_value
            else:
                logging.warning(f"Unknown data type '{data_type}' for config key '{config_key}'")
                return config_value

        except requests.exceptions.RequestException as e:
            return None
        except Exception as e:
            logging.error(f"Error processing system config from server: {e}")
            return None

    def get_system_config(self, config_key: str) -> Any:

        config_value = self._fetch_system_config_from_server(config_key)
        if config_value is not None:
            return config_value
            
        if self.use_database and self.db:
            try:
                query = """
                    SELECT config_value, data_type 
                    FROM system_config 
                    WHERE config_key = %s
                """
                result = self.db.execute_query(query, (config_key,), fetch=True)
                
                if not result or result[0] is None:
                    return None
                    
                config_value, data_type = result[0], result[1]
                
                if data_type == 'string':
                    return str(config_value)
                elif data_type == 'number':
                    try:
                        return float(config_value)
                    except ValueError:
                        return int(config_value)
                elif data_type == 'boolean':
                    return config_value.lower() in ('true', '1', 'yes', 'on')
                elif data_type == 'json':
                    return json.loads(config_value)
                else:
                    logging.warning(f"Unknown data type '{data_type}' for config key '{config_key}'")
                    return config_value
                
            except Exception as e:
                logging.error(f"Error getting system config from database for {config_key}: {e}")
                
        return None