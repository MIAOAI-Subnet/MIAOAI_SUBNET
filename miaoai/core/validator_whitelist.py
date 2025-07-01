import json
import time
import requests
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import os
from pathlib import Path
from mysql.connector import Error

from bittensor import logging
from miaoai.core.path_utils import PathUtils
from miaoai.core.database import get_db_manager

from miaoai.core.constants import DEFAULT_PENALTY_COEFFICIENT,  OWNER_DEFAULT_SCORE


@dataclass
class ValidatorListConfig:
    whitelist: List[str]
    blacklist: List[str]
    penalty_coefficient: float
    owner_default_score: float
    last_updated: int
    cache_duration: int = 300


class ValidatorWhitelistManager:

    def __init__(self, config_url: Optional[str] = None, validator_token: Optional[str] = None, cache_file: Optional[str] = None, use_database: bool = True, hotkey: Optional[str] = None):

        self.config_url = config_url or os.getenv("VALIDATOR_CONFIG_URL", "http://206.233.201.2:5000/config")
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
            blacklist=[],
            penalty_coefficient= DEFAULT_PENALTY_COEFFICIENT,
            owner_default_score = OWNER_DEFAULT_SCORE,
            last_updated=0,
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
                blacklist=data.get('blacklist', []),
                penalty_coefficient=data.get('penalty_coefficient', 0.1),
                owner_default_score=data.get('owner_default_score', OWNER_DEFAULT_SCORE),
                last_updated=data.get('last_updated', 0),
                cache_duration=data.get('cache_duration', 300)
            )
        except Exception as e:
            return None
            
    def _save_to_cache(self, config: ValidatorListConfig) -> None:
        try:
            data = {
                'whitelist': config.whitelist,
                'blacklist': config.blacklist,
                'penalty_coefficient': config.penalty_coefficient,
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
            logging.debug("No validator config URL provided")
            return None
            
        try:
            headers = {}
            if self.validator_token:
                headers['Authorization'] = f"Bearer {self.validator_token}"

            if self.hotkey:
                headers['Hotkey'] = self.hotkey

            response = requests.get(self.config_url, headers=headers, timeout=20)
            response.raise_for_status()
            
            data = response.json()
            
            config = ValidatorListConfig(
                whitelist=data.get('whitelist', []),
                blacklist=data.get('blacklist', []),
                penalty_coefficient=data.get('penalty_coefficient', DEFAULT_PENALTY_COEFFICIENT),
                owner_default_score=data.get('owner_default_score', OWNER_DEFAULT_SCORE),
                last_updated=int(time.time()),
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
            
            penalty_coefficient = self.db.get_config('penalty_coefficient', 0.1)
            owner_default_score = self.db.get_config('owner_default_score', OWNER_DEFAULT_SCORE)
            cache_duration = self.db.get_config('cache_duration', 1200)
            
            return ValidatorListConfig(
                whitelist=whitelist,
                blacklist=blacklist,
                penalty_coefficient=penalty_coefficient,
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
            logging.warning("Using default validator config")
            config = self.default_config
            
        self._cached_config = config
        return config
        
    def is_validator_whitelisted(self, validator_hotkey: str) -> bool:
        config = self.get_config()
        return validator_hotkey in config.whitelist
        
    def is_validator_blacklisted(self, validator_hotkey: str) -> bool:
        config = self.get_config()
        return validator_hotkey in config.blacklist
        
    def get_penalty_coefficient(self) -> float:
        config = self.get_config()
        return config.penalty_coefficient
        
    def get_filtered_validators(self, all_validators: List[str]) -> List[str]:

        config = self.get_config()
        blacklist_set = set(config.blacklist)
        
        filtered = [v for v in all_validators if v not in blacklist_set]

            
        return filtered
        
    def apply_whitelist_penalty(self, validator_hotkey: str, original_score: float) -> float:

        config = self.get_config()
        
        if validator_hotkey in config.blacklist:
            return 0.0
            
        if validator_hotkey in config.whitelist:
            return original_score
            
        penalized_score = original_score * config.penalty_coefficient
        
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
            "penalty_coefficient": config.penalty_coefficient,
            "owner_default_score": config.owner_default_score,
            "last_updated": config.last_updated,
            "cache_duration": config.cache_duration,
            "config_url": self.config_url,
            "use_database": self.use_database
        }
        
    def add_to_whitelist(self, validator_hotkey: str, added_by: str = "system", reason: str = "") -> bool:
        if not self.use_database or not self.db:
            return False
            
        try:
            self.remove_from_blacklist(validator_hotkey)
            
            query = '''
                INSERT INTO validator_whitelist (validator_hotkey, added_by, reason)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE 
                    is_active = TRUE,
                    added_by = VALUES(added_by),
                    reason = VALUES(reason),
                    added_at = CURRENT_TIMESTAMP
            '''
            self.db.execute_query(query, (validator_hotkey, added_by, reason))
            
            self._cached_config = None
            
            return True
            
        except Error as e:
            logging.error(f"Error adding validator {validator_hotkey} to whitelist: {e}")
            return False
            
    def remove_from_whitelist(self, validator_hotkey: str) -> bool:
        if not self.use_database or not self.db:
            return False
            
        try:
            query = "UPDATE validator_whitelist SET is_active = FALSE WHERE validator_hotkey = %s"
            rows_affected = self.db.execute_query(query, (validator_hotkey,))
            
            self._cached_config = None
            
            if rows_affected > 0:
                logging.info(f"Removed validator {validator_hotkey} from whitelist")
                return True
            else:
                logging.warning(f"Validator {validator_hotkey} not found in whitelist")
                return False
                
        except Error as e:
            logging.error(f"Error removing validator {validator_hotkey} from whitelist: {e}")
            return False
            
    def add_to_blacklist(self, validator_hotkey: str, added_by: str = "system", reason: str = "") -> bool:
        if not self.use_database or not self.db:
            return False
            
        try:
            self.remove_from_whitelist(validator_hotkey)
            
            query = '''
                INSERT INTO validator_blacklist (validator_hotkey, added_by, reason)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE 
                    is_active = TRUE,
                    added_by = VALUES(added_by),
                    reason = VALUES(reason),
                    added_at = CURRENT_TIMESTAMP
            '''
            self.db.execute_query(query, (validator_hotkey, added_by, reason))
            
            self._cached_config = None
            
            logging.info(f"Added validator {validator_hotkey} to blacklist")
            return True
            
        except Error as e:
            logging.error(f"Error adding validator {validator_hotkey} to blacklist: {e}")
            return False
            
    def remove_from_blacklist(self, validator_hotkey: str) -> bool:
        if not self.use_database or not self.db:
            return False
            
        try:
            query = "UPDATE validator_blacklist SET is_active = FALSE WHERE validator_hotkey = %s"
            rows_affected = self.db.execute_query(query, (validator_hotkey,))
            
            self._cached_config = None
            
            if rows_affected > 0:
                logging.info(f"Removed validator {validator_hotkey} from blacklist")
                return True
            else:
                logging.warning(f"Validator {validator_hotkey} not found in blacklist")
                return False
                
        except Error as e:
            logging.error(f"Error removing validator {validator_hotkey} from blacklist: {e}")
            return False
            
    def set_penalty_coefficient(self, coefficient: float, updated_by: str = "system") -> bool:
        if not self.use_database or not self.db:
            return False
            
        try:
            self.db.set_config('penalty_coefficient', coefficient, 'number', updated_by)
            
            self._cached_config = None
            
            logging.info(f"Set penalty coefficient to {coefficient}")
            return True
            
        except Error as e:
            logging.error(f"Error setting penalty coefficient: {e}")
            return False 