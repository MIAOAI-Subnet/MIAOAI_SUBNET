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
    """验证者列表配置"""
    whitelist: List[str]
    blacklist: List[str]
    penalty_coefficient: float  # 不在白名单的验证者的惩罚系数
    owner_default_score: float
    last_updated: int  # 最后更新时间戳
    cache_duration: int = 300  # 缓存持续时间（秒），默认5分钟


class ValidatorWhitelistManager:
    """验证者白名单管理器"""
    
    def __init__(self, config_url: Optional[str] = None, validator_token: Optional[str] = None, cache_file: Optional[str] = None, use_database: bool = True, hotkey: Optional[str] = None):
        """
        初始化白名单管理器
        
        Args:
            config_url: 服务端配置URL
            cache_file: 本地缓存文件路径
            use_database: 是否使用数据库存储
        """
        self.config_url = config_url or os.getenv("VALIDATOR_CONFIG_URL", "http://206.233.201.2:5000/config")
        self.validator_token = validator_token or os.getenv("VALIDATOR_TOKEN", "")
        self.hotkey = hotkey
        self.use_database = use_database

        
        # 如果使用数据库，初始化数据库连接
        if self.use_database:
            self.db = get_db_manager()
        else:
            self.db = None
        
        # 设置缓存文件路径
        if cache_file:
            self.cache_file = Path(cache_file)
        else:
            project_root = PathUtils.get_project_root()
            self.cache_file = project_root / "data" / "validator_config.json"
            
        # 确保缓存目录存在
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 默认配置
        self.default_config = ValidatorListConfig(
            whitelist=[],
            blacklist=[],
            penalty_coefficient= DEFAULT_PENALTY_COEFFICIENT,  # 默认惩罚系数1e-9
            owner_default_score = OWNER_DEFAULT_SCORE,
            last_updated=0,
            cache_duration=1200
        )
        
        self._cached_config: Optional[ValidatorListConfig] = None
        
    def _load_from_cache(self) -> Optional[ValidatorListConfig]:
        """从缓存文件加载配置"""
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
            # logging.error(f"Failed to load validator config from cache: {e}")
            return None
            
    def _save_to_cache(self, config: ValidatorListConfig) -> None:
        """保存配置到缓存文件"""
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
        """从服务端获取配置"""
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
            
            # 保存到缓存
            self._save_to_cache(config)
            
            # logging.info(f"Successfully fetched validator config from server: "
            #             f"{len(config.whitelist)} whitelist, "
            #             f"{len(config.blacklist)} blacklist, "
            #             f"penalty coefficient: {config.penalty_coefficient}")
            
            return config
            
        except Exception as e:
            # logging.error(f"Failed to fetch validator config from server: {e}")
            return None
            
    def _load_from_database(self) -> Optional[ValidatorListConfig]:
        """从数据库加载配置"""
        if not self.use_database or not self.db:
            return None
            
        try:
            # 获取白名单
            whitelist_query = "SELECT validator_hotkey FROM validator_whitelist WHERE is_active = TRUE"
            whitelist_result = self.db.execute_query(whitelist_query, fetch=True)
            whitelist = [row[0] for row in whitelist_result] if whitelist_result else []
            
            # 获取黑名单
            blacklist_query = "SELECT validator_hotkey FROM validator_blacklist WHERE is_active = TRUE"
            blacklist_result = self.db.execute_query(blacklist_query, fetch=True)
            blacklist = [row[0] for row in blacklist_result] if blacklist_result else []
            
            # 获取系统配置
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
        """
        获取验证者配置
        
        Args:
            force_refresh: 是否强制从服务端刷新
            
        Returns:
            ValidatorListConfig: 验证者配置
        """
        current_time = int(time.time())
        
        # 如果有缓存且未过期，直接返回
        if (not force_refresh and 
            self._cached_config and 
            current_time - self._cached_config.last_updated < self._cached_config.cache_duration):
            return self._cached_config
            
        # 首先尝试从数据库获取
        if self.use_database:
            config = self._load_from_database()
            if config:
                self._cached_config = config
                return config
            
        # 尝试从服务端获取
        config = self._fetch_from_server()
        
        # 如果服务端获取失败，从缓存加载
        if not config:
            config = self._load_from_cache()
            
        # 如果缓存也没有，使用默认配置
        if not config:
            logging.warning("Using default validator config")
            config = self.default_config
            
        self._cached_config = config
        return config
        
    def is_validator_whitelisted(self, validator_hotkey: str) -> bool:
        """检查验证者是否在白名单中"""
        config = self.get_config()
        return validator_hotkey in config.whitelist
        
    def is_validator_blacklisted(self, validator_hotkey: str) -> bool:
        """检查验证者是否在黑名单中"""
        config = self.get_config()
        return validator_hotkey in config.blacklist
        
    def get_penalty_coefficient(self) -> float:
        """获取惩罚系数"""
        config = self.get_config()
        return config.penalty_coefficient
        
    def get_filtered_validators(self, all_validators: List[str]) -> List[str]:
        """
        过滤掉黑名单中的验证者
        
        Args:
            all_validators: 所有验证者列表
            
        Returns:
            List[str]: 过滤后的验证者列表
        """
        config = self.get_config()
        blacklist_set = set(config.blacklist)
        
        filtered = [v for v in all_validators if v not in blacklist_set]
        
        # if len(filtered) != len(all_validators):
            # logging.info(f"Filtered out {len(all_validators) - len(filtered)} blacklisted validators")
            
        return filtered
        
    def apply_whitelist_penalty(self, validator_hotkey: str, original_score: float) -> float:
        """
        应用白名单惩罚
        
        Args:
            validator_hotkey: 验证者热键
            original_score: 原始分数
            
        Returns:
            float: 调整后的分数
        """
        config = self.get_config()
        
        # 如果在黑名单中，返回0分
        if validator_hotkey in config.blacklist:
            return 0.0
            
        # 如果在白名单中，返回原始分数
        if validator_hotkey in config.whitelist:
            return original_score
            
        # 如果不在白名单中，应用惩罚系数
        penalized_score = original_score * config.penalty_coefficient
        
        # logging.debug(f"Applied penalty to validator {validator_hotkey}: "
        #              f"{original_score:.4f} -> {penalized_score:.4f} "
        #              f"(coefficient: {config.penalty_coefficient})")
        
        return penalized_score
        
    def refresh_config(self) -> bool:
        """刷新配置"""
        try:
            self.get_config(force_refresh=True)
            return True
        except Exception as e:
            logging.error(f"Failed to refresh validator config: {e}")
            return False
            
    def get_stats(self) -> Dict:
        """获取统计信息"""
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
        """添加验证者到白名单"""
        if not self.use_database or not self.db:
            return False
            
        try:
            # 从黑名单中移除（如果存在）
            self.remove_from_blacklist(validator_hotkey)
            
            # 添加到白名单
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
            
            # 清除缓存
            self._cached_config = None
            
            # logging.info(f"Added validator {validator_hotkey} to whitelist")
            return True
            
        except Error as e:
            logging.error(f"Error adding validator {validator_hotkey} to whitelist: {e}")
            return False
            
    def remove_from_whitelist(self, validator_hotkey: str) -> bool:
        """从白名单移除验证者"""
        if not self.use_database or not self.db:
            return False
            
        try:
            query = "UPDATE validator_whitelist SET is_active = FALSE WHERE validator_hotkey = %s"
            rows_affected = self.db.execute_query(query, (validator_hotkey,))
            
            # 清除缓存
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
        """添加验证者到黑名单"""
        if not self.use_database or not self.db:
            return False
            
        try:
            # 从白名单中移除（如果存在）
            self.remove_from_whitelist(validator_hotkey)
            
            # 添加到黑名单
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
            
            # 清除缓存
            self._cached_config = None
            
            logging.info(f"Added validator {validator_hotkey} to blacklist")
            return True
            
        except Error as e:
            logging.error(f"Error adding validator {validator_hotkey} to blacklist: {e}")
            return False
            
    def remove_from_blacklist(self, validator_hotkey: str) -> bool:
        """从黑名单移除验证者"""
        if not self.use_database or not self.db:
            return False
            
        try:
            query = "UPDATE validator_blacklist SET is_active = FALSE WHERE validator_hotkey = %s"
            rows_affected = self.db.execute_query(query, (validator_hotkey,))
            
            # 清除缓存
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
        """设置惩罚系数"""
        if not self.use_database or not self.db:
            return False
            
        try:
            self.db.set_config('penalty_coefficient', coefficient, 'number', updated_by)
            
            # 清除缓存
            self._cached_config = None
            
            logging.info(f"Set penalty coefficient to {coefficient}")
            return True
            
        except Error as e:
            logging.error(f"Error setting penalty coefficient: {e}")
            return False 