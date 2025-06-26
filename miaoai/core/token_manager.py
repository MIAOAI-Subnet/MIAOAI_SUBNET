import secrets
import string
import hashlib
import time
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from mysql.connector import Error

from bittensor import logging
from miaoai.core.database import get_db_manager


class TokenManager:
    """验证者Token管理器"""
    
    def __init__(self):
        self.db = get_db_manager()
        
    def generate_token(self, length: int = 64) -> str:
        """
        生成安全的token
        
        Args:
            length: token长度
            
        Returns:
            生成的token字符串
        """
        alphabet = string.ascii_letters + string.digits
        token = ''.join(secrets.choice(alphabet) for _ in range(length))
        return token
        
    def hash_token(self, token: str) -> str:
        """
        对token进行哈希处理
        
        Args:
            token: 原始token
            
        Returns:
            哈希后的token
        """
        return hashlib.sha256(token.encode()).hexdigest()
        
    def create_token(self, validator_hotkey: str, description: str = "", created_by: str = "system") -> Optional[str]:
        """
        为验证者创建新token
        
        Args:
            validator_hotkey: 验证者热键
            description: token描述
            created_by: 创建者
            
        Returns:
            生成的token（原始），如果失败返回None
        """
        try:
            # 检查该验证者是否已有活跃token
            max_tokens = self.db.get_config('max_tokens_per_validator', 1)
            
            query = "SELECT COUNT(*) FROM validator_tokens WHERE validator_hotkey = %s AND is_active = TRUE"
            result = self.db.execute_query(query, (validator_hotkey,), fetch=True)
            
            if result and result[0][0] >= max_tokens:
                logging.warning(f"Validator {validator_hotkey} already has maximum number of active tokens ({max_tokens})")
                return None
                
            # 生成新token
            token = self.generate_token()
            
            # 存储到数据库
            query = '''
                INSERT INTO validator_tokens (validator_hotkey, token, description)
                VALUES (%s, %s, %s)
            '''
            self.db.execute_query(query, (validator_hotkey, token, description))
            
            logging.info(f"Created new token for validator {validator_hotkey}")
            return token
            
        except Error as e:
            logging.error(f"Error creating token for validator {validator_hotkey}: {e}")
            return None
            
    def validate_token(self, token: str) -> Optional[Dict]:
        """
        验证token并返回验证者信息
        
        Args:
            token: 要验证的token
            
        Returns:
            验证者信息字典，如果验证失败返回None
        """
        try:
            query = '''
                SELECT validator_hotkey, created_at, last_used_at, description
                FROM validator_tokens
                WHERE token = %s AND is_active = TRUE
            '''
            result = self.db.execute_query(query, (token,), fetch=True)
            
            if not result:
                return None
                
            validator_hotkey, created_at, last_used_at, description = result[0]
            
            # 检查token是否过期
            expiry_days = self.db.get_config('token_expiry_days', 365)
            if expiry_days > 0:
                expiry_date = created_at + timedelta(days=expiry_days)
                if datetime.now() > expiry_date:
                    logging.warning(f"Token for validator {validator_hotkey} has expired")
                    return None
                    
            # 更新最后使用时间
            self._update_last_used(token)
            
            return {
                'validator_hotkey': validator_hotkey,
                'created_at': created_at,
                'last_used_at': last_used_at,
                'description': description
            }
            
        except Error as e:
            logging.error(f"Error validating token: {e}")
            return None
            
    def _update_last_used(self, token: str):
        """更新token最后使用时间"""
        try:
            query = "UPDATE validator_tokens SET last_used_at = CURRENT_TIMESTAMP WHERE token = %s"
            self.db.execute_query(query, (token,))
        except Error as e:
            logging.error(f"Error updating token last used time: {e}")
            
    def revoke_token(self, token: str = None, validator_hotkey: str = None) -> bool:
        """
        撤销token
        
        Args:
            token: 要撤销的token
            validator_hotkey: 验证者热键（撤销该验证者的所有token）
            
        Returns:
            是否成功撤销
        """
        try:
            if token:
                query = "UPDATE validator_tokens SET is_active = FALSE WHERE token = %s"
                params = (token,)
            elif validator_hotkey:
                query = "UPDATE validator_tokens SET is_active = FALSE WHERE validator_hotkey = %s"
                params = (validator_hotkey,)
            else:
                logging.error("Either token or validator_hotkey must be provided")
                return False
                
            rows_affected = self.db.execute_query(query, params)
            
            if rows_affected > 0:
                target = token or f"validator {validator_hotkey}"
                logging.info(f"Revoked token(s) for {target}")
                return True
            else:
                logging.warning(f"No active tokens found to revoke")
                return False
                
        except Error as e:
            logging.error(f"Error revoking token: {e}")
            return False
            
    def list_tokens(self, validator_hotkey: str = None, active_only: bool = True) -> List[Dict]:
        """
        列出tokens
        
        Args:
            validator_hotkey: 特定验证者的热键，如果为None则列出所有
            active_only: 是否只显示活跃的token
            
        Returns:
            token信息列表
        """
        try:
            base_query = '''
                SELECT validator_hotkey, token, created_at, last_used_at, is_active, description
                FROM validator_tokens
            '''
            
            conditions = []
            params = []
            
            if validator_hotkey:
                conditions.append("validator_hotkey = %s")
                params.append(validator_hotkey)
                
            if active_only:
                conditions.append("is_active = TRUE")
                
            if conditions:
                query = base_query + " WHERE " + " AND ".join(conditions)
            else:
                query = base_query
                
            query += " ORDER BY created_at DESC"
            
            result = self.db.execute_query(query, tuple(params), fetch=True)
            
            tokens = []
            for row in result:
                validator_hotkey, token, created_at, last_used_at, is_active, description = row
                
                # 为了安全，只显示token的前8位和后8位
                masked_token = f"{token[:8]}...{token[-8:]}" if len(token) > 16 else token
                
                tokens.append({
                    'validator_hotkey': validator_hotkey,
                    'token': masked_token,
                    'full_token': token,  # 完整token，仅用于内部操作
                    'created_at': created_at,
                    'last_used_at': last_used_at,
                    'is_active': bool(is_active),
                    'description': description
                })
                
            return tokens
            
        except Error as e:
            logging.error(f"Error listing tokens: {e}")
            return []
            
    def get_validator_by_token(self, token: str) -> Optional[str]:
        """
        根据token获取验证者热键
        
        Args:
            token: token字符串
            
        Returns:
            验证者热键，如果没找到返回None
        """
        validator_info = self.validate_token(token)
        return validator_info['validator_hotkey'] if validator_info else None
        
    def cleanup_expired_tokens(self) -> int:
        """
        清理过期的token
        
        Returns:
            清理的token数量
        """
        try:
            expiry_days = self.db.get_config('token_expiry_days', 365)
            if expiry_days <= 0:
                return 0  # 永不过期
                
            query = '''
                UPDATE validator_tokens 
                SET is_active = FALSE 
                WHERE is_active = TRUE 
                AND created_at < DATE_SUB(NOW(), INTERVAL %s DAY)
            '''
            rows_affected = self.db.execute_query(query, (expiry_days,))
            
            if rows_affected > 0:
                logging.info(f"Cleaned up {rows_affected} expired tokens")
                
            return rows_affected
            
        except Error as e:
            logging.error(f"Error cleaning up expired tokens: {e}")
            return 0
            
    def get_token_stats(self) -> Dict:
        """
        获取token统计信息
        
        Returns:
            统计信息字典
        """
        try:
            stats = {}
            
            # 总token数
            query = "SELECT COUNT(*) FROM validator_tokens"
            result = self.db.execute_query(query, fetch=True)
            stats['total_tokens'] = result[0][0] if result else 0
            
            # 活跃token数
            query = "SELECT COUNT(*) FROM validator_tokens WHERE is_active = TRUE"
            result = self.db.execute_query(query, fetch=True)
            stats['active_tokens'] = result[0][0] if result else 0
            
            # 今天创建的token数
            query = "SELECT COUNT(*) FROM validator_tokens WHERE DATE(created_at) = CURDATE()"
            result = self.db.execute_query(query, fetch=True)
            stats['tokens_created_today'] = result[0][0] if result else 0
            
            # 最近使用的token数（最近24小时）
            query = "SELECT COUNT(*) FROM validator_tokens WHERE last_used_at > DATE_SUB(NOW(), INTERVAL 24 HOUR)"
            result = self.db.execute_query(query, fetch=True)
            stats['tokens_used_24h'] = result[0][0] if result else 0
            
            # 有token的验证者数量
            query = "SELECT COUNT(DISTINCT validator_hotkey) FROM validator_tokens WHERE is_active = TRUE"
            result = self.db.execute_query(query, fetch=True)
            stats['validators_with_tokens'] = result[0][0] if result else 0
            
            return stats
            
        except Error as e:
            logging.error(f"Error getting token stats: {e}")
            return {}
            
    def regenerate_token(self, validator_hotkey: str, description: str = "") -> Optional[str]:
        """
        重新生成验证者的token（撤销旧的，创建新的）
        
        Args:
            validator_hotkey: 验证者热键
            description: 新token的描述
            
        Returns:
            新生成的token，如果失败返回None
        """
        try:
            # 撤销旧token
            self.revoke_token(validator_hotkey=validator_hotkey)
            
            # 创建新token
            new_token = self.create_token(validator_hotkey, description)
            
            if new_token:
                logging.info(f"Regenerated token for validator {validator_hotkey}")
                
            return new_token
            
        except Exception as e:
            logging.error(f"Error regenerating token for validator {validator_hotkey}: {e}")
            return None 