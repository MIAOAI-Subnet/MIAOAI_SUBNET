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

    def __init__(self):
        self.db = get_db_manager()
        
    def generate_token(self, length: int = 64) -> str:

        alphabet = string.ascii_letters + string.digits
        token = ''.join(secrets.choice(alphabet) for _ in range(length))
        return token
        
    def hash_token(self, token: str) -> str:

        return hashlib.sha256(token.encode()).hexdigest()
        
    def create_token(self, validator_hotkey: str, description: str = "", created_by: str = "system") -> Optional[str]:

        try:
            max_tokens = self.db.get_config('max_tokens_per_validator', 1)
            
            query = "SELECT COUNT(*) FROM validator_tokens WHERE validator_hotkey = %s AND is_active = TRUE"
            result = self.db.execute_query(query, (validator_hotkey,), fetch=True)
            
            if result and result[0][0] >= max_tokens:
                logging.warning(f"Validator {validator_hotkey} already has maximum number of active tokens ({max_tokens})")
                return None
                
            token = self.generate_token()
            
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
            
            expiry_days = self.db.get_config('token_expiry_days', 365)
            if expiry_days > 0:
                expiry_date = created_at + timedelta(days=expiry_days)
                if datetime.now() > expiry_date:
                    logging.warning(f"Token for validator {validator_hotkey} has expired")
                    return None
                    
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
        try:
            query = "UPDATE validator_tokens SET last_used_at = CURRENT_TIMESTAMP WHERE token = %s"
            self.db.execute_query(query, (token,))
        except Error as e:
            logging.error(f"Error updating token last used time: {e}")
            
    def revoke_token(self, token: str = None, validator_hotkey: str = None) -> bool:

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
                
                masked_token = f"{token[:8]}...{token[-8:]}" if len(token) > 16 else token
                
                tokens.append({
                    'validator_hotkey': validator_hotkey,
                    'token': masked_token,
                    'full_token': token,
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

        validator_info = self.validate_token(token)
        return validator_info['validator_hotkey'] if validator_info else None
        
    def cleanup_expired_tokens(self) -> int:

        try:
            expiry_days = self.db.get_config('token_expiry_days', 365)
            if expiry_days <= 0:
                return 0
                
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

        try:
            stats = {}
            
            query = "SELECT COUNT(*) FROM validator_tokens"
            result = self.db.execute_query(query, fetch=True)
            stats['total_tokens'] = result[0][0] if result else 0
            
            query = "SELECT COUNT(*) FROM validator_tokens WHERE is_active = TRUE"
            result = self.db.execute_query(query, fetch=True)
            stats['active_tokens'] = result[0][0] if result else 0
            
            query = "SELECT COUNT(*) FROM validator_tokens WHERE DATE(created_at) = CURDATE()"
            result = self.db.execute_query(query, fetch=True)
            stats['tokens_created_today'] = result[0][0] if result else 0
            
            query = "SELECT COUNT(*) FROM validator_tokens WHERE last_used_at > DATE_SUB(NOW(), INTERVAL 24 HOUR)"
            result = self.db.execute_query(query, fetch=True)
            stats['tokens_used_24h'] = result[0][0] if result else 0
            
            query = "SELECT COUNT(DISTINCT validator_hotkey) FROM validator_tokens WHERE is_active = TRUE"
            result = self.db.execute_query(query, fetch=True)
            stats['validators_with_tokens'] = result[0][0] if result else 0
            
            return stats
            
        except Error as e:
            logging.error(f"Error getting token stats: {e}")
            return {}
            
    def regenerate_token(self, validator_hotkey: str, description: str = "") -> Optional[str]:

        try:
            self.revoke_token(validator_hotkey=validator_hotkey)
            
            new_token = self.create_token(validator_hotkey, description)
            
            if new_token:
                logging.info(f"Regenerated token for validator {validator_hotkey}")
                
            return new_token
            
        except Exception as e:
            logging.error(f"Error regenerating token for validator {validator_hotkey}: {e}")
            return None 