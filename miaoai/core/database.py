import os
import logging
import mysql.connector
from mysql.connector import Error
from typing import Optional, Dict, List, Any
from contextlib import contextmanager
import time

from bittensor import logging as bt_logging


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, 
                 host: str = None,
                 port: int = None,
                 database: str = None,
                 username: str = None,
                 password: str = None):
        """
        初始化数据库管理器
        
        Args:
            host: 数据库主机地址
            port: 数据库端口
            database: 数据库名称
            username: 用户名
            password: 密码
        """
        self.host = host or os.getenv("DB_HOST", "localhost")
        self.port = port or int(os.getenv("DB_PORT", "3306"))
        self.database = database or os.getenv("DB_NAME", "cognify")
        self.username = username or os.getenv("DB_USER", "root")
        self.password = password or os.getenv("DB_PASSWORD", "")
        
        self.pool_name = "cognify_pool"
        self.pool_size = int(os.getenv("DB_POOL_SIZE", "5"))
        self.pool_reset_session = True
        
        self._pool = None
        self._init_pool()
        
    def _init_pool(self):
        """初始化连接池"""
        try:
            config = {
                'host': self.host,
                'port': self.port,
                'database': self.database,
                'user': self.username,
                'password': self.password,
                'pool_name': self.pool_name,
                'pool_size': self.pool_size,
                'pool_reset_session': self.pool_reset_session,
                'autocommit': True,
                'charset': 'utf8mb4',
                'collation': 'utf8mb4_unicode_ci'
            }
            
            self._pool = mysql.connector.pooling.MySQLConnectionPool(**config)
            bt_logging.info(f"Database connection pool initialized: {self.host}:{self.port}/{self.database}")
            
        except Error as e:
            bt_logging.error(f"Error creating database connection pool: {e}")
            raise
            
    @contextmanager
    def get_connection(self):
        """获取数据库连接的上下文管理器"""
        connection = None
        try:
            connection = self._pool.get_connection()
            yield connection
        except Error as e:
            bt_logging.error(f"Database connection error: {e}")
            if connection:
                connection.rollback()
            raise
        finally:
            if connection and connection.is_connected():
                connection.close()
                
    def execute_query(self, query: str, params: tuple = None, fetch: bool = False) -> Optional[List[tuple]]:
        """
        执行SQL查询
        
        Args:
            query: SQL查询语句
            params: 查询参数
            fetch: 是否获取结果
            
        Returns:
            查询结果（如果fetch=True）
        """
        with self.get_connection() as connection:
            cursor = connection.cursor()
            try:
                cursor.execute(query, params or ())
                if fetch:
                    return cursor.fetchall()
                else:
                    connection.commit()
                    return cursor.rowcount
            finally:
                cursor.close()
                
    def execute_many(self, query: str, params_list: List[tuple]) -> int:
        """
        批量执行SQL语句
        
        Args:
            query: SQL查询语句
            params_list: 参数列表
            
        Returns:
            影响的行数
        """
        with self.get_connection() as connection:
            cursor = connection.cursor()
            try:
                cursor.executemany(query, params_list)
                connection.commit()
                return cursor.rowcount
            finally:
                cursor.close()
                
    def create_tables(self):
        """创建必要的数据库表"""
        tables = {
            'validator_tokens': '''
                CREATE TABLE IF NOT EXISTS validator_tokens (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    validator_hotkey VARCHAR(255) UNIQUE NOT NULL,
                    token VARCHAR(255) UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    last_used_at TIMESTAMP NULL,
                    description TEXT,
                    INDEX idx_validator_hotkey (validator_hotkey),
                    INDEX idx_token (token),
                    INDEX idx_active (is_active)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            ''',
            
            'validator_whitelist': '''
                CREATE TABLE IF NOT EXISTS validator_whitelist (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    validator_hotkey VARCHAR(255) UNIQUE NOT NULL,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    added_by VARCHAR(255),
                    reason TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    INDEX idx_validator_hotkey (validator_hotkey),
                    INDEX idx_active (is_active)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            ''',
            
            'validator_blacklist': '''
                CREATE TABLE IF NOT EXISTS validator_blacklist (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    validator_hotkey VARCHAR(255) UNIQUE NOT NULL,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    added_by VARCHAR(255),
                    reason TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    INDEX idx_validator_hotkey (validator_hotkey),
                    INDEX idx_active (is_active)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            ''',
            
            'system_config': '''
                CREATE TABLE IF NOT EXISTS system_config (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    config_key VARCHAR(255) UNIQUE NOT NULL,
                    config_value TEXT,
                    data_type ENUM('string', 'number', 'boolean', 'json') DEFAULT 'string',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    updated_by VARCHAR(255),
                    description TEXT,
                    INDEX idx_config_key (config_key)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            ''',
            
            'api_logs': '''
                CREATE TABLE IF NOT EXISTS api_logs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    validator_hotkey VARCHAR(255),
                    token VARCHAR(255),
                    endpoint VARCHAR(255),
                    method VARCHAR(10),
                    ip_address VARCHAR(45),
                    user_agent TEXT,
                    request_data TEXT,
                    response_status INT,
                    response_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_validator_hotkey (validator_hotkey),
                    INDEX idx_created_at (created_at),
                    INDEX idx_endpoint (endpoint)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            '''
        }
        
        for table_name, create_sql in tables.items():
            try:
                self.execute_query(create_sql)
                bt_logging.info(f"Table {table_name} created/verified successfully")
            except Error as e:
                bt_logging.error(f"Error creating table {table_name}: {e}")
                raise
                
    def init_default_config(self):
        """初始化默认配置"""
        default_configs = [
            ('penalty_coefficient', '0.000000001', 'number', 'Penalty coefficient for non-whitelisted validators'),
            ('cache_duration', '1200', 'number', 'Cache duration in seconds'),
            ('max_tokens_per_validator', '1', 'number', 'Maximum tokens per validator'),
            ('token_expiry_days', '90', 'number', 'Token expiry in days'),
        ]
        
        for config_key, config_value, data_type, description in default_configs:
            query = '''
                INSERT INTO system_config (config_key, config_value, data_type, description, updated_by)
                VALUES (%s, %s, %s, %s, 'system')
                ON DUPLICATE KEY UPDATE 
                    config_value = VALUES(config_value),
                    data_type = VALUES(data_type),
                    description = VALUES(description),
                    updated_at = CURRENT_TIMESTAMP
            '''
            try:
                self.execute_query(query, (config_key, config_value, data_type, description))
            except Error as e:
                bt_logging.error(f"Error initializing config {config_key}: {e}")
                
    def get_config(self, config_key: str, default_value: Any = None) -> Any:
        """获取配置值"""
        query = "SELECT config_value, data_type FROM system_config WHERE config_key = %s"
        result = self.execute_query(query, (config_key,), fetch=True)
        
        if not result:
            return default_value
            
        config_value, data_type = result[0]
        
        # 根据数据类型转换值
        if data_type == 'number':
            try:
                return float(config_value) if '.' in config_value else int(config_value)
            except ValueError:
                return default_value
        elif data_type == 'boolean':
            return config_value.lower() in ('true', '1', 'yes', 'on')
        elif data_type == 'json':
            import json
            try:
                return json.loads(config_value)
            except json.JSONDecodeError:
                return default_value
        else:
            return config_value
            
    def set_config(self, config_key: str, config_value: Any, data_type: str = 'string', updated_by: str = 'system'):
        """设置配置值"""
        if data_type == 'json':
            import json
            config_value = json.dumps(config_value)
        else:
            config_value = str(config_value)
            
        query = '''
            INSERT INTO system_config (config_key, config_value, data_type, updated_by)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE 
                config_value = VALUES(config_value),
                data_type = VALUES(data_type),
                updated_by = VALUES(updated_by),
                updated_at = CURRENT_TIMESTAMP
        '''
        self.execute_query(query, (config_key, config_value, data_type, updated_by))
        
    def test_connection(self) -> bool:
        """测试数据库连接"""
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                cursor.close()
                return result is not None
        except Error as e:
            bt_logging.error(f"Database connection test failed: {e}")
            return False


# 全局数据库管理器实例
_db_manager = None

def get_db_manager() -> DatabaseManager:
    """获取数据库管理器单例"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
        # _db_manager.create_tables()
        # _db_manager.init_default_config()
    return _db_manager 