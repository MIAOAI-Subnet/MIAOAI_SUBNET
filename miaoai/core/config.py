from dataclasses import dataclass
from typing import Optional

@dataclass
class MiaoAIConfig:
    stake_weight_ratio: float = 0.2
    min_blocks_per_validator: int = 10
    eval_interval: int = 25
    weights_interval: int = 100
    
    model_name: str = "Qwen/Qwen3-32B"
    max_sequence_length: int = 512
    batch_size: int = 32
    response_timeout: float = 12.0
    
    min_success_rate: float = 0.8
    max_response_time: float = 2.0
    quality_threshold: float = 0.7
    
    base_reward_rate: float = 1.0
    quality_bonus_ratio: float = 0.7
    history_bonus_ratio: float = 0.1