import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
from bittensor import logging

from miaoai.core.storage.base_storage import BaseStorage

class ValidatorStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    BLACKLISTED = "blacklisted"

@dataclass
class ValidatorInfo:
    hotkey: str
    status: ValidatorStatus
    reputation_score: float
    last_request_block: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    locked_since_block: Optional[int] = None
    lock_duration_blocks: Optional[int] = None

class ValidatorManager:

    def __init__(self, storage: BaseStorage):
        self.storage = storage
        self.current_block = 0
        
        self.inactivity_threshold = 100
        self.min_reputation_score = 0.3
        self.default_lock_duration = 1000
        self.reputation_decay_rate = 0.99
        
        self._load_state()
        
    def _load_state(self):
        try:
            state = self.storage.get("validator_state", {})
            self.validators: Dict[str, ValidatorInfo] = {}
            
            for hotkey, info in state.get("validators", {}).items():
                status_str = info.pop('status', 'active')
                info['status'] = ValidatorStatus(status_str)
                self.validators[hotkey] = ValidatorInfo(**info)
                
            self.current_locked_validator = state.get("current_locked_validator")

        except Exception as e:
            logging.error(f"Failed to load validator state: {e}")
            self.validators = {}
            self.current_locked_validator = None
            
    def _save_state(self):
        try:
            state = {
                "validators": {
                    hotkey: {
                        **{k: v for k, v in vars(info).items() if k != 'status'},
                        'status': info.status.value
                    } for hotkey, info in self.validators.items()
                },
                "current_locked_validator": self.current_locked_validator
            }
            self.storage.set("validator_state", state)
        except Exception as e:
            if "not JSON serializable" in str(e):
                logging.warning(f"Detected corrupted state file, attempting to delete and retry...")
                try:
                    self.storage.delete("validator_state")
                    state = {
                        "validators": {
                            hotkey: {
                                **{k: v for k, v in vars(info).items() if k != 'status'},
                                'status': info.status.value
                            } for hotkey, info in self.validators.items()
                        },
                        "current_locked_validator": self.current_locked_validator
                    }
                    self.storage.set("validator_state", state)
                except Exception as retry_e:
                    logging.error(f"Failed to save state even after cleanup: {retry_e}")
            else:
                logging.error(f"Failed to save validator state: {e}")
            
    def update_block(self, current_block: int):
        self.current_block = current_block
        self._check_lock_expiry()
        self._update_validator_status()
        
    def _check_lock_expiry(self):
        if not self.current_locked_validator:
            return
            
        validator = self.validators.get(self.current_locked_validator)
        if not validator:
            return
            
        if (validator.locked_since_block and 
            self.current_block - validator.locked_since_block >= validator.lock_duration_blocks):
            self._unlock_validator()
            
    def _update_validator_status(self):
        for hotkey, validator in self.validators.items():
            if (self.current_block - validator.last_request_block > self.inactivity_threshold and
                validator.status == ValidatorStatus.ACTIVE):
                validator.status = ValidatorStatus.INACTIVE
                logging.warning(f"Validator {hotkey} marked as inactive")
                
            validator.reputation_score *= self.reputation_decay_rate
            
    def can_serve_validator(self, validator_hotkey: str) -> bool:
        if validator_hotkey not in self.validators:
            self.validators[validator_hotkey] = ValidatorInfo(
                hotkey=validator_hotkey,
                status=ValidatorStatus.ACTIVE,
                reputation_score=0.5,
                last_request_block=self.current_block,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                avg_response_time=0.0
            )
            
        validator = self.validators[validator_hotkey]
        
        if validator.status == ValidatorStatus.BLACKLISTED:
            return False
            
        if validator.reputation_score < self.min_reputation_score:
            return False
            
        if (self.current_locked_validator and
            self.current_locked_validator != validator_hotkey):
            return False
            
        return True
        
    def lock_validator(self, validator_hotkey: str, duration_blocks: Optional[int] = None):
        if not self.can_serve_validator(validator_hotkey):
            return False
            
        validator = self.validators[validator_hotkey]
        validator.locked_since_block = self.current_block
        validator.lock_duration_blocks = duration_blocks or self.default_lock_duration
        self.current_locked_validator = validator_hotkey
        
        self._save_state()
        return True
        
    def _unlock_validator(self):
        if self.current_locked_validator:
            validator = self.validators[self.current_locked_validator]
            validator.locked_since_block = None
            validator.lock_duration_blocks = None
            self.current_locked_validator = None
            self._save_state()
            
    def emergency_unlock(self):
        if self.current_locked_validator:
            logging.warning(f"Emergency unlock triggered for validator {self.current_locked_validator}")
            self._unlock_validator()
            
    def update_validator_metrics(self, 
                               validator_hotkey: str, 
                               success: bool, 
                               response_time: float):

        if validator_hotkey not in self.validators:
            self.validators[validator_hotkey] = ValidatorInfo(
                hotkey=validator_hotkey,
                status=ValidatorStatus.ACTIVE,
                reputation_score=0.5,
                last_request_block=self.current_block,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                avg_response_time=0.0
            )
        
        validator = self.validators[validator_hotkey]
        
        validator.total_requests += 1
        if success:
            validator.successful_requests += 1
        else:
            validator.failed_requests += 1
            
        n = validator.total_requests
        validator.avg_response_time = (
            (validator.avg_response_time * (n-1) + response_time) / n
        )
        
        if success:
            validator.reputation_score = min(1.0, validator.reputation_score + 0.01)
        else:
            validator.reputation_score = max(0.0, validator.reputation_score - 0.05)
            
        validator.last_request_block = self.current_block
        
        self._save_state()
        
    def get_validator_info(self, validator_hotkey: str) -> Optional[Dict[str, Any]]:
        validator = self.validators.get(validator_hotkey)
        if not validator:
            return None
            
        return {
            "status": validator.status.value,
            "reputation_score": validator.reputation_score,
            "total_requests": validator.total_requests,
            "success_rate": (
                validator.successful_requests / validator.total_requests 
                if validator.total_requests > 0 else 0
            ),
            "avg_response_time": validator.avg_response_time,
            "is_locked": (
                validator_hotkey == self.current_locked_validator
            ),
            "lock_remaining_blocks": (
                validator.lock_duration_blocks - (self.current_block - validator.locked_since_block)
                if validator.locked_since_block is not None
                else 0
            )
        }

    def can_miner_get_task(self, miner_uid: str, miner_hotkey: str, synapse: Optional['TaskSynapse'] = None) -> bool:

        if not miner_uid:
            return False
        if miner_hotkey in self.storage.get("miner_blacklist", []):
            logging.warning(f"Miner {miner_uid} is blacklisted")
            return False
            
        miner_history = self.storage.get(f"miner_history_{miner_uid}", {})
        
        if synapse and synapse.miner_history:
            miner_history.update(synapse.miner_history)
            self.storage.set(f"miner_history_{miner_uid}", miner_history)
            
        enable_recent_failures = os.getenv("ENABLE_RECENT_FAILURES", "true").lower() == "true"
        recent_failures = miner_history.get("recent_failures", 0)
        if enable_recent_failures and  recent_failures >= 10:  # 如果最近连续失败10次
            logging.warning(f"Miner {miner_uid} has too many recent failures: {recent_failures}")
            return False
            
        current_task = miner_history.get("current_task")
        if isinstance(current_task, dict):
            task_id = current_task.get("task_id", "")
            if str(task_id).strip().lower() not in ["", "none", "null"]:
                logging.warning(f"Miner {miner_uid} already has an active task: {current_task}")
                return False

        return True

    def update_miner_metrics(self, 
                           miner_uid: str,
                           task_id: str,
                           success: bool, 
                           response_time: float,
                           synapse: Optional['TaskSynapse'] = None):

        miner_history = self.storage.get(f"miner_history_{miner_uid}", {})
        
        if synapse and synapse.miner_history:
            miner_history.update(synapse.miner_history)
        
        miner_history["total_requests"] = miner_history.get("total_requests", 0) + 1
        if success:
            miner_history["successful_requests"] = miner_history.get("successful_requests", 0) + 1
            miner_history["recent_failures"] = 0
        else:
            miner_history["failed_requests"] = miner_history.get("failed_requests", 0) + 1
            miner_history["recent_failures"] = miner_history.get("recent_failures", 0) + 1
            
        n = miner_history["total_requests"]
        old_avg = miner_history.get("avg_response_time", 0.0)
        miner_history["avg_response_time"] = (old_avg * (n-1) + response_time) / n
        
        if success:
            miner_history["current_task"] = None
        else:
            miner_history["current_task"] = {"task_id": task_id, "timestamp": synapse.timestamp if synapse else None}
            
        self.storage.set(f"miner_history_{miner_uid}", miner_history)
        
    def get_miner_info(self, miner_hotkey: str, miner_uid: str) -> Optional[Dict[str, Any]]:

        miner_history = self.storage.get(f"miner_history_{miner_uid}")
        if not miner_history:
            return None
            
        return {
            "total_tasks": miner_history["total_tasks"],
            "successful_tasks": miner_history["successful_tasks"],
            "success_rate": (miner_history["successful_tasks"] / miner_history["total_tasks"] 
                           if miner_history["total_tasks"] > 0 else 0),
            "avg_response_time": miner_history["avg_response_time"],
            "recent_failures": miner_history["recent_failures"],
            "current_task": miner_history["current_task"]
        } 