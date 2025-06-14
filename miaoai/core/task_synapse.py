# MIT License
import typing
import bittensor as bt

class TaskSynapse(bt.Synapse):
    task_id: typing.Optional[str] = None
    task_text: typing.Optional[str] = None
    task_metadata: typing.Optional[dict] = None
    blocks_allocated: typing.Optional[int] = None
    signature: typing.Optional[str] = None
    
    response: typing.Optional[dict] = None
    response_time: typing.Optional[float] = None
    success: typing.Optional[bool] = None
    error_message: typing.Optional[str] = None
    
    miner_hotkey: typing.Optional[str] = None
    validator_hotkey: typing.Optional[str] = None
    timestamp: typing.Optional[int] = None
    
    miner_uid: typing.Optional[str] = None
    miner_history: typing.Optional[dict] = None
    total_tasks_completed: typing.Optional[int] = None
    recent_failures: typing.Optional[int] = None
    current_task: typing.Optional[dict] = None
    
    def deserialize(self) -> dict:
        return {
            'task_id': self.task_id,
            'task_text': self.task_text,
            'task_metadata': self.task_metadata,
            'blocks_allocated': self.blocks_allocated,
            
            'response': self.response,
            'response_time': self.response_time,
            'success': self.success,
            'error_message': self.error_message,
            
            'signature': self.signature,
            'miner_hotkey': self.miner_hotkey,
            'validator_hotkey': self.validator_hotkey,
            'timestamp': self.timestamp,
            
            # 状态同步字段
            'miner_uid': self.miner_uid,
            'miner_history': self.miner_history,
            'total_tasks_completed': self.total_tasks_completed,
            'recent_failures': self.recent_failures,
            'current_task': self.current_task,
        } 