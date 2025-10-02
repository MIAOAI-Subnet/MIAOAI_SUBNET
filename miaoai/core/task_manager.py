import os
import random
import requests
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from bittensor import logging
import ijson

from miaoai.core.storage.base_storage import BaseStorage
from miaoai.core.constants import DEFAULT_TASK_POOL_SIZE, MAX_TASK_POOL_SIZE, MAX_VALIDATOR_BLOCKS

@dataclass
class TaskData:
    task_id: str
    text: str
    metadata: Dict[str, Any]
    ground_truth: Optional[float] = None
    difficulty: float = 1.0
    completed: bool = False

class TaskManager:

    def __init__(self, storage: BaseStorage, task_data_path: str, max_tasks: int = DEFAULT_TASK_POOL_SIZE, 
                 central_server_url: str = None, auth_token: str = None):
        self.storage = storage
        self.task_data_path = task_data_path
        self.max_tasks = max_tasks
        self.central_server_url = central_server_url or os.getenv("CENTRAL_SERVER_URL", "http://206.233.201.2:5000")
        self.auth_token = auth_token or os.getenv("VALIDATOR_TOKEN", "")
        self.use_central_server = os.getenv("USE_CENTRAL_SERVER", "true").lower() == "true"
        
        self.task_pool: Dict[str, TaskData] = {}
        self.assigned_tasks: Dict[str, Dict[str, List[str]]] = {}  # {miner_hotkey: {validator_hotkey: [task_ids]}}
        
        self.file_positions: Dict[str, int] = {}  # {file_path: current_position}
        self.used_task_ids: set = set()
        self.file_task_counts: Dict[str, int] = {}
        self.total_blocks_run: int = 0
        
        self._load_state()
        self._load_tasks()
        
    def _count_tasks_in_file(self, file_path: str) -> int:
        try:
            if file_path in self.file_task_counts:
                return self.file_task_counts[file_path]
                
            count = 0
            with open(file_path, 'r',  encoding="utf-8") as f:
                parser = ijson.parse(f)
                for prefix, event, value in parser:
                    if prefix.endswith('.task_id'):
                        count += 1
                        
            self.file_task_counts[file_path] = count
            return count
        except Exception as e:
            logging.error(f"Error counting tasks in file {file_path}: {e}")
            return 0
        
    def _load_tasks(self):

        try:
            check_max_blocks = os.getenv("CHECK_MAX_BLOCKS", "false").lower() == "true"
            
            self.task_pool.clear()
            self.reset_all_assignments()
            
            if self.use_central_server:
                self._load_tasks_from_central_server(check_max_blocks)
            else:
                if not os.path.exists(self.task_data_path):
                    logging.warning(f"Task data path not found: {self.task_data_path}")
                    return
                    
                if os.path.isfile(self.task_data_path):
                    self._load_task_file(self.task_data_path, check_max_blocks)
                elif os.path.isdir(self.task_data_path):
                    for filename in os.listdir(self.task_data_path):
                        if filename.endswith('.json'):
                            file_path = os.path.join(self.task_data_path, filename)
                            if not check_max_blocks and len(self.task_pool) >= self.max_tasks:
                                break
                            self._load_task_file(file_path, check_max_blocks)
                        
            logging.info(f"Loaded {len(self.task_pool)} tasks from {'central server' if self.use_central_server else self.task_data_path}")
            
        except Exception as e:
            logging.error(f"Error loading task data: {e}")
            
    def _load_tasks_from_central_server(self, check_max_blocks: bool):
        try:
            remaining_slots = self.max_tasks - len(self.task_pool)
            if remaining_slots <= 0:
                return
                
            params = {
                'limit': max(remaining_slots, 200),
                'version': 'v1.0.4'
            }
            
            headers = {}
            if self.auth_token:
                headers['Authorization'] = f'Bearer {self.auth_token}'
            
            response = requests.get(
                f"{self.central_server_url}/tasks",
                params=params,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                tasks_data = data.get('tasks', [])
                
                for task_data in tasks_data:
                    if len(self.task_pool) >= self.max_tasks:
                        break
                        
                    task = TaskData(
                        task_id=task_data['task_id'],
                        text=task_data['text'],
                        metadata=task_data.get('metadata', {}),
                        ground_truth=task_data.get('ground_truth'),
                        difficulty=task_data.get('difficulty', 1.0),
                        completed=task_data.get('completed', False)
                    )
                    self.task_pool[task.task_id] = task
                
                self._save_state()
                
            else:
                logging.error(f"Failed to get tasks from central server: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            logging.error(f"Network error when loading tasks from central server: {e}")
        except Exception as e:
            logging.error(f"Error loading tasks from central server: {e}")
            
    def _load_task_file(self, file_path: str, check_max_blocks: bool):
        try:
            total_tasks = self._count_tasks_in_file(file_path)
            if total_tasks == 0:
                return
                
            current_position = self.file_positions.get(file_path, 0)
            
            if current_position >= total_tasks:
                current_position = 0
                
            remaining_slots = self.max_tasks - len(self.task_pool)
            if remaining_slots <= 0:
                return
                
            tasks_loaded = 0
            skipped_tasks = 0
            
            with open(file_path, 'r',  encoding="utf-8") as f:
                parser = ijson.items(f, 'item')
                
                for task in parser:
                    if skipped_tasks < current_position:
                        skipped_tasks += 1
                        continue
                        
                    if tasks_loaded >= remaining_slots:
                        break
                        
                    if not check_max_blocks and task['task_id'] in self.used_task_ids:
                        continue
                        
                    task_data = TaskData(
                        task_id=task['task_id'],
                        text=task['text'],
                        metadata=task.get('metadata', {}),
                        ground_truth=task.get('ground_truth'),
                        difficulty=task.get('difficulty', 1.0)
                    )
                    self.task_pool[task_data.task_id] = task_data
                    tasks_loaded += 1

            new_position = current_position + tasks_loaded
            if new_position >= total_tasks:
                new_position = 0

            self.file_positions[file_path] = new_position

            self._save_state()
                
        except Exception as e:
            logging.error(f"Error loading task file {file_path}: {e}")
            
    def _load_state(self):
        try:
            state = self.storage.get("task_state", {})
            self.assigned_tasks = state.get("assigned_tasks", {})
            self.file_positions = state.get("file_positions", {})
            self.used_task_ids = set(state.get("used_task_ids", []))
            self.file_task_counts = state.get("file_task_counts", {})
            self.total_blocks_run = state.get("total_blocks_run", 0)
            
            completed_tasks = state.get("completed_tasks", {})
            for task_id, completed in completed_tasks.items():
                if task_id in self.task_pool:
                    self.task_pool[task_id].completed = completed
                    
        except Exception as e:
            logging.error(f"Failed to load task state: {e}")
            
    def _save_state(self):
        try:
            completed_tasks = {
                task_id: task.completed 
                for task_id, task in self.task_pool.items()
            }
            
            state = {
                "assigned_tasks": self.assigned_tasks,
                "completed_tasks": completed_tasks,
                "file_positions": self.file_positions,
                "used_task_ids": list(self.used_task_ids),
                "file_task_counts": self.file_task_counts,
                "total_blocks_run": self.total_blocks_run
            }
            self.storage.set("task_state", state)
        except Exception as e:
            logging.error(f"Failed to save task state: {e}")
            
    def update_blocks_run(self, blocks: int):
        self.total_blocks_run = blocks
        self._save_state()
            
    def reset_all_assignments(self):
        self.assigned_tasks.clear()
        self._save_state()
        logging.info("Reset all task assignments")
            
    def get_task_for_miner(self, miner_hotkey: str, validator_hotkey: str) -> Optional[TaskData]:
        try:
            check_max_blocks = os.getenv("CHECK_MAX_BLOCKS", "false").lower() == "true"
            if check_max_blocks and self.total_blocks_run >= MAX_VALIDATOR_BLOCKS:
                return None
            
            available_tasks = [
                task for task in self.task_pool.values()
                if not task.completed
            ]
            
            if not available_tasks:
                self.task_pool.clear()
                
                if not check_max_blocks:
                    self.reset_all_assignments()
                    state = self.storage.get("task_state", {})
                    state["completed_tasks"] = {}
                    self.storage.set("task_state", state)
                    
                self._load_tasks()
                
                available_tasks = [
                    task for task in self.task_pool.values()
                    if not task.completed
                ]
                if not available_tasks:
                    logging.warning("No available tasks after reload")
                return None
                
            miner_tasks = self.assigned_tasks.get(miner_hotkey, {})
            validator_tasks = miner_tasks.get(validator_hotkey, [])
            
            unassigned_tasks = [
                task for task in available_tasks
                if task.task_id not in validator_tasks
            ]
            
            if not unassigned_tasks:
                return None
                
            task = random.choice(unassigned_tasks)
            
            logging.info(f"get_task_for_miner task: {task}")
            if miner_hotkey not in self.assigned_tasks:
                self.assigned_tasks[miner_hotkey] = {}
            if validator_hotkey not in self.assigned_tasks[miner_hotkey]:
                self.assigned_tasks[miner_hotkey][validator_hotkey] = []
                
            self.assigned_tasks[miner_hotkey][validator_hotkey].append(task.task_id)
            
            if not check_max_blocks:
                self.used_task_ids.add(task.task_id)
            
            self._save_state()
            return task
            
        except Exception as e:
            logging.error(f"Error getting task for miner: {e}")
            return None
            
    def mark_task_completed(self, task_id: str, success: bool = True):
        if task_id in self.task_pool:
            self.task_pool[task_id].completed = success
            self._save_state()
            
    def get_task_stats(self) -> Dict[str, Any]:
        total_tasks = len(self.task_pool)
        completed_tasks = len([t for t in self.task_pool.values() if t.completed])
        
        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "completion_rate": completed_tasks / total_tasks if total_tasks > 0 else 0,
            "assigned_miners": len(self.assigned_tasks),
            "used_tasks": len(self.used_task_ids)
        }
        
    def reset_miner_tasks(self, miner_hotkey: str):
        if miner_hotkey in self.assigned_tasks:
            del self.assigned_tasks[miner_hotkey]
            self._save_state() 