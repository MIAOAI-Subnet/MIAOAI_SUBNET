from pathlib import Path
import os

class PathUtils:

    @staticmethod
    def get_project_root() -> Path:

        return Path(__file__).parent.parent.parent
        
    @staticmethod
    def get_task_data_path(task_data_path: str = None) -> Path:

        root_dir = PathUtils.get_project_root()
        
        if task_data_path is None:
            task_data_path = os.getenv("TASK_DATA_PATH", "tasks/ecommerce_tasks.json")
            
        if os.path.isabs(task_data_path):
            try:
                task_data_path = str(Path(task_data_path).relative_to(root_dir))
            except ValueError:
                task_data_path = "tasks/ecommerce_tasks.json"
                
        return root_dir / task_data_path
        
    @staticmethod
    def get_env_file_path(env_type: str = None) -> Path:
        root_dir = PathUtils.get_project_root()
        if env_type:
            return root_dir / f".env.{env_type}"
        return root_dir / ".env" 