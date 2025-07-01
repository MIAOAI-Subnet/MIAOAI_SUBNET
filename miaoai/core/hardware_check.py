import os
import sys
import psutil
import torch
from bittensor import logging
from pathlib import Path
from typing import Dict, Tuple
from transformers import AutoConfig

class HardwareChecker:

    RECOMMENDED_SPECS = {
        "gpu_memory": 14,
        "cpu_cores": 8,
        "ram": 16,
        "disk_space": 50,
        "cuda_version": "11.8"
    }

    @staticmethod
    def get_check_config():
        return {
            "check_cuda": os.getenv("CHECK_CUDA", "true").lower() == "true",
            "check_gpu_memory": os.getenv("CHECK_GPU_MEMORY", "true").lower() == "true",
            "check_cpu": os.getenv("CHECK_CPU", "true").lower() == "true",
            "check_ram": os.getenv("CHECK_RAM", "true").lower() == "true",
            "check_disk": os.getenv("CHECK_DISK", "true").lower() == "true",
            "check_model": os.getenv("CHECK_MODEL", "true").lower() == "true"
        }
    
    @staticmethod
    def check_hardware() -> Tuple[bool, Dict[str, str]]:
        results = {}
        passed = True
        
        check_config = HardwareChecker.get_check_config()
        
        if check_config["check_cuda"]:
            if not torch.cuda.is_available():
                results["cuda"] = "CUDA not available"
                return False, results
                
            cuda_version = torch.version.cuda
            if cuda_version is None:
                results["cuda_version"] = "CUDA version not found"
                passed = False
            elif float(cuda_version.split(".")[0]) < float(HardwareChecker.RECOMMENDED_SPECS["cuda_version"].split(".")[0]):
                results["cuda_version"] = f"CUDA version {cuda_version} is lower than recommended {HardwareChecker.RECOMMENDED_SPECS['cuda_version']}"
                passed = False
            else:
                results["cuda_version"] = f"CUDA {cuda_version} ✓"
        else:
            results["cuda"] = "CUDA check skipped"
            
        if check_config["check_gpu_memory"]:
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if gpu_memory < HardwareChecker.RECOMMENDED_SPECS["gpu_memory"]:
                    results["gpu_memory"] = f"GPU memory {gpu_memory:.1f}GB is lower than recommended {HardwareChecker.RECOMMENDED_SPECS['gpu_memory']}GB"
                    passed = False
                else:
                    results["gpu_memory"] = f"GPU Memory {gpu_memory:.1f}GB ✓"
            except Exception as e:
                results["gpu_memory"] = f"Failed to check GPU memory: {str(e)}"
                passed = False
        else:
            results["gpu_memory"] = "GPU memory check skipped"
            
        if check_config["check_cpu"]:
            cpu_cores = psutil.cpu_count(logical=False)
            if cpu_cores < HardwareChecker.RECOMMENDED_SPECS["cpu_cores"]:
                results["cpu_cores"] = f"CPU cores {cpu_cores} is lower than recommended {HardwareChecker.RECOMMENDED_SPECS['cpu_cores']}"
                passed = False
            else:
                results["cpu_cores"] = f"CPU Cores {cpu_cores} ✓"
        else:
            results["cpu_cores"] = "CPU check skipped"
            
        if check_config["check_ram"]:
            ram_gb = psutil.virtual_memory().total / 1024**3
            if ram_gb < HardwareChecker.RECOMMENDED_SPECS["ram"]:
                results["ram"] = f"RAM {ram_gb:.1f}GB is lower than recommended {HardwareChecker.RECOMMENDED_SPECS['ram']}GB"
                passed = False
            else:
                results["ram"] = f"RAM {ram_gb:.1f}GB ✓"
        else:
            results["ram"] = "RAM check skipped"
            
        if check_config["check_disk"]:
            disk = psutil.disk_usage("/")
            disk_gb = disk.free / 1024**3
            if disk_gb < HardwareChecker.RECOMMENDED_SPECS["disk_space"]:
                results["disk_space"] = f"Free disk space {disk_gb:.1f}GB is lower than recommended {HardwareChecker.RECOMMENDED_SPECS['disk_space']}GB"
                passed = False
            else:
                results["disk_space"] = f"Disk Space {disk_gb:.1f}GB ✓"
        else:
            results["disk_space"] = "Disk space check skipped"
            
        if all(value.endswith("skipped") for value in results.values()):
            passed = True
            
        return passed, results
        
    @staticmethod
    def check_model_availability(model_name: str) -> Tuple[bool, str]:

        if not HardwareChecker.get_check_config()["check_model"]:
            return True, "Model check skipped"
            
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            model_dir = cache_dir / model_name
            
            if model_dir.exists():
                return True, f"Model {model_name} is available ✓"
            else:
                return False, f"Model {model_name} is not installed"
                
        except Exception as e:
            return False, f"Error checking model {model_name}: {str(e)}" 