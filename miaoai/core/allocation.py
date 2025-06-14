from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import bittensor as bt
from dataclasses import dataclass

@dataclass
class TaskAllocation:
    validator_hotkey: str
    miner_hotkey:str
    blocks_allocated: int
    proportion: float = 1.0

class BaseAllocationStrategy(ABC):

    def __init__(self, min_blocks: int = 5):
        if min_blocks:
            self.min_blocks = min_blocks
        else:
            self.min_blocks = 5
    
    @abstractmethod
    def allocate(
        self,
        available_blocks: int,
        validators: List[str],
        metagraph: "bt.MetagraphInfo"
    ) -> List[TaskAllocation]:
        pass

class StakeBasedAllocation(BaseAllocationStrategy):

    def allocate(
        self,
        available_blocks: int,
        validators: List[str],
        metagraph: "bt.MetagraphInfo"
    ) -> List[TaskAllocation]:
        if not validators:
            return []
            
        total_stake = sum(
            float(metagraph.total_stake[metagraph.hotkeys.index(v)].rao)
            for v in validators
        )
        
        allocations = []
        remaining_blocks = available_blocks
        
        sorted_validators = sorted(
            validators,
            key=lambda v: metagraph.total_stake[metagraph.hotkeys.index(v)],
            reverse=True
        )
        
        for validator in sorted_validators:
            if remaining_blocks < self.min_blocks:
                break
                
            stake = float(metagraph.total_stake[metagraph.hotkeys.index(validator)].rao)
            fair_share = int((stake / total_stake) * available_blocks)
            
            blocks = max(fair_share, self.min_blocks)
            blocks = min(blocks, remaining_blocks)

            uid = metagraph.hotkeys.index(validator)

            allocations.append(
                TaskAllocation(
                    validator_hotkey=validator,
                    miner_hotkey=validator,
                    blocks_allocated=blocks
                )
            )
            
            remaining_blocks -= blocks
            
        return allocations

class EqualDistributionAllocation(BaseAllocationStrategy):

    def allocate(
        self,
        available_blocks: int,
        validators: List[str],
        metagraph: "bt.MetagraphInfo"
    ) -> List[TaskAllocation]:
        if not validators:
            return []
            
        base_blocks = available_blocks // len(validators)
        remainder = available_blocks % len(validators)
        
        allocations = []
        
        for i, validator in enumerate(validators):
            blocks = base_blocks + (1 if i < remainder else 0)
            
            if blocks < self.min_blocks:
                continue
                
            allocations.append(
                TaskAllocation(
                    validator_hotkey=validator,
                    miner_hotkey=validator,
                    blocks_allocated=blocks
                )
            )
            
        return allocations

class AllocationManager:

    def __init__(self, config):
        self.config = config
        self.stake_based = StakeBasedAllocation(config.min_blocks_per_validator)
        self.equal_distribution = EqualDistributionAllocation(config.min_blocks_per_validator)
        
    def allocate_tasks(
        self,
        strategy: str,
        available_blocks: int,
        validators: List[str],
        metagraph: "bt.MetagraphInfo"
    ) -> List[TaskAllocation]:
        if strategy == "stake":
            return self.stake_based.allocate(available_blocks, validators, metagraph)
        elif strategy == "equal":
            return self.equal_distribution.allocate(available_blocks, validators, metagraph)
        else:
            raise ValueError(f"Unknown allocation strategy: {strategy}")
            
    def get_validator_allocation(
        self,
        validator_hotkey: str,
        allocations: List[TaskAllocation]
    ) -> Optional[TaskAllocation]:
        for allocation in allocations:
            if allocation.validator_hotkey == validator_hotkey:
                return allocation
        return None 