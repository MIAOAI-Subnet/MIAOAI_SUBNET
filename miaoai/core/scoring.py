import numpy as np
from dataclasses import dataclass
from typing import Dict, List
from miaoai.core.config import MiaoAIConfig

@dataclass
class ValidatorPerformance:
    stake_weight: float
    historical_score: float
    expected_reward_rate: float
    blocks_allocated: int
    success_rate: float
    response_time: float
    quality_score: float

class ScoringSystem:
    def __init__(self, config: MiaoAIConfig):
        self.config = config
        self.performance_history: Dict[str, List[ValidatorPerformance]] = {}
        self.current_cycle_scores: Dict[str, List[float]] = {}

    def update_score(self, hotkey: str, performance: ValidatorPerformance) -> None:
        if hotkey not in self.performance_history:
            self.performance_history[hotkey] = []
        self.performance_history[hotkey].append(performance)

        if len(self.performance_history[hotkey]) > 1000:
            self.performance_history[hotkey] = self.performance_history[hotkey][-1000:]

    def get_historical_score(self, hotkey: str) -> float:


        if hotkey not in self.performance_history:
            return 0.0

        history = self.performance_history[hotkey]
        if not history:
            return 0.0

        weights = np.exp(np.linspace(-1, 0, len(history)))
        weights /= weights.sum()

        quality_scores = []
        for perf in history:
            base_score = perf.quality_score * perf.success_rate

            time_factor = 1 / (1 + perf.response_time)

            score = base_score * time_factor
            quality_scores.append(score)

        return float(np.average(quality_scores, weights=weights))

    def record_quality_score(self, hotkey: str, score: float) -> None:
        if hotkey not in self.current_cycle_scores:
            self.current_cycle_scores[hotkey] = []
        self.current_cycle_scores[hotkey].append(score)

    def get_current_cycle_score(self, hotkey: str) -> float:
        if hotkey not in self.current_cycle_scores or not self.current_cycle_scores[hotkey]:
            return 0.0
        return sum(self.current_cycle_scores[hotkey]) / len(self.current_cycle_scores[hotkey])

    def clear_current_cycle_scores(self) -> None:
        self.current_cycle_scores.clear()

    def calculate_score(self, validator_hotkey: str, current_performance: ValidatorPerformance) -> float:
        current_quality_score = self.get_current_cycle_score(validator_hotkey)

        stake_score = current_performance.stake_weight * 0.2
        quality_score = current_quality_score * 0.7
        history_score = current_performance.historical_score * 0.1

        response_factor = 1 / (1 + current_performance.response_time)

        final_score = (stake_score + quality_score + history_score) * response_factor

        if validator_hotkey not in self.performance_history:
            self.performance_history[validator_hotkey] = []
        self.performance_history[validator_hotkey].append(current_performance)

        return final_score

    def calculate_reward(self, validator_hotkey: str, blocks_completed: int) -> float:
        if validator_hotkey not in self.performance_history:
            return 0.0

        performance = self.performance_history[validator_hotkey][-1]

        base_reward = min(1.0, blocks_completed / (720 * 2))

        quality_bonus = performance.quality_score * self.config.quality_bonus_ratio

        history_bonus = performance.historical_score * self.config.history_bonus_ratio

        final_reward = base_reward * (1 + quality_bonus + history_bonus)

        return min(1.0, max(0.0, final_reward))

    def update_historical_score(self, validator_hotkey: str) -> float:
        if validator_hotkey not in self.performance_history:
            return 0.0

        history = self.performance_history[validator_hotkey]
        if not history:
            return 0.0

        weights = np.exp(np.linspace(-1, 0, len(history)))
        weights /= weights.sum()

        scores = [p.quality_score * p.success_rate for p in history]
        return float(np.average(scores, weights=weights)) 