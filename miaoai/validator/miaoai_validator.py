import time
import os
import numpy as np
import random
import logging as python_logging
from typing import Tuple, Optional, List, Dict, Any

from dotenv import load_dotenv
from tabulate import tabulate

import bittensor as bt
from bittensor import logging

from miaoai.core.auth import load_public_key, verify_signature
from miaoai.core.config_manager import ConfigManager

from miaoai.core.scoring import ScoringSystem, ValidatorPerformance
from miaoai.core.allocation import AllocationManager, TaskAllocation
from miaoai.core.config import MiaoAIConfig
from miaoai.core.constants import (
    DEFAULT_ALLOCATION_STRATEGY,
    VERSION_KEY,
    BAD_COLDKEYS,
    U16_MAX,
    MIN_VALIDATOR_STAKE_DTAO,
    MIN_MINER_STAKE_DTAO,
    OWNER_DEFAULT_SCORE,
    FINAL_MIN_SCORE,
    DEFAULT_LOG_PATH,
    MAX_VALIDATOR_BLOCKS,
    CHECK_NODE_ACTIVE
)
from miaoai.core.path_utils import PathUtils
from miaoai.validator import BaseValidator
from miaoai.core.task_manager import TaskManager
from miaoai.core.validator_manager import ValidatorManager
from miaoai.core.task_type import TaskType
from miaoai.core.task_synapse import TaskSynapse

class MiaoAIValidator(BaseValidator):

    def __init__(self):
        self.current_allocations: List[TaskAllocation] = []
        self.allocation_strategy = os.getenv("ALLOCATION_STRATEGY", DEFAULT_ALLOCATION_STRATEGY)

        project_root = PathUtils.get_project_root()
        log_path = os.getenv("LOG_PATH", DEFAULT_LOG_PATH)
        self.log_dir = project_root / log_path
        self.log_dir.mkdir(parents=True, exist_ok=True)

        super().__init__()

        task_data_path = os.getenv("TASK_DATA_PATH", "tasks")
        task_data_full_path = PathUtils.get_task_data_path(task_data_path)
        self.task_manager = TaskManager(self.storage, str(task_data_full_path))
        self.validator_manager = ValidatorManager(self.storage)

        self.setup_bittensor_objects()

        self._dendrite = bt.Dendrite(wallet=self.wallet)

        check_validator_stake = os.getenv("CHECK_VALIDATOR_STAKE", "false").lower() == "true"
        if check_validator_stake:
            self.check_validator_stake()

        use_database = os.getenv("USE_DATABASE", "false").lower() == "true"
        validator_token = os.getenv("VALIDATOR_TOKEN", "")

        self._auth_headers = {"Authorization": f"Bearer {validator_token}"}

        self.config_manager = ConfigManager(
            use_database=use_database,
            hotkey=self.validator_hotkey,
            validator_token=validator_token
        )

        miaoai_config = MiaoAIConfig(
            stake_weight_ratio=0.2,
            min_blocks_per_validator=10,
            eval_interval=25,
            weights_interval=100,
            quality_bonus_ratio=0.7,
            config_manager=self.config_manager,
            history_bonus_ratio=0.1
        )

        self.scoring_system = ScoringSystem(config=miaoai_config)
        self.allocation_manager = AllocationManager(self.config)
        self.current_block = 0
        self.eval_interval = self.config.eval_interval
        self.last_update = 0

        self.total_blocks_run = 0

        self.blocks_since_last_weights = 0

        self.alpha = 0.95

        self.weights_interval = self.tempo *  (1 / 3)

        self.miner_tasks: Dict[str, str] = {}

        if not self.config.neuron.axon_off:

            self.axon = bt.axon(
                wallet=self.wallet,
                config=self.config,
            )

            self.axon.attach(
                forward_fn=self.forward
            )

            self.axon.start()

            self.subtensor.serve_axon(
                netuid=self.config.netuid,
                axon=self.axon,
            )

    def check_validator_stake(self):

        try:

            neurons = self.subtensor.neurons_lite(netuid=self.config.netuid)
            validator_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            validator_stake = float(neurons[validator_uid].stake)

            if validator_stake < MIN_VALIDATOR_STAKE_DTAO:
                raise ValueError(
                    f"Validator stake ({validator_stake:.2f} τ) is below minimum requirement "
                    f"({MIN_VALIDATOR_STAKE_DTAO} τ)"
                )

        except Exception as e:
            logging.error(f"Failed to verify validator stake: {str(e)}")
            raise

    def setup_logging_path(self) -> None:

        self.config.full_path = str(
            f"{self.log_dir}/validator/{self.config.wallet.name}/{self.config.wallet.hotkey}/netuid{self.config.netuid}")
        os.makedirs(self.config.full_path, exist_ok=True)

        self.config.logging_dir = self.config.full_path
        self.config.record_log = True

    def blacklist_fn(self, synapse: bt.Synapse) -> bool:
        if not synapse.dendrite.hotkey:
            return True
        return synapse.dendrite.hotkey in self.config.blacklist

    def priority_fn(self, synapse: bt.Synapse) -> float:
        if not synapse.dendrite.hotkey:
            return 0.0
        return 1.0

    def allocate_tasks(self) -> List[TaskAllocation]:

        check_max_blocks = os.getenv("CHECK_MAX_BLOCKS", "false").lower() == "true"
        if check_max_blocks and self.total_blocks_run >= MAX_VALIDATOR_BLOCKS:
            self.current_allocations = []
            return []

        self.task_manager.reset_all_assignments()

        available_miners = []
        neurons = self.subtensor.neurons_lite(netuid=self.config.netuid)

        miner_min_stake = self.config_manager.get_config().miner_min_stake_dtao

        for hotkey in self.metagraph.hotkeys:
            try:
                idx = self.metagraph.hotkeys.index(hotkey)
                neuron = neurons[idx]
                stake = float(neuron.stake)
                ip = neuron.axon_info.ip

                if ip == '0.0.0.0':
                    continue

                if stake < miner_min_stake:
                    logging.debug(f"Skipping miner {hotkey}: insufficient stake ({stake} < {miner_min_stake})")
                    continue

                available_miners.append(hotkey)

            except Exception as e:
                continue

        if not available_miners:
            return []

        allocations = self.allocation_manager.allocate_tasks(
            self.allocation_strategy,
            self.tempo * 2,
            available_miners,
            self.metagraph_info
        )

        self.current_allocations = allocations

        for alloc in allocations:
            uid = self.metagraph.hotkeys.index(alloc.miner_hotkey)

        return allocations

    async def send_tasks_to_miners(self):
        if not self.current_allocations:
            return

        validator_hotkey = self.wallet.hotkey.ss58_address

        for allocation in self.current_allocations:
            try:
                miner_hotkey = allocation.miner_hotkey
                miner_uid = self.metagraph.hotkeys.index(miner_hotkey)

                miner_axon = self.metagraph.axons[miner_uid]
                if not miner_axon or miner_axon.ip == '0.0.0.0':
                    continue

                task = self.task_manager.get_task_for_miner(miner_hotkey, validator_hotkey)
                if not task:
                    continue

                synapse = TaskSynapse()
                synapse.task_id = task.task_id
                synapse.task_text = task.text
                synapse.task_metadata = task.metadata
                synapse.blocks_allocated = allocation.blocks_allocated
                synapse.miner_hotkey = miner_hotkey
                synapse.validator_hotkey = validator_hotkey

                response = await self._dendrite.forward(
                    miner_axon,
                    synapse,
                    deserialize=True,
                    timeout=10.0
                )

                logging.info(f"Response from miner {miner_hotkey}: {response}")

                if response and response.get("success", False):
                    logging.info(f"Successfully sent task to miner {miner_hotkey}")
                    self.miner_tasks[miner_hotkey] = task.task_id

                    synapse.response = response.get("response")

                    self._process_response(miner_uid, miner_hotkey, synapse)
                else:
                    logging.warning(f"Failed to send task to miner {miner_hotkey}")

            except Exception as e:
                logging.error(f"Error sending task to miner {allocation.miner_hotkey}: {str(e)}")
                continue

    async def process_sentiment_request(self, synapse: TaskSynapse) -> Tuple[float, float]:
        start_time = time.time()

        try:
            signature = synapse.signature
            timestamp = synapse.timestamp
            miner_hotkey = synapse.dendrite.hotkey
            miner_uid = synapse.miner_uid
            if not miner_uid:
                return 0.0, 0.0

            is_valid = False

            if signature is not None and timestamp != 0:
                is_valid = True

            if not is_valid:
                return 0.0, 0.0

            if not self.current_allocations:
                self.current_allocations = []

            allocation = next(
                (alloc for alloc in self.current_allocations
                 if alloc.miner_hotkey == miner_hotkey),
                None
            )

            if not allocation:
                return 0.0, 0.0

            if not self.validator_manager.can_miner_get_task(miner_uid, miner_hotkey, synapse):
                return 0.0, 0.0

            validator_hotkey = self.wallet.hotkey.ss58_address

            task = self.task_manager.get_task_for_miner(miner_hotkey, validator_hotkey)

            if not task:
                logging.warning(f"No task available for miner {miner_uid}")
                return 0.0, 0.0

            synapse.task_id = task.task_id
            synapse.task_text = task.text
            synapse.task_metadata = task.metadata
            synapse.blocks_allocated = allocation.blocks_allocated

            response_time = time.time() - start_time
            return 1.0, response_time

        except Exception as e:
            logging.error(f"Error processing sentiment request: {str(e)}")
            return 0.0, 0.0

    def _process_response(self, miner_uid: int, miner_hotkey: str, synapse: TaskSynapse) -> bool:

        if not self._validate_response(synapse.task_id, synapse.response):

            self.validator_manager.update_validator_metrics(
                self.wallet.hotkey.ss58_address,
                False,
                synapse.response_time or 0.0
            )

            self.validator_manager.update_miner_metrics(
                miner_uid,
                synapse.task_id,
                False,
                synapse.response_time or 0.0,
                synapse
            )

            synapse.success = False
            synapse.error_message = "Invalid response"
            return synapse

        score = self._evaluate_response(synapse.task_id, synapse.response)

        self.scoring_system.record_quality_score(miner_hotkey, score)

        self.validator_manager.update_validator_metrics(
            self.wallet.hotkey.ss58_address,
            score > 0,
            synapse.response_time or 0.0
        )

        self.validator_manager.update_miner_metrics(
            miner_uid,
            synapse.task_id,
            score > 0,
            synapse.response_time or 0.0,
            synapse
        )

        if score > 0:
            self.task_manager.mark_task_completed(synapse.task_id)
            synapse.success = True
        else:
            synapse.success = False
            synapse.error_message = "Low quality response"

        return synapse

    def _validate_response(self, task_id: str, response: Dict[str, Any]) -> bool:

        try:
            task = self.task_manager.task_pool.get(task_id)
            if not task:
                return False

            task_type = TaskType.from_str(task.metadata.get("category", ""))

            if task_type == TaskType.OBJECT_DETECTION:
                if not isinstance(response.get("objects"), list):
                    return False
                min_objects = task.metadata.get("min_objects", 0)
                if len(response["objects"]) < min_objects:
                    return False

            elif task_type == TaskType.SENTIMENT_ANALYSIS:

                if "sentiment_score" not in response:
                    logging.warning("Missing sentiment_score in response")
                    return False
                if not isinstance(response["sentiment_score"], (int, float)):
                    logging.warning("sentiment_score must be a number")
                    return False
                if not 0 <= response["sentiment_score"] <= 1:
                    logging.warning("sentiment_score must be between 0 and 1")
                    return False

                if "confidence" not in response:
                    logging.warning("Missing confidence in response")
                    return False
                if not isinstance(response["confidence"], (int, float)):
                    logging.warning("confidence must be a number")
                    return False
                if not 0 <= response["confidence"] <= 1:
                    logging.warning("confidence must be between 0 and 1")
                    return False

            elif task_type == TaskType.TEXT_CLASSIFICATION:

                if "class_id" not in response:
                    logging.warning("Missing class_id in response")
                    return False
                if not isinstance(response["class_id"], (int, str)):
                    logging.warning("class_id must be an integer or string")
                    return False

                if "confidence" not in response:
                    logging.warning("Missing confidence in response")
                    return False
                if not isinstance(response["confidence"], (int, float)):
                    logging.warning("confidence must be a number")
                    return False
                if not 0 <= response["confidence"] <= 1:
                    logging.warning("confidence must be between 0 and 1")
                    return False

                if "description" in response and not isinstance(response["description"], str):
                    logging.warning("class_label must be a string")
                    return False

                if "allowed_classes" in task.metadata:
                    allowed_classes = task.metadata["allowed_classes"]
                    class_id = str(response["class_id"])
                    if class_id not in [str(c) for c in allowed_classes]:
                        logging.warning(f"class_id {class_id} not in allowed classes: {allowed_classes}")
                        return False

            elif task_type in [TaskType.SCENE_UNDERSTANDING, TaskType.EMOTION_ANALYSIS]:
                if not isinstance(response.get("description"), str):
                    return False

            return True

        except Exception as e:
            logging.error(f"Error validating response: {e}")
            return False

    def _evaluate_response(self, task_id: str, response: Dict[str, Any]) -> float:

        try:
            task = self.task_manager.task_pool.get(task_id)
            if not task:
                return 0.0

            task_type = TaskType.from_str(task.metadata.get("category", ""))

            if task_type == TaskType.OBJECT_DETECTION and task.ground_truth:

                detected_objects = set(response.get("objects", []))
                ground_truth_objects = set(task.ground_truth)
                correct_objects = detected_objects.intersection(ground_truth_objects)

                if not ground_truth_objects:
                    return 0.8

                precision = len(correct_objects) / len(detected_objects) if detected_objects else 0
                recall = len(correct_objects) / len(ground_truth_objects)

                if precision + recall == 0:
                    return 0.0
                f1_score = 2 * (precision * recall) / (precision + recall)
                return f1_score

            elif task_type in [TaskType.SENTIMENT_ANALYSIS, TaskType.EMOTION_ANALYSIS]:

                if "sentiment_score" not in response or "confidence" not in response:
                    return 0.0

                if task.ground_truth:
                    predicted_sentiment = "Positive" if response["sentiment_score"] > 0.5 else "Negative"
                    if predicted_sentiment == task.ground_truth:
                        return min(1.0, 0.7 + response["confidence"] * 0.3)
                    return max(0.0, response["confidence"] * 0.5)

                return min(0.8, 0.5 + response["confidence"] * 0.3)

            elif task_type == TaskType.SCENE_UNDERSTANDING:

                if "description" not in response or "confidence" not in response:
                    return 0.0

                if task.ground_truth:
                    description_match = response["description"].lower() in task.ground_truth.lower()
                    if description_match:
                        return min(1.0, 0.7 + response["confidence"] * 0.3)
                    return max(0.3, response["confidence"] * 0.5)

                return min(0.8, 0.5 + response["confidence"] * 0.3)

            elif task_type == TaskType.TEXT_CLASSIFICATION:
                if "class_id" not in response or "confidence" not in response:
                    return 0.0

                if task.ground_truth:
                    if str(response["class_id"]) == str(task.ground_truth):
                        return min(1.0, 0.7 + response["confidence"] * 0.3)
                    return max(0.0, response["confidence"] * 0.5)

                return min(0.8, 0.5 + response["confidence"] * 0.3)

            return 0.5

        except Exception as e:
            logging.error(f"Error evaluating response: {e}")
            return 0.0

    async def forward(self, synapse: TaskSynapse) -> TaskSynapse:
        try:
            miner_hotkey = synapse.dendrite.hotkey
            miner_uid = synapse.miner_uid


            if not synapse.response:

                score, response_time = await self.process_sentiment_request(synapse)
                if score > 0:

                    self.validator_manager.update_validator_metrics(
                        self.wallet.hotkey.ss58_address,
                        True,
                        response_time
                    )

                    self.validator_manager.update_miner_metrics(
                        miner_uid,
                        synapse.task_id,
                        True,
                        response_time,
                        synapse
                    )
                    synapse.success = True
                    return synapse
                else:
                    self.validator_manager.update_validator_metrics(
                        self.wallet.hotkey.ss58_address,
                        False,
                        response_time
                    )

                    self.validator_manager.update_miner_metrics(
                        miner_uid,
                        synapse.task_id,
                        False,
                        response_time,
                        synapse
                    )

                    synapse.success = False
                    synapse.error_message = "Failed to process task request"

                    return synapse

            else:
                if not self._validate_response(synapse.task_id, synapse.response):
                    self.validator_manager.update_validator_metrics(
                        self.wallet.hotkey.ss58_address,
                        False,
                        synapse.response_time or 0.0
                    )

                    self.validator_manager.update_miner_metrics(
                        miner_uid,
                        synapse.task_id,
                        False,
                        synapse.response_time or 0.0,
                        synapse
                    )

                    synapse.success = False
                    synapse.error_message = "Invalid response"
                    return synapse

                score = self._evaluate_response(synapse.task_id, synapse.response)

                self.scoring_system.record_quality_score(miner_hotkey, score)

                self.validator_manager.update_validator_metrics(
                    self.wallet.hotkey.ss58_address,
                    score > 0,
                    synapse.response_time or 0.0
                )

                self.validator_manager.update_miner_metrics(
                    miner_uid,
                    synapse.task_id,
                    score > 0,
                    synapse.response_time or 0.0,
                    synapse
                )

                if score > 0:
                    self.task_manager.mark_task_completed(synapse.task_id)
                    synapse.success = True
                else:
                    synapse.success = False
                    synapse.error_message = "Low quality response"

                return synapse

        except Exception as e:
            logging.error(f"Error in forward: {str(e)}")
            synapse.success = False
            synapse.error_message = str(e)
            return synapse

    def get_stats(self) -> Dict[str, Any]:
        task_stats = self.task_manager.get_task_stats()
        validator_stats = self.validator_manager.get_stats()

        return {
            **task_stats,
            **validator_stats,
            "validator_permit": self.validator_permit,
            "active_miners": len(self.miner_tasks)
        }

    def run(self):
        if self.config.state == "restore":
            self.restore_state_and_evaluate()
        else:
            self.resync_metagraph()


        self.allocate_tasks()
        next_sync_block = self.current_block + self.eval_interval

        try:
            while True:
                try:
                    if self.subtensor.wait_for_block(next_sync_block):
                        self.resync_metagraph()

                        self.total_blocks_run += self.eval_interval
                        self.blocks_since_last_weights += self.eval_interval
                        self.task_manager.update_blocks_run(self.total_blocks_run)

                        blocks_since_last = self.subtensor.blocks_since_last_update(
                            self.config.netuid,
                            self.uid
                        )

                        self.allocate_tasks()

                        import asyncio
                        asyncio.run(self.send_tasks_to_miners())

                        if blocks_since_last >= self.weights_interval and self.blocks_since_last_weights >= self.weights_interval:
                            success, msg = self.set_weights()
                            consensus = self.metagraph.consensus[self.uid].item()
                            incentive = self.metagraph.incentive[self.uid].item()
                            dividends = self.metagraph.dividends[self.uid].item()
                            current_epoch = self.current_block // 360
                            bt.logging.info(
                                f"after set_weights  consensus: {consensus}, incentive:{incentive}, dividends: {dividends}, current_epoch: {current_epoch}")
                            if success:
                                self.blocks_since_last_weights = 0
                            else:
                                continue

                        self.save_state()
                        validator_trust = self.subtensor.query_subtensor(
                            "ValidatorTrust",
                            params=[self.config.netuid],
                        )
                        normalized_validator_trust = validator_trust[self.uid] / U16_MAX if validator_trust[
                                                                                                self.uid] > 0 else 0
                        next_sync_block, reason = self.get_next_sync_block()

                except KeyboardInterrupt:
                    bt.logging.success("Keyboard interrupt detected. Exiting validator.")
                    break
                except Exception as e:
                    bt.logging.error(f"Error in validator loop: {str(e)}")
                    continue
        finally:
            if hasattr(self, 'axon'):
                self.axon.stop()
                bt.logging.info("Axon server stopped")

    def switch_allocation_strategy(self, strategy: str):
        if strategy not in ["stake", "equal"]:
            raise ValueError(f"Unknown allocation strategy: {strategy}")
        self.allocation_strategy = strategy
        self.allocate_tasks()

    def save_state(self):

        scores = [float(score) for score in self.scores]
        moving_avg_scores = [float(score) for score in self.moving_avg_scores]
        hotkeys = [str(key) for key in self.hotkeys]
        block_at_registration = [int(block) for block in self.block_at_registration]

        state = {
            "current_block": int(self.current_block),
            "total_blocks_run": int(self.total_blocks_run),
            "scores": scores,
            "moving_avg_scores": moving_avg_scores,
            "hotkeys": hotkeys,
            "block_at_registration": block_at_registration
        }
        self.storage.save_state(state)

    def restore_state_and_evaluate(self) -> None:
        state = self.storage.load_latest_state()
        if not state or "current_block" not in state:
            return

        self.total_blocks_run = state.get("total_blocks_run", 0)

        blocks_down = self.current_block - state["current_block"]
        if blocks_down >= (self.tempo * 1.5):
            return

        total_hotkeys = len(state.get("hotkeys", []))
        self.scores = state.get("scores", [0.0] * total_hotkeys)
        self.moving_avg_scores = state.get("moving_avg_scores", [0.0] * total_hotkeys)
        self.hotkeys = state.get("hotkeys", [])
        self.block_at_registration = state.get("block_at_registration", [])

        self.resync_metagraph()

        for idx in range(len(self.hotkeys)):
            if self.metagraph.coldkeys[idx] in BAD_COLDKEYS:
                self.moving_avg_scores[idx] = 0.0


    def set_weights(self) -> Tuple[bool, str]:

        try:
            check_max_blocks = os.getenv("CHECK_MAX_BLOCKS", "false").lower() == "true"
            if check_max_blocks and self.total_blocks_run >= MAX_VALIDATOR_BLOCKS:
                return True, ""
            miner_indices = []
            weights = []
            total_stake = 0.0


            neurons = self.subtensor.neurons_lite(netuid=self.config.netuid)

            validator_trust = self.subtensor.query_subtensor(
                "ValidatorTrust",
                params=[self.config.netuid],
            )

            total_stake = sum(float(neurons[idx].stake) for idx in range(len(self.metagraph.hotkeys))
                              if not validator_trust[idx] > 0)

            for idx, hotkey in enumerate(self.metagraph.hotkeys):
                is_validator = False
                try:
                    neuron = neurons[idx]
                    ip = neuron.axon_info.ip
                    if ip == '0.0.0.0':
                        continue
                    is_validator = self.metagraph.validator_permit[idx]
                    is_active = bool(neuron.active)
                    stake = float(neuron.stake)

                    if CHECK_NODE_ACTIVE and not is_active:
                        logging.debug(f"Skipping inactive node: {hotkey}")
                        continue

                except Exception as e:
                    bt.logging.warning(f"Error checking node status for {hotkey}: {e}")
                    continue

                historical_score = self.scoring_system.get_historical_score(hotkey)
                current_quality_score = self.scoring_system.get_current_cycle_score(hotkey)

                if historical_score == 0.0:
                    historical_score = self.alpha

                stake_weight = (stake / total_stake) * 0.2 if total_stake > 0 else 0

                final_score = (
                        stake_weight +
                        current_quality_score * 0.7 +
                        historical_score * 0.1
                )

                if current_quality_score > 0:
                    miner_indices.append(idx)
                    weights.append(final_score)

            validator_trust = validator_trust[self.uid]

            is_blacklisted = self.config_manager.is_validator_blacklisted(self.validator_hotkey)

            if is_blacklisted:
                return False, ""

            if not weights or all(w == 0 for w in weights):
                owner_uid = self.get_subnet_owner_uid()
                if owner_uid is None:
                    return False, "No subnet owner found"

                weights = [0.0] * len(self.metagraph.hotkeys)
                weights[owner_uid] = self.config_manager.get_config().owner_default_score
                miner_indices = list(range(len(self.metagraph.hotkeys)))
                logging.info(f"No valid miners found, setting all weight to subnet owner (uid: {owner_uid})")
            else:
                total_weight = sum(weights)
                if total_weight > 0:
                    MIN_WEIGHT_THRESHOLD = 0.001

                    weights = [w if w >= MIN_WEIGHT_THRESHOLD else 0.0 for w in weights]

                    total_weight = sum(weights)
                    if total_weight > 0:
                        weights = [w / total_weight for w in weights]
                    else:
                        owner_uid = self.get_subnet_owner_uid()
                        if owner_uid is not None:
                            weights = [0.0] * len(self.metagraph.hotkeys)
                            weights[owner_uid] = self.config_manager.get_config().owner_default_score

            success = self.subtensor.set_weights(
                netuid=self.config.netuid,
                wallet=self.wallet,
                uids=miner_indices,
                weights=weights,
                wait_for_inclusion=True,
                version_key=VERSION_KEY
            )

            if success:
                self.last_update = self.current_block
                self.scoring_system.clear_current_cycle_scores()
            else:
                logging.error("Failed to set weights")

            return success, ""

        except Exception as e:
            error_msg = f"Error setting weights: {str(e)}"
            logging.error(error_msg)
            return False, error_msg

    def _log_weights(self, indices: List[int], weights: List[float]) -> None:
        rows = []
        headers = ["UID", "Hotkey", "Weight", "Normalized (%)"]

        sorted_pairs = sorted(zip(indices, weights), key=lambda x: x[1], reverse=True)

        for idx, weight in sorted_pairs:
            if weight > 0:
                hotkey = self.metagraph.hotkeys[idx]

                rows.append([
                    idx,
                    f"{hotkey[:10]}...{hotkey[-6:]}",
                    f"{weight:.10f}",
                    f"{weight * 100:.10f}%"
                ])

        if not rows:
            logging.info(f"No miners receiving weights at block {self.current_block}")
            return

        table = tabulate(
            rows,
            headers=headers,
            tablefmt="grid",
            numalign="right",
            stralign="left"
        )
        logging.info(f"Weight distribution at block {self.current_block}:\n{table}")

    def get_subnet_owner_uid(self) -> Optional[int]:
        try:
            owner_coldkey = self.subtensor.query_subtensor(
                "SubnetOwner",
                params=[self.config.netuid]
            )
            return self.metagraph.coldkeys.index(owner_coldkey)
        except Exception as e:
            logging.error(f"Error getting subnet owner: {str(e)}")
            return None

    def serve_axon(self):

        try:


            self.axon = bt.axon(wallet=self.wallet, config=self.config)
            self.axon.attach(
                forward_fn=self.forward
            )

            self.axon.start()

            try:
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )
            except Exception as e:
                logging.error(f"Failed to serve Axon with exception: {e}")
                raise e

        except Exception as e:
            logging.error(
                f"Failed to create Axon initialize with exception: {e}"
            )
            raise e

    def setup_logging(self) -> None:

        logging(config=self.config, logging_dir=self.config.full_path)

        root_logger = python_logging.getLogger()
        root_logger.handlers = []
        null_handler = python_logging.NullHandler()
        root_logger.addHandler(null_handler)

        bt_logger = python_logging.getLogger("bittensor")
        bt_logger.propagate = False

        log_level = os.getenv("BT_LOGGING_INFO", "INFO")
        if log_level == "TRACE":
            logging.set_trace(True)
        else:
            logging.set_trace(False)
            logging.setLevel(log_level)

        check_max_blocks = os.getenv("CHECK_MAX_BLOCKS", "false").lower() == "true"



# Run the validator.
if __name__ == "__main__":
    validator_env = PathUtils.get_env_file_path("validator")
    default_env = PathUtils.get_env_file_path()

    if validator_env.exists():
        load_dotenv(validator_env)
    else:
        load_dotenv(default_env)
    validator = MiaoAIValidator()
    validator.run()
