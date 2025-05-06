# MIT License
# Copyright © 2025 MIAO

import time
import bittensor as bt
import asyncio
import base64
import numpy as np
import os
import json
import random
import hashlib
import hmac
import requests
import torch
from template.base.validator import BaseValidatorNeuron
from template.protocol import CatSoundProtocol


class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        if config is None:
            config = bt.config()
            config.neuron = bt.Config()
            config.neuron.validation_interval = 5
            config.neuron.sample_size = 2

        elif not hasattr(config, 'neuron'):
            config.neuron = bt.Config()
            config.neuron.validation_interval = 5
            config.neuron.sample_size = 2

        # Initialize attributes before parent class initialization
        self.miner_history = {}
        self.miner_model_status = {}
        self.test_round = 0
        self.last_test_sample = None
        self.last_test_label = None
        self._special_mark = "MIAO_SPECIAL_MARK_VERSION_2024"
        self._verification_key = "eZx7K9Lp2QsTw5RmNvGbHj"
        self._special_aware_miners = set()
        self.min_dtao_balance = 50.0
        self.miner_dtao_balance = {}
        self.last_balance_check = {}
        self.raw_scores = {}

        # Call parent class initialization method
        super(Validator, self).__init__(config=config)

        bt.logging.info("Loading validator status")

        self.test_database = self.initialize_test_database()
        self.load_state()

    def initialize_test_database(self):
        db = {
            "special_test_cat_1": {"is_cat": True, "difficulty": "easy"},
            "special_test_cat_2": {"is_cat": True, "difficulty": "medium"},
            "special_test_cat_3": {"is_cat": True, "difficulty": "hard"},
            "special_test_not_cat_1": {"is_cat": False, "difficulty": "easy"},
            "special_test_not_cat_2": {"is_cat": False, "difficulty": "medium"},
            "special_test_not_cat_3": {"is_cat": False, "difficulty": "hard"},
        }
        return db

    async def forward(
            self, synapse: CatSoundProtocol
    ) -> CatSoundProtocol:
        try:
            hotkey = synapse.dendrite.hotkey if hasattr(synapse, 'dendrite') and hasattr(synapse.dendrite,
                                                                                         'hotkey') else None

            bt.logging.info(
                f"Validation request from UUID={synapse.dendrite.uuid if hasattr(synapse, 'dendrite') else 'Unknown'}")

            should_send_special_test = random.random() < 0.9

            if should_send_special_test:
                test_id, is_cat = self.select_test_sample()
                encoded_test = self._create_special_test(test_id)
                bt.logging.debug(f"Sending hidden test {test_id}")
            else:
                test_id, is_cat = self.select_test_sample()
                encoded_test = base64.b64encode(f"TEST:{test_id}".encode()).decode('utf-8')

            if self.last_test_sample is None:
                synapse.audio_data = encoded_test
                self.last_test_sample = test_id
                self.last_test_label = is_cat
            else:
                self.process_test_results(synapse)
                synapse.audio_data = encoded_test
                self.last_test_sample = test_id
                self.last_test_label = is_cat

        except Exception as e:
            bt.logging.error(f"Error processing validation request: {e}")

        return synapse

    def _create_special_test(self, test_id):
        timestamp = str(int(time.time()))
        nonce = hashlib.sha256(os.urandom(16)).hexdigest()[:8]
        challenge = hmac.new(
            self._verification_key.encode(),
            f"{timestamp}:{nonce}".encode(),
            hashlib.sha256
        ).hexdigest()[:16]

        special_data = f"TEST:{test_id}::{self._special_mark}:{timestamp}:{nonce}:{challenge}"
        return base64.b64encode(special_data.encode()).decode('utf-8')

    def _verify_special_response(self, synapse, hotkey):
        special_response = getattr(synapse, 'special_verification', None)

        if not special_response:
            return False

        try:
            parts = special_response.split(':')
            if len(parts) != 4:
                return False

            marker, resp_timestamp, resp_nonce, resp_signature = parts

            if marker != self._special_mark:
                return False

            current_time = int(time.time())
            response_time = int(resp_timestamp)
            if current_time - response_time > 600:
                return False

            self._special_aware_miners.add(hotkey)
            bt.logging.info(f"Miner {hotkey} successfully identified special test")

            return True
        except Exception as e:
            bt.logging.error(f"Error verifying special response: {e}")
            return False

    def select_test_sample(self):
        test_keys = list(self.test_database.keys())

        test_id = random.choice(test_keys)
        is_cat = self.test_database[test_id]["is_cat"]

        return test_id, is_cat

    def process_test_results(self, synapse):
        if synapse.is_cat_sound is not None:
            result = "miao" if synapse.is_cat_sound else "not miao"
            bt.logging.info(
                f"Miner returned: {result}, probability: {synapse.probability}, confidence: {synapse.confidence_level}, response time: {synapse.response_time}")

            try:
                if hasattr(synapse, 'dendrite') and hasattr(synapse.dendrite, 'hotkey') and synapse.dendrite.hotkey:
                    uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
                    hotkey = synapse.dendrite.hotkey

                    if hotkey not in self.miner_history:
                        self.miner_history[hotkey] = {
                            "total_tests": 0,
                            "correct_tests": 0,
                            "response_times": [],
                            "last_responses": [],
                            "special_test_success": False,
                        }

                    if self.last_test_sample and "::" in getattr(synapse, 'audio_data', ''):
                        special_valid = self._verify_special_response(synapse, hotkey)
                        if special_valid:
                            self.miner_history[hotkey]["special_test_success"] = True

                    self.miner_history[hotkey]["total_tests"] += 1

                    is_correct = (synapse.is_cat_sound == self.last_test_label)
                    if is_correct:
                        self.miner_history[hotkey]["correct_tests"] += 1

                    if synapse.response_time is not None:
                        self.miner_history[hotkey]["response_times"].append(synapse.response_time)

                    if len(self.miner_history[hotkey]["last_responses"]) >= 5:
                        self.miner_history[hotkey]["last_responses"].pop(0)
                    self.miner_history[hotkey]["last_responses"].append({
                        "is_cat_sound": synapse.is_cat_sound,
                        "probability": synapse.probability,
                        "confidence_level": synapse.confidence_level
                    })

                    self.check_dtao_balance(hotkey)
                    self.detect_model_usage(hotkey)

                    score = self.calculate_score(hotkey, synapse, is_correct)

                    self.scores[uid] = score
                    self.raw_scores[uid] = score
                    bt.logging.info(f"Miner {uid} score: {score}")
                else:
                    bt.logging.warning("Unable to get miner's hotkey information")
            except ValueError as e:
                bt.logging.warning(f"Unable to find corresponding miner UID: {e}")
        else:
            bt.logging.warning("Miner did not return valid response")

    def check_dtao_balance(self, hotkey):
        current_time = time.time()
        if (hotkey in self.last_balance_check and
                current_time - self.last_balance_check.get(hotkey, 0) < 3600):
            return

        try:
            balance = float(self.metagraph.S[self.metagraph.hotkeys.index(hotkey)]) * 1000

            self.miner_dtao_balance[hotkey] = balance
            self.last_balance_check[hotkey] = current_time

            bt.logging.debug(f"Miner {hotkey} dTAO balance: {balance}")

        except Exception as e:
            if hotkey not in self.miner_dtao_balance:
                self.miner_dtao_balance[hotkey] = 0.0
            bt.logging.error(f"Failed to check dTAO balance: {e}")

    def detect_model_usage(self, hotkey):
        history = self.miner_history.get(hotkey, {})

        if history.get("total_tests", 0) >= 5:
            accuracy = history.get("correct_tests", 0) / history.get("total_tests", 1)

            response_times = history.get("response_times", [])
            if len(response_times) >= 3:
                avg_time = sum(response_times) / len(response_times)
                time_std = np.std(response_times) if len(response_times) > 1 else 0

                time_reasonable = 0.1 <= avg_time <= 5.0
                time_variability = time_std > 0.05
            else:
                time_reasonable = True
                time_variability = True

            last_responses = history.get("last_responses", [])
            if len(last_responses) >= 3:
                same_result = all(
                    r.get("is_cat_sound") == last_responses[0].get("is_cat_sound", False) for r in last_responses)

                probs = [r.get("probability", 0.0) for r in last_responses if r.get("probability") is not None]
                same_prob = False
                if probs and len(probs) >= 3:
                    prob_std = np.std(probs)
                    same_prob = prob_std < 0.01

                response_variability = not (same_result and same_prob)
            else:
                response_variability = True

            handles_special_tests = hotkey in self._special_aware_miners

            uses_real_model = (
                    accuracy >= 0.6 and
                    time_reasonable and
                    time_variability and
                    response_variability
            )

            if handles_special_tests:
                uses_real_model = True

            self.miner_model_status[hotkey] = uses_real_model

            bt.logging.info(f"Miner {hotkey} using real model: {uses_real_model}")
            return uses_real_model
        else:
            return None

    def calculate_score(self, hotkey, synapse, is_correct):
        balance = self.miner_dtao_balance.get(hotkey, 0.0)
        if balance < self.min_dtao_balance:
            bt.logging.warning(
                f"Miner {hotkey} insufficient dTAO balance (current: {balance}, required: {self.min_dtao_balance}), score is 0")
            return 0.0

        uses_model = self.miner_model_status.get(hotkey)
        
        if uses_model is not None and uses_model is False:
            bt.logging.info(f"Miner {hotkey} not running real model, score is 0")
            return 0.0
            
        if not is_correct:
            bt.logging.info(f"Miner {hotkey} answered incorrectly, score is 0")
            return 0.0

        bt.logging.info(f"Miner {hotkey} running model correctly and answered correctly, score is 1.0")
        return 1.0

    def set_weights(self):
        try:
            weights = torch.zeros(self.metagraph.n)

            for uid, score in self.raw_scores.items():
                if uid < len(weights):
                    weights[uid] = score
        
            non_zero_sum = torch.sum(weights)
            non_zero_count = torch.sum(weights > 0).item()
        
            bt.logging.info(f"Original weights: non-zero count: {non_zero_count}, sum: {non_zero_sum.item()}")
        
            if non_zero_count > 0 and non_zero_sum > non_zero_count:
                scale_factor = non_zero_count / non_zero_sum
                weights = weights * scale_factor
                bt.logging.info(f"Applied scale factor: {scale_factor}, new sum: {torch.sum(weights).item()}")
        
            uids = torch.arange(0, len(weights))
            self.subtensor.set_weights(
                netuid=self.config.netuid,
                wallet=self.wallet,
                uids=uids,
                weights=weights,
                wait_for_inclusion=True
            )
            bt.logging.info(f"Set {len(weights)} weights")
        
        except Exception as e:
            bt.logging.error(f"Error setting weights: {e}")

    async def create_synapse(self) -> CatSoundProtocol:
        should_send_special_test = random.random() < 0.9

        if should_send_special_test:
            test_id, is_cat = self.select_test_sample()
            encoded_test = self._create_special_test(test_id)
            base64_audio = encoded_test
            bt.logging.debug(f"Sending test {test_id}")
        else:
            test_id, is_cat = self.select_test_sample()
            test_audio = f"TEST:{test_id}".encode()
            base64_audio = base64.b64encode(test_audio).decode('utf-8')

        synapse = CatSoundProtocol(
            audio_data=base64_audio
        )

        self.last_test_sample = test_id
        self.last_test_label = is_cat

        return synapse

    async def concurrent_forward(self):
        try:
            coroutines = []
            for _ in range(self.config.neuron.num_concurrent_forwards):
                synapse = await self.create_synapse()
                coroutines.append(self.forward(synapse))
            await asyncio.gather(*coroutines)

            self.test_round += 1

            if self.test_round % 10 == 0:
                self.save_state()

        except Exception as e:
            bt.logging.error(f"Error during concurrent validation: {e}")

    def save_state(self):
        try:
            if not hasattr(self, 'miner_history'):
                self.miner_history = {}
            if not hasattr(self, 'miner_model_status'):
                self.miner_model_status = {}
            if not hasattr(self, 'test_round'):
                self.test_round = 0
            if not hasattr(self, '_special_aware_miners'):
                self._special_aware_miners = set()
            if not hasattr(self, 'miner_dtao_balance'):
                self.miner_dtao_balance = {}
            if not hasattr(self, 'last_balance_check'):
                self.last_balance_check = {}

            serializable_history = {}
            for hotkey, history in self.miner_history.items():
                serializable_history[hotkey] = {
                    "total_tests": history.get("total_tests", 0),
                    "correct_tests": history.get("correct_tests", 0),
                    "response_times": history.get("response_times", []),
                    "special_test_success": history.get("special_test_success", False),
                    "last_responses": [
                        {
                            "is_cat_sound": r.get("is_cat_sound", False),
                            "probability": float(r.get("probability", 0.0)) if r.get(
                                "probability") is not None else None,
                            "confidence_level": r.get("confidence_level", "None")
                        }
                        for r in history.get("last_responses", [])
                    ]
                }

            data_to_save = {
                "miner_history": serializable_history,
                "miner_model_status": self.miner_model_status,
                "test_round": self.test_round,
                "special_aware_miners": list(self._special_aware_miners),
                "miner_dtao_balance": self.miner_dtao_balance,
                "last_balance_check": self.last_balance_check
            }

            data_dir = os.path.join(self.config.neuron.full_path if hasattr(self.config.neuron, 'full_path') else ".",
                                    "data")
            os.makedirs(data_dir, exist_ok=True)

            state_path = os.path.join(data_dir, "validator_state.json")
            with open(state_path, "w") as f:
                json.dump(data_to_save, f)

            bt.logging.info(f"Validator state saved to {state_path}")

        except Exception as e:
            bt.logging.error(f"Error saving validator state: {e}")
            try:
                data_dir = os.path.join(
                    self.config.neuron.full_path if hasattr(self.config.neuron, 'full_path') else ".", "data")
                os.makedirs(data_dir, exist_ok=True)

                state_path = os.path.join(data_dir, "validator_state.json")
                minimal_state = {
                    "miner_history": {},
                    "miner_model_status": {},
                    "test_round": 0,
                    "special_aware_miners": [],
                    "miner_dtao_balance": {},
                    "last_balance_check": {}
                }

                with open(state_path, "w") as f:
                    json.dump(minimal_state, f)

                bt.logging.info(f"Created minimal state file: {state_path}")
            except Exception as inner_e:
                bt.logging.error(f"Failed to create minimal state file: {inner_e}")

    def load_state(self):
        try:
            data_dir = os.path.join(self.config.neuron.full_path if hasattr(self.config.neuron, 'full_path') else ".",
                                    "data")
            state_path = os.path.join(data_dir, "validator_state.json")

            if os.path.exists(state_path):
                with open(state_path, "r") as f:
                    data = json.load(f)

                self.miner_history = data.get("miner_history", {})
                self.miner_model_status = data.get("miner_model_status", {})
                self.test_round = data.get("test_round", 0)
                special_aware_miners = data.get("special_aware_miners", [])
                self._special_aware_miners = set(special_aware_miners)
                self.miner_dtao_balance = data.get("miner_dtao_balance", {})
                self.last_balance_check = data.get("last_balance_check", {})

                bt.logging.info(f"Validator state loaded, current test round: {self.test_round}")
            else:
                bt.logging.info("Validator state file not found, using default initial state")
        except Exception as e:
            bt.logging.error(f"Error loading validator state: {e}")


def get_config():
    parser = argparse.ArgumentParser()
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    parser.add_argument("--netuid", type=int, default=1, help="Subnet ID where the validator operates")
    parser.add_argument(
        "--neuron.validation_interval",
        type=int,
        default=5,
        help="Time interval between validations (seconds)",
    )
    parser.add_argument(
        "--neuron.sample_size", type=int, default=10, help="Miner sample size for each validation"
    )
    return bt.config(parser)


if __name__ == "__main__":
    import argparse

    config = get_config()
    config.neuron.full_path = os.path.expanduser(
        os.path.dirname(os.path.abspath(__file__))
    )

    bt.logging.info("Initializing validator node...")
    validator = Validator(config=config)

    while True:
        bt.logging.info("Validator running ... Press CTRL+C to stop")
        time.sleep(60)
