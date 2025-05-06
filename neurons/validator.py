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

        # Call parent class initialization
        super(Validator, self).__init__(config=config)

        bt.logging.info("Loading validator status")

        self.test_database = self.initialize_test_database()
        self.load_state()

    def initialize_test_database(self):
        return {
            "special_test_cat_1": {"is_cat": True, "difficulty": "easy"},
            "special_test_cat_2": {"is_cat": True, "difficulty": "medium"},
            "special_test_cat_3": {"is_cat": True, "difficulty": "hard"},
            "special_test_not_cat_1": {"is_cat": False, "difficulty": "easy"},
            "special_test_not_cat_2": {"is_cat": False, "difficulty": "medium"},
            "special_test_not_cat_3": {"is_cat": False, "difficulty": "hard"},
        }

    async def forward(self, synapse: CatSoundProtocol) -> CatSoundProtocol:
        try:
            hotkey = synapse.dendrite.hotkey if hasattr(synapse, 'dendrite') and hasattr(synapse.dendrite, 'hotkey') else None
            bt.logging.info(f"Validation request from UUID={getattr(synapse.dendrite, 'uuid', 'Unknown')}")

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
            if int(time.time()) - int(resp_timestamp) > 600:
                return False
            self._special_aware_miners.add(hotkey)
            bt.logging.info(f"Miner {hotkey} successfully identified special test")
            return True
        except Exception as e:
            bt.logging.error(f"Error verifying special response: {e}")
            return False

    def select_test_sample(self):
        test_id = random.choice(list(self.test_database.keys()))
        return test_id, self.test_database[test_id]["is_cat"]

    def process_test_results(self, synapse):
        if synapse.is_cat_sound is None:
            bt.logging.warning("Miner did not return valid response")
            return

        result = "miao" if synapse.is_cat_sound else "not miao"
        bt.logging.info(
            f"Miner returned: {result}, probability: {synapse.probability}, "
            f"confidence: {synapse.confidence_level}, response time: {synapse.response_time}"
        )
        try:
            if not (hasattr(synapse, 'dendrite') and hasattr(synapse.dendrite, 'hotkey') and synapse.dendrite.hotkey):
                bt.logging.warning("Unable to get miner's hotkey information")
                return

            hotkey = synapse.dendrite.hotkey
            uid = self.metagraph.hotkeys.index(hotkey)

            hist = self.miner_history.setdefault(hotkey, {
                "total_tests": 0,
                "correct_tests": 0,
                "response_times": [],
                "last_responses": [],
                "special_test_success": False,
            })

            # special test check
            if self.last_test_sample and "::" in getattr(synapse, 'audio_data', ''):
                if self._verify_special_response(synapse, hotkey):
                    hist["special_test_success"] = True

            hist["total_tests"] += 1

            # correctness via ground-truth
            is_correct = hasattr(synapse, 'ground_truth') and synapse.is_cat_sound == synapse.ground_truth
            if is_correct:
                hist["correct_tests"] += 1

            # record response time & last responses
            if synapse.response_time is not None:
                hist["response_times"].append(synapse.response_time)
            hist["last_responses"].append({
                "is_cat_sound": synapse.is_cat_sound,
                "probability": synapse.probability,
                "confidence_level": synapse.confidence_level
            })
            if len(hist["last_responses"]) > 5:
                hist["last_responses"].pop(0)

            # update histories & statuses
            self.check_dtao_balance(hotkey)
            self.detect_model_usage(hotkey)

            # score calculation
            score = self.calculate_score(hotkey, synapse, is_correct)
            self.scores[uid] = score
            self.raw_scores[uid] = score
            bt.logging.info(f"Miner {uid} score: {score}")

        except ValueError as e:
            bt.logging.warning(f"Unable to find corresponding miner UID: {e}")

    def check_dtao_balance(self, hotkey):
        now = time.time()
        if hotkey in self.last_balance_check and now - self.last_balance_check[hotkey] < 3600:
            return
        try:
            balance = float(self.metagraph.S[self.metagraph.hotkeys.index(hotkey)]) * 1000
            self.miner_dtao_balance[hotkey] = balance
            self.last_balance_check[hotkey] = now
            bt.logging.debug(f"Miner {hotkey} dTAO balance: {balance}")
        except Exception as e:
            self.miner_dtao_balance.setdefault(hotkey, 0.0)
            bt.logging.error(f"Failed to check dTAO balance: {e}")

    def detect_model_usage(self, hotkey):
        hist = self.miner_history.get(hotkey, {})
        if hist.get("total_tests", 0) < 5:
            return None

        accuracy = hist["correct_tests"] / hist["total_tests"]
        times = hist["response_times"]
        if len(times) >= 3:
            avg = sum(times) / len(times)
            std = np.std(times) if len(times) > 1 else 0.0
            time_ok = 0.1 <= avg <= 5.0
            var_ok = std > 0.05
        else:
            time_ok = var_ok = True

        last_res = hist["last_responses"]
        if len(last_res) >= 3:
            same = all(r["is_cat_sound"] == last_res[0]["is_cat_sound"] for r in last_res)
            probs = [r["probability"] for r in last_res if r.get("probability") is not None]
            prob_std = np.std(probs) if len(probs) >= 3 else 1.0
            resp_var = not (same and prob_std < 0.01)
        else:
            resp_var = True

        uses_real = (accuracy >= 0.6 and time_ok and var_ok and resp_var) or (hotkey in self._special_aware_miners)
        self.miner_model_status[hotkey] = uses_real
        bt.logging.info(f"Miner {hotkey} using real model: {uses_real}")
        return uses_real

    def calculate_score(self, hotkey, synapse, is_correct):
        balance = self.miner_dtao_balance.get(hotkey, 0.0)
        if balance < self.min_dtao_balance:
            bt.logging.warning(
                f"Miner {hotkey} insufficient dTAO balance ({balance} < {self.min_dtao_balance}), score=0"
            )
            return 0.0
        uses_model = self.miner_model_status.get(hotkey)
        if uses_model is False:
            bt.logging.info(f"Miner {hotkey} not running real model, score=0")
            return 0.0
        if not is_correct:
            bt.logging.info(f"Miner {hotkey} answered incorrectly, score=0")
            return 0.0
        bt.logging.info(f"Miner {hotkey} correct and real model, score=1.0")
        return 1.0

    async def create_synapse(self) -> CatSoundProtocol:
        # Clear scores at the start of each round
        self.scores = {}

        if random.random() < 0.9:
            test_id, is_cat = self.select_test_sample()
            encoded = self._create_special_test(test_id)
            base64_audio = encoded
        else:
            test_id, is_cat = self.select_test_sample()
            base64_audio = base64.b64encode(f"TEST:{test_id}".encode()).decode('utf-8')

        synapse = CatSoundProtocol(audio_data=base64_audio)
        # Attach sample ID and ground truth for scoring
        synapse.sample_id = test_id
        synapse.ground_truth = is_cat

        self.last_test_sample = test_id
        self.last_test_label = is_cat
        return synapse

    async def concurrent_forward(self):
        try:
            coros = [self.forward(await self.create_synapse())
                     for _ in range(self.config.neuron.num_concurrent_forwards)]
            await asyncio.gather(*coros)

            self.test_round += 1
            if self.test_round % 10 == 0:
                self.save_state()

            # call parent to submit weights on-chain
            await super().concurrent_forward()

        except Exception as e:
            bt.logging.error(f"Error during concurrent validation: {e}")

    def save_state(self):
        try:
            # ensure all attrs exist
            self.miner_history = getattr(self, "miner_history", {})
            self.miner_model_status = getattr(self, "miner_model_status", {})
            self.test_round = getattr(self, "test_round", 0)
            self._special_aware_miners = getattr(self, "_special_aware_miners", set())
            self.miner_dtao_balance = getattr(self, "miner_dtao_balance", {})
            self.last_balance_check = getattr(self, "last_balance_check", {})

            # prepare serializable history
            serial = {}
            for hk, h in self.miner_history.items():
                serial[hk] = {
                    "total_tests": h.get("total_tests", 0),
                    "correct_tests": h.get("correct_tests", 0),
                    "response_times": h.get("response_times", []),
                    "special_test_success": h.get("special_test_success", False),
                    "last_responses": [
                        {
                            "is_cat_sound": r.get("is_cat_sound", False),
                            "probability": float(r.get("probability", 0.0)),
                            "confidence_level": r.get("confidence_level", None)
                        }
                        for r in h.get("last_responses", [])
                    ]
                }

            data = {
                "miner_history": serial,
                "miner_model_status": self.miner_model_status,
                "test_round": self.test_round,
                "special_aware_miners": list(self._special_aware_miners),
                "miner_dtao_balance": self.miner_dtao_balance,
                "last_balance_check": self.last_balance_check
            }

            data_dir = os.path.join(
                getattr(self.config.neuron, "full_path", "."), "data")
            os.makedirs(data_dir, exist_ok=True)
            path = os.path.join(data_dir, "validator_state.json")
            with open(path, "w") as f:
                json.dump(data, f)
            bt.logging.info(f"Validator state saved to {path}")

        except Exception as e:
            bt.logging.error(f"Error saving validator state: {e}")

    def load_state(self):
        try:
            data_dir = os.path.join(
                getattr(self.config.neuron, "full_path", "."), "data")
            path = os.path.join(data_dir, "validator_state.json")
            if os.path.exists(path):
                with open(path, "r") as f:
                    data = json.load(f)
                self.miner_history = data.get("miner_history", {})
                self.miner_model_status = data.get("miner_model_status", {})
                self.test_round = data.get("test_round", 0)
                self._special_aware_miners = set(data.get("special_aware_miners", []))
                self.miner_dtao_balance = data.get("miner_dtao_balance", {})
                self.last_balance_check = data.get("last_balance_check", {})
            bt.logging.info(f"Validator state loaded, test_round={self.test_round}")
        except Exception as e:
            bt.logging.error(f"Error loading validator state: {e}")
        # Initialize scores dictionary for this session
        self.scores = {}

    def score(self, uid):
        """Return the calculated score for miner uid from the last round."""
        return self.scores.get(uid, 0.0)


def get_config():
    parser = argparse.ArgumentParser()
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    parser.add_argument("--netuid", type=int, default=1, help="Subnet ID where the validator operates")
    parser.add_argument("--neuron.validation_interval", type=int, default=5,
                        help="Time interval between validations (seconds)")
    parser.add_argument("--neuron.sample_size", type=int, default=10,
                        help="Miner sample size for each validation")
    return bt.config(parser)


if __name__ == "__main__":
    import argparse

    config = get_config()
    config.neuron.full_path = os.path.expanduser(os.path.dirname(os.path.abspath(__file__)))

    bt.logging.info("Initializing validator node...")
    validator = Validator(config=config)

    while True:
        bt.logging.info("Validator running ... Press CTRL+C to stop")
        time.sleep(60)
