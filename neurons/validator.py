# MIT License
# Copyright © 2025 MIAO

import time
import os
import json
import random
import base64
import hashlib
import hmac
import asyncio

import bittensor as bt
import numpy as np
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

        # 初始化历史与状态容器
        self.miner_history = {}
        self.miner_model_status = {}
        self.test_round = 0
        self._special_mark = "MIAO_SPECIAL_MARK_VERSION_2024"
        self._verification_key = "eZx7K9Lp2QsTw5RmNvGbHj"
        self._special_aware_miners = set()
        self.min_dtao_balance = 50.0
        self.miner_dtao_balance = {}
        self.last_balance_check = {}
        # 本轮分数
        self.scores = {}

        super(Validator, self).__init__(config=config)
        bt.logging.info("Loading validator status")

        self.test_database = self.initialize_test_database()
        self.load_state()

    def initialize_test_database(self):
        return {
            "special_test_cat_1":     {"is_cat": True,  "difficulty": "easy"},
            "special_test_cat_2":     {"is_cat": True,  "difficulty": "medium"},
            "special_test_cat_3":     {"is_cat": True,  "difficulty": "hard"},
            "special_test_not_cat_1": {"is_cat": False, "difficulty": "easy"},
            "special_test_not_cat_2": {"is_cat": False, "difficulty": "medium"},
            "special_test_not_cat_3": {"is_cat": False, "difficulty": "hard"},
        }

    async def forward(self, synapse: CatSoundProtocol) -> CatSoundProtocol:
        try:
            hotkey = getattr(synapse.dendrite, 'hotkey', None)
            bt.logging.info(f"Validation request from UUID={getattr(synapse.dendrite, 'uuid', 'Unknown')}")

            # 90% 发送 hidden test, 10% 普通 test
            if random.random() < 0.9:
                test_id, is_cat = self.select_test_sample()
                encoded = self._create_special_test(test_id)
            else:
                test_id, is_cat = self.select_test_sample()
                encoded = base64.b64encode(f"TEST:{test_id}".encode()).decode('utf-8')

            # 首次发包
            if not hasattr(synapse, 'sent_once'):
                synapse.audio_data    = encoded
                synapse.sent_once     = True
                synapse.sample_id     = test_id
                synapse.ground_truth  = is_cat
            else:
                # 处理上一次返回，再发新包
                self.process_test_results(synapse)
                synapse.audio_data    = encoded
                synapse.sample_id     = test_id
                synapse.ground_truth  = is_cat

        except Exception as e:
            bt.logging.error(f"Error in forward(): {e}")

        return synapse

    def _create_special_test(self, test_id: str) -> str:
        timestamp = str(int(time.time()))
        nonce     = hashlib.sha256(os.urandom(16)).hexdigest()[:8]
        challenge = hmac.new(
            self._verification_key.encode(),
            f"{timestamp}:{nonce}".encode(),
            hashlib.sha256
        ).hexdigest()[:16]
        payload = f"TEST:{test_id}::{self._special_mark}:{timestamp}:{nonce}:{challenge}"
        return base64.b64encode(payload.encode()).decode('utf-8')

    def _verify_special_response(self, synapse: CatSoundProtocol, hotkey: str) -> bool:
        special = getattr(synapse, 'special_verification', None)
        if not special:
            return False
        try:
            marker, ts, nonce, sig = special.split(':')
            if marker != self._special_mark:
                return False
            if int(time.time()) - int(ts) > 600:
                return False
            self._special_aware_miners.add(hotkey)
            bt.logging.info(f"Miner {hotkey} passed special test")
            return True
        except Exception as e:
            bt.logging.error(f"Error in _verify_special_response(): {e}")
            return False

    def select_test_sample(self):
        test_id = random.choice(list(self.test_database.keys()))
        return test_id, self.test_database[test_id]["is_cat"]

    def process_test_results(self, synapse: CatSoundProtocol):
        if synapse.is_cat_sound is None:
            bt.logging.warning("Miner did not return valid response")
            return

        hotkey = getattr(synapse.dendrite, 'hotkey', None)
        if hotkey is None:
            bt.logging.warning("Cannot get miner hotkey")
            return

        uid = self.metagraph.hotkeys.index(hotkey)
        hist = self.miner_history.setdefault(hotkey, {
            "total_tests": 0,
            "correct_tests": 0,
            "response_times": [],
            "last_responses": [],
            "special_test_success": False,
        })

        # special test 验证
        if "::" in getattr(synapse, 'audio_data', ''):
            if self._verify_special_response(synapse, hotkey):
                hist["special_test_success"] = True

        hist["total_tests"] += 1

        # 根据 ground_truth 判断正确性
        is_correct = synapse.is_cat_sound == getattr(synapse, 'ground_truth', False)
        if is_correct:
            hist["correct_tests"] += 1

        # 记录延迟与历史响应
        if synapse.response_time is not None:
            hist["response_times"].append(synapse.response_time)
        hist["last_responses"].append({
            "is_cat_sound": synapse.is_cat_sound,
            "probability":  synapse.probability,
            "confidence_level": synapse.confidence_level
        })
        if len(hist["last_responses"]) > 5:
            hist["last_responses"].pop(0)

        # 更新 dTAO 余额与模型使用状态
        self.check_dtao_balance(hotkey)
        self.detect_model_usage(hotkey)

        # 计算分数
        score = self.calculate_score(hotkey, synapse, is_correct)
        self.scores[uid] = score
        bt.logging.info(f"Miner {uid} score: {score}")

    def check_dtao_balance(self, hotkey: str):
        now = time.time()
        if hotkey in self.last_balance_check and now - self.last_balance_check[hotkey] < 3600:
            return
        try:
            bal = float(self.metagraph.S[self.metagraph.hotkeys.index(hotkey)]) * 1000
            self.miner_dtao_balance[hotkey] = bal
            self.last_balance_check[hotkey] = now
            bt.logging.debug(f"{hotkey} dTAO balance: {bal}")
        except Exception as e:
            self.miner_dtao_balance.setdefault(hotkey, 0.0)
            bt.logging.error(f"check_dtao_balance() error: {e}")

    def detect_model_usage(self, hotkey: str):
        hist = self.miner_history.get(hotkey, {})
        if hist.get("total_tests", 0) < 5:
            return None

        acc = hist["correct_tests"] / hist["total_tests"]
        times = hist["response_times"]
        if len(times) >= 3:
            avg = sum(times)/len(times)
            std = np.std(times) if len(times)>1 else 0
            time_ok = 0.1 <= avg <= 5.0
            var_ok  = std > 0.05
        else:
            time_ok = var_ok = True

        last = hist["last_responses"]
        if len(last) >= 3:
            same = all(r["is_cat_sound"]==last[0]["is_cat_sound"] for r in last)
            probs = [r["probability"] for r in last if r.get("probability") is not None]
            prob_std = np.std(probs) if len(probs)>=3 else 1.0
            resp_ok = not (same and prob_std<0.01)
        else:
            resp_ok = True

        uses_real = (acc>=0.6 and time_ok and var_ok and resp_ok) or (hotkey in self._special_aware_miners)
        self.miner_model_status[hotkey] = uses_real
        bt.logging.info(f"{hotkey} using real model: {uses_real}")
        return uses_real

    def calculate_score(self, hotkey: str, synapse: CatSoundProtocol, is_correct: bool) -> float:
        bal = self.miner_dtao_balance.get(hotkey, 0.0)
        if bal < self.min_dtao_balance:
            bt.logging.warning(f"{hotkey} low dTAO ({bal}), score=0")
            return 0.0
        if self.miner_model_status.get(hotkey) is False:
            bt.logging.info(f"{hotkey} not running real model, score=0")
            return 0.0
        if not is_correct:
            bt.logging.info(f"{hotkey} incorrect, score=0")
            return 0.0
        return 1.0

    async def create_synapse(self) -> CatSoundProtocol:
        # 清分数交给 concurrent_forward 统一做
        if random.random() < 0.9:
            test_id, is_cat = self.select_test_sample()
            audio = self._create_special_test(test_id)
        else:
            test_id, is_cat = self.select_test_sample()
            audio = base64.b64encode(f"TEST:{test_id}".encode()).decode('utf-8')

        syn = CatSoundProtocol(audio_data=audio)
        syn.sample_id    = test_id
        syn.ground_truth = is_cat
        return syn

    async def concurrent_forward(self):
        # 清空上一轮分数
        self.scores.clear()
        try:
            tasks = [ self.forward(await self.create_synapse())
                      for _ in range(self.config.neuron.num_concurrent_forwards) ]
            await asyncio.gather(*tasks)

            self.test_round += 1
            if self.test_round % 10 == 0:
                self.save_state()

            # 上链设置权重
            self.set_weights()

        except Exception as e:
            bt.logging.error(f"concurrent_forward() error: {e}")

    def score(self, uid: int) -> float:
        return self.scores.get(uid, 0.0)


    def set_weights(self):
        """
        收集本轮分数，只对链上活跃矿工做归一化与缩放，
        保证非零权重之和 == 活跃矿工数量，非活跃矿工权重始终为 0。
        """
        try:
            n = self.metagraph.n
            active = self.metagraph.active.to(torch.bool)
            scores = torch.zeros(n)
            for uid in range(n):
                scores[uid] = self.score(uid)

            alive = scores[active]
            if alive.sum().item() == 0:
                scores[active] = 1.0
                alive = scores[active]

            scores[active] = alive / alive.sum()
            total_active = float(active.sum().item())
            scores[active] = scores[active] * total_active

            uids = torch.arange(n, dtype=torch.long)
            self.subtensor.set_weights(
                netuid = self.config.netuid,
                wallet = self.wallet,
                uids   = uids,
                weights= scores,
                wait_for_inclusion = True
            )
            bt.logging.info(f"设置权重完成：活跃矿工={int(total_active)}，权重之和={scores.sum().item():.3f}")
        except Exception as e:
            bt.logging.error(f"设置权重失败：{e}")


    def save_state(self):
        try:
            data = {
                "miner_history":       self.miner_history,
                "miner_model_status":  self.miner_model_status,
                "test_round":          self.test_round,
                "_special_aware_miners": list(self._special_aware_miners),
                "miner_dtao_balance":  self.miner_dtao_balance,
                "last_balance_check":  self.last_balance_check,
            }
            path = os.path.join(getattr(self.config.neuron, 'full_path', '.'), 'data')
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, 'validator_state.json'), 'w') as f:
                json.dump(data, f)
            bt.logging.info("Validator state saved")
        except Exception as e:
            bt.logging.error(f"save_state() error: {e}")

    def load_state(self):
        try:
            path = os.path.join(getattr(self.config.neuron, 'full_path', '.'), 'data', 'validator_state.json')
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                self.miner_history         = data.get("miner_history", {})
                self.miner_model_status    = data.get("miner_model_status", {})
                self.test_round            = data.get("test_round", 0)
                self._special_aware_miners = set(data.get("_special_aware_miners", []))
                self.miner_dtao_balance    = data.get("miner_dtao_balance", {})
                self.last_balance_check    = data.get("last_balance_check", {})
            bt.logging.info("Validator state loaded")
        except Exception as e:
            bt.logging.error(f"load_state() error: {e}")
        self.scores = {}


def get_config():
    import argparse
    parser = argparse.ArgumentParser()
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    parser.add_argument("--netuid", type=int, default=1, help="Subnet ID")
    parser.add_argument("--neuron.validation_interval", type=int, default=5, help="Seconds")
    parser.add_argument("--neuron.sample_size",        type=int, default=10, help="Samples/round")
    return bt.config(parser)


if __name__ == "__main__":
    config = get_config()
    config.neuron.full_path = os.path.expanduser(os.path.dirname(os.path.abspath(__file__)))
    bt.logging.info("Initializing validator...")
    validator = Validator(config=config)
    while True:
        bt.logging.info("Validator alive")
        time.sleep(config.neuron.validation_interval)
