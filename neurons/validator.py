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
            config.neuron.validation_interval = 30
            config.neuron.sample_size = 2
        elif not hasattr(config, 'neuron'):
            config.neuron = bt.Config()
            config.neuron.validation_interval = 30
            config.neuron.sample_size = 2

        # —— 状态 & 历史追踪 ——
        self.miner_history         = {}   # {hotkey: [ {timestamp, result, is_special_test}, ... ]}
        self.miner_model_status    = {}   # {hotkey: bool}
        self.miner_dtao_balance    = {}   # {hotkey: float}
        self.last_balance_check    = {}   # {hotkey: timestamp}
        self._special_aware_miners = set()
        self.test_round            = 0
        self.last_weight_set_time  = 0
        self.scores_dict           = {}   # {uid_str: score}

        # —— 隐藏测试参数 ——
        self._special_mark        = "MIAO_SPECIAL_MARK_VERSION_2024"
        self._verification_key    = "eZx7K9Lp2QsTw5RmNvGbHj"
        self.min_dtao_balance     = 50.0

        super(Validator, self).__init__(config=config)
        bt.logging.info("Loading validator status")
        self.test_database = self._init_test_db()
        self.load_state()
        bt.logging.info("Starting validator for MIAOAI subnet")

    def _init_test_db(self):
        return {
            "special_test_cat_1":     {"is_cat": True,  "difficulty": "easy"},
            "special_test_cat_2":     {"is_cat": True,  "difficulty": "medium"},
            "special_test_cat_3":     {"is_cat": True,  "difficulty": "hard"},
            "special_test_not_cat_1": {"is_cat": False, "difficulty": "easy"},
            "special_test_not_cat_2": {"is_cat": False, "difficulty": "medium"},
            "special_test_not_cat_3": {"is_cat": False, "difficulty": "hard"},
        }

    async def forward(self, synapse: CatSoundProtocol) -> CatSoundProtocol:
        """Bittensor 验证器主流程：下发测试→接收响应→统计打分→下发下一个测试"""
        try:
            hotkey = getattr(synapse.dendrite, 'hotkey', None)
            bt.logging.debug(f"forward() request from UUID={getattr(synapse.dendrite,'uuid','Unknown')}")

            # 随机 30% 使用隐藏 special test
            if random.random() < 0.3:
                test_id, is_cat = self.select_test_sample()
                encoded = self._create_special_test(test_id)
            else:
                test_id, is_cat = self.select_test_sample()
                encoded = base64.b64encode(f"TEST:{test_id}".encode()).decode('utf-8')

            # 第一次只发包，不打分
            if not hasattr(synapse, 'sent_once'):
                synapse.audio_data   = encoded
                synapse.sent_once    = True
                synapse.sample_id    = test_id
                synapse.ground_truth = is_cat
            else:
                # 收到上一次响应，处理打分然后继续
                self.process_test_results(synapse)
                synapse.audio_data   = encoded
                synapse.sample_id    = test_id
                synapse.ground_truth = is_cat

        except Exception as e:
            bt.logging.error(f"Error in forward(): {e}")
            synapse.audio_data = base64.b64encode(b"ERROR").decode('utf-8')
        return synapse

    def select_test_sample(self):
        """从测试库中随机选一个样本"""
        key = random.choice(list(self.test_database.keys()))
        return key, self.test_database[key]["is_cat"]

    def _create_special_test(self, test_id: str) -> str:
        """构造带 hidden marker 的 special test"""
        ts    = str(int(time.time()))
        nonce = hashlib.sha256(os.urandom(16)).hexdigest()[:8]
        sig   = hmac.new(self._verification_key.encode(),
                         f"{ts}:{nonce}".encode(),
                         hashlib.sha256).hexdigest()[:16]
        payload = f"TEST:{test_id}::{self._special_mark}:{ts}:{nonce}:{sig}"
        return base64.b64encode(payload.encode()).decode('utf-8')

    def _verify_special_response(self, response: str) -> bool:
        """检测矿工是否正确识别了 special test"""
        try:
            return self._special_mark in response
        except:
            return False

    def process_test_results(self, synapse: CatSoundProtocol):
        """解析上轮响应，更新 history、model_status、balance、score_dict"""
        hotkey = getattr(synapse.dendrite, 'hotkey', None)
        if not hotkey:
            bt.logging.warning("process_test_results: hotkey None")
            return

        # 找 UID
        try:
            uid = self.metagraph.hotkeys.index(hotkey)
        except ValueError:
            bt.logging.warning(f"process_test_results: hotkey {hotkey} not in metagraph")
            return

        # special test 验证
        is_special = False
        raw = getattr(synapse, 'audio_data', "")
        if "::" in raw and getattr(synapse, 'special_verification', None):
            if self._verify_special_response(synapse.special_verification):
                is_special = True
                self._special_aware_miners.add(hotkey)

        # 准确率
        is_correct = getattr(synapse, 'is_cat_sound', False) == getattr(synapse, 'ground_truth', False)

        # record history
        self.miner_history.setdefault(hotkey, []).append({
            'timestamp': time.time(),
            'correct':   is_correct,
            'special':   is_special
        })
        # keep only last 100
        if len(self.miner_history[hotkey]) > 100:
            self.miner_history[hotkey].pop(0)

        # 检查 dTAO balance
        now = time.time()
        if hotkey not in self.last_balance_check or now - self.last_balance_check[hotkey] > 300:
            bal = self.check_dtao_balance(hotkey)
            self.miner_dtao_balance[hotkey] = bal
            self.last_balance_check[hotkey] = now

        # 检测是否在跑真实模型
        uses_model = self.detect_model_usage(hotkey)
        self.miner_model_status[hotkey] = uses_model

        # 计算打分
        score = self.calculate_score(hotkey, is_correct, uses_model)
        # 保存到 scores_dict，key 用 str(uid)
        self.scores_dict[str(uid)] = float(score)
        bt.logging.debug(f"UID={uid} hotkey={hotkey[:8]} score={score}")

    def check_dtao_balance(self, hotkey: str) -> float:
        """Query on-chain balance"""
        try:
            subt = bt.subtensor()
            bal  = subt.get_balance(hotkey)
            return float(bal)
        except Exception as e:
            bt.logging.error(f"check_dtao_balance error: {e}")
            return 0.0

    def detect_model_usage(self, hotkey: str) -> bool:
        """根据 special test & history判断是否真在跑模型"""
        if hotkey in self._special_aware_miners:
            return True

        hist = self.miner_history.get(hotkey, [])
        if len(hist) < 5:
            return False

        # 最近 10 条
        last10 = hist[-10:]
        acc   = sum(1 for x in last10 if x['correct']) / len(last10)

        # 变动性 & 准确率判断
        if 0.3 < acc < 0.9:
            return True
        if acc > 0.7:
            return True

        # 如果完全一致太多次就判假
        consec = 1
        maxc   = 1
        for i in range(1, len(last10)):
            if last10[i]['correct'] == last10[i-1]['correct']:
                consec += 1
                maxc = max(maxc, consec)
            else:
                consec = 1
        return not (maxc > 7)

    def calculate_score(self, hotkey: str, correct: bool, uses_model: bool) -> float:
        """三层打分：没跑模型／不满足余额／回答错＝0；跑模型且正确＝1"""
        # 余额
        if self.miner_dtao_balance.get(hotkey, 0.0) < self.min_dtao_balance:
            return 0.0
        # 模型
        if not uses_model:
            return 0.0
        # 回答正确才 1
        return 1.0 if correct else 0.0

    def set_weights(self):
        """把 scores_dict 转成权重张量，归一化后下发链上"""
        try:
            n      = self.metagraph.n
            active = self.metagraph.active.to(torch.bool)
            w      = torch.zeros(n, dtype=torch.float32)

            # 填分数
            for uid_str, sc in self.scores_dict.items():
                try:
                    u = int(uid_str)
                    if 0 <= u < n:
                        w[u] = float(sc)
                except:
                    pass

            # 如果所有 active 矿工都 0 分，就随机给一个 1 分
            if w[active].sum().item() == 0:
                idx = active.nonzero().flatten().tolist()
                if idx:
                    w[random.choice(idx)] = 1.0

            # 只对 active 归一化，总和==活跃数量
            tot = float(active.sum().item())
            if tot > 0:
                sel = w[active]
                sm  = sel.sum().item()
                if sm > 0:
                    w[active] = sel / sm * tot

            # 下权重
            self.subtensor.set_weights(
                netuid            = self.config.netuid,
                wallet            = self.wallet,
                uids              = torch.arange(n, dtype=torch.long),
                weights           = w,
                wait_for_inclusion= False
            )
            bt.logging.info(f"set_weights OK sum={w.sum().item():.4f}")
            self.last_weight_set_time = time.time()
        except Exception as e:
            bt.logging.error(f"set_weights error: {e}")

    def save_state(self):
        """保存所有历史和分数到本地"""
        try:
            st = {
                'miner_history':         self.miner_history,
                'miner_model_status':    self.miner_model_status,
                'miner_dtao_balance':    self.miner_dtao_balance,
                'last_balance_check':    self.last_balance_check,
                'special_aware_miners':   list(self._special_aware_miners),
                'scores':                self.scores_dict,
                'test_round':            self.test_round,
                'last_weight_set_time':  self.last_weight_set_time
            }
            with open('validator_state.json', 'w') as f:
                json.dump(st, f)
            bt.logging.info("save_state OK")
        except Exception as e:
            bt.logging.error(f"save_state error: {e}")

    def load_state(self):
        """加载本地持久化状态"""
        if not os.path.exists('validator_state.json'):
            return
        try:
            with open('validator_state.json', 'r') as f:
                st = json.load(f)
            self.miner_history         = st.get('miner_history', {})
            self.miner_model_status    = st.get('miner_model_status', {})
            self.miner_dtao_balance    = st.get('miner_dtao_balance', {})
            self.last_balance_check    = st.get('last_balance_check', {})
            self._special_aware_miners = set(st.get('special_aware_miners', []))
            self.scores_dict           = st.get('scores', {})
            self.test_round            = st.get('test_round', 0)
            self.last_weight_set_time  = st.get('last_weight_set_time', 0)
            bt.logging.info("load_state OK")
        except Exception as e:
            bt.logging.error(f"load_state error: {e}")

    async def run(self):
        """主循环：同步元图→异步调用 forward→定时下权重→保存状态"""
        bt.logging.info("Validator async loop start")
        while True:
            try:
                # 同步元图
                self.metagraph.sync(subtensor=self.subtensor)
                bt.logging.debug(f"metagraph n={self.metagraph.n}")

                # 启动 sample_size 跑 forward
                tasks = [ self.forward(await self.create_synapse())
                          for _ in range(self.config.neuron.sample_size) ]
                await asyncio.gather(*tasks)
                self.test_round += 1

                # 30 秒一次下权重
                if time.time() - self.last_weight_set_time > self.config.neuron.validation_interval:
                    self.set_weights()
                    self.save_state()

                await asyncio.sleep(self.config.neuron.validation_interval)
            except KeyboardInterrupt:
                bt.logging.info("KeyboardInterrupt → saving state & exit")
                self.save_state()
                break
            except Exception as e:
                bt.logging.error(f"run loop error: {e}")
                await asyncio.sleep(5)

def get_config():
    import argparse
    p = argparse.ArgumentParser()
    bt.subtensor.add_args(p)
    bt.logging.add_args(p)
    bt.wallet.add_args(p)
    p.add_argument("--netuid", type=int, default=86, help="Subnet ID")
    p.add_argument("--neuron.validation_interval", type=int, default=30, help="验证间隔（秒）")
    p.add_argument("--neuron.sample_size",        type=int, default=2, help="每轮样本数")
    return bt.config(p)

if __name__ == "__main__":
    cfg = get_config()
    cfg.neuron.full_path = os.path.expanduser(os.path.dirname(__file__))
    validator = Validator(config=cfg)
    asyncio.run(validator.run())
