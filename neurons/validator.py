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
            config.neuron.validation_interval = 60  # 更长的验证间隔，避免过于频繁
            config.neuron.sample_size = 2
        elif not hasattr(config, 'neuron'):
            config.neuron = bt.Config()
            config.neuron.validation_interval = 60  # 更长的验证间隔，避免过于频繁
            config.neuron.sample_size = 2

        # 初始化追踪变量
        self.miner_history = {}
        self.miner_model_status = {}
        self.test_round = 0
        self._special_mark = "MIAO_SPECIAL_MARK_VERSION_2024"
        self._verification_key = "eZx7K9Lp2QsTw5RmNvGbHj"
        self._special_aware_miners = set()
        self.min_dtao_balance = 50.0
        self.miner_dtao_balance = {}
        self.last_balance_check = {}
        self.scores_dict = {}  # {uid: score}
        self.last_weight_set_time = 0
        self.is_running_step = False  # 防止重复运行的标志

        super(Validator, self).__init__(config=config)
        bt.logging.info("Loading validator status")

        self.test_database = self.initialize_test_database()
        self.load_state()
        bt.logging.info("Starting validator for MIAOAI subnet (zero-score for non-model miners)")

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
            bt.logging.debug(f"Validation request from UUID={getattr(synapse.dendrite, 'uuid', 'Unknown')}")

            if random.random() < 0.3:  # 30% chance of special test
                test_id, is_cat = self.select_test_sample()
                encoded = self._create_special_test(test_id)
            else:
                test_id, is_cat = self.select_test_sample()
                encoded = base64.b64encode(f"TEST:{test_id}".encode()).decode('utf-8')

            if not hasattr(synapse, 'sent_once'):
                synapse.audio_data    = encoded
                synapse.sent_once     = True
                synapse.sample_id     = test_id
                synapse.ground_truth  = is_cat
            else:
                self.process_test_results(synapse)
                synapse.audio_data    = encoded
                synapse.sample_id     = test_id
                synapse.ground_truth  = is_cat

        except Exception as e:
            bt.logging.error(f"Error in forward: {e}")
            synapse.audio_data = "ERROR"

        return synapse

    def select_test_sample(self):
        test_id = random.choice(list(self.test_database.keys()))
        return test_id, self.test_database[test_id]["is_cat"]

    def _create_special_test(self, test_id):
        # Add special marker for testing if miners run real model
        special_data = f"TEST:{test_id}:{self._special_mark}"
        return base64.b64encode(special_data.encode()).decode('utf-8')

    def _verify_special_response(self, response, expected_id):
        if not response or not isinstance(response, str):
            return False
        
        try:
            # Check if response contains awareness of special marker
            return self._special_mark in response
        except:
            return False

    def process_test_results(self, synapse):
        hotkey = getattr(synapse.dendrite, 'hotkey', None)
        if not hotkey:
            bt.logging.warning("No hotkey found in synapse")
            return
            
        uid = None
        try:
            uid = self.metagraph.hotkeys.index(hotkey)
        except ValueError:
            bt.logging.warning(f"Hotkey {hotkey} not found in metagraph")
            return

        bt.logging.debug(f"Processing results from UID={uid}")
        
        # Process response based on test type
        if synapse.sample_id:
            expected_result = synapse.ground_truth
            
            # Check if this was a special test
            is_special_test = False
            if hasattr(synapse, 'response') and synapse.response:
                if hasattr(synapse.response, 'hidden_marker_response'):
                    hidden_response = synapse.response.hidden_marker_response
                    if hidden_response:
                        is_special_test = self._verify_special_response(hidden_response, synapse.sample_id)
                        if is_special_test:
                            bt.logging.info(f"Miner {uid} passed special test")
                            if uid is not None:
                                self._special_aware_miners.add(uid)
            
            # Record prediction accuracy
            if hasattr(synapse, 'response') and synapse.response:
                if hasattr(synapse.response, 'predictions') and synapse.response.predictions:
                    pred = synapse.response.predictions[0] if isinstance(synapse.response.predictions, list) and len(synapse.response.predictions) > 0 else None
                    if pred and isinstance(pred, dict) and 'is_cat' in pred:
                        result = pred['is_cat'] == expected_result
                        
                        # Track miner response history
                        if uid not in self.miner_history:
                            self.miner_history[uid] = []
                        
                        self.miner_history[uid].append({
                            'timestamp': time.time(),
                            'result': result,
                            'is_special_test': is_special_test
                        })
                        
                        # Check if the miner is using a real model
                        model_usage = self.detect_model_usage(uid)
                        self.miner_model_status[uid] = model_usage
                        
                        # Check dTAO balance periodically (every 300 seconds)
                        current_time = time.time()
                        if uid not in self.last_balance_check or (current_time - self.last_balance_check.get(uid, 0)) > 300:
                            balance = self.check_dtao_balance(hotkey)
                            self.miner_dtao_balance[uid] = balance
                            self.last_balance_check[uid] = current_time
                            bt.logging.debug(f"Updated dTAO balance for UID={uid}: {balance}")
                            
                        # Calculate score
                        score = self.calculate_score(uid, result)
                        self.scores_dict[str(uid)] = float(score)  # Store score in dictionary format
                        
                        bt.logging.info(f"Score for UID={uid}: {score}")
                    else:
                        # 预测格式无效，设为0分
                        self.scores_dict[str(uid)] = 0.0
                        bt.logging.info(f"Invalid prediction format from UID={uid}, setting score to 0.0")
                else:
                    # 没有预测结果，设为0分
                    self.scores_dict[str(uid)] = 0.0
                    bt.logging.info(f"No predictions from UID={uid}, setting score to 0.0")
            else:
                # 没有响应，设为0分
                bt.logging.warning(f"No response from miner {uid}")
                self.scores_dict[str(uid)] = 0.0
                bt.logging.info(f"Set score to 0.0 for non-responsive miner UID={uid}")

    def detect_model_usage(self, uid):
        # Check if miner is aware of special tests
        if uid in self._special_aware_miners:
            return True
            
        # Check recent history
        if uid in self.miner_history:
            history = self.miner_history[uid]
            
            # Only consider miners with sufficient history
            if len(history) >= 5:
                # Check if miner has consistent performance
                results = [entry['result'] for entry in history[-10:]]
                if sum(results) / len(results) > 0.7:
                    # Miners with consistently good performance likely run real models
                    return True
                
                # Check if there's variance in success rate
                if 0.3 < sum(results) / len(results) < 0.9:
                    # Some variance indicates real model behavior
                    return True
                    
                # Check for patterns that indicate artificial responses
                consecutive_same = 1
                max_consecutive = 1
                for i in range(1, len(results)):
                    if results[i] == results[i-1]:
                        consecutive_same += 1
                        max_consecutive = max(max_consecutive, consecutive_same)
                    else:
                        consecutive_same = 1
                
                # Long streaks of identical results suggest fake responses
                if max_consecutive > 7:
                    return False
                    
        # Default to assuming model is NOT running
        return False

    def check_dtao_balance(self, hotkey):
        try:
            # Get the actual dTAO balance from the blockchain
            subtensor = bt.subtensor()
            balance = subtensor.get_balance(hotkey)
            bt.logging.debug(f"Checked real dTAO balance for {hotkey}: {balance}")
            return float(balance)
        except Exception as e:
            bt.logging.error(f"Error checking dTAO balance: {e}")
            return 0.0

    def calculate_score(self, uid, result):
        # 修改评分系统，确保不工作的矿工得分为0
        # 0.0 - 作弊、无响应或不满足dTAO余额要求的矿工，或未运行模型的矿工
        # 0.8-1.0 - 运行真实模型并根据性能调整的矿工
        
        # 检查矿工是否有足够的历史记录
        if uid not in self.miner_history or len(self.miner_history[uid]) < 3:
            bt.logging.info(f"UID={uid} has insufficient history, setting score to 0.0")
            return 0.0
            
        # 检查dTAO余额要求
        if uid in self.miner_dtao_balance and self.miner_dtao_balance[uid] < self.min_dtao_balance:
            bt.logging.info(f"UID={uid} has insufficient dTAO balance ({self.miner_dtao_balance[uid]} < {self.min_dtao_balance})")
            return 0.0
            
        # 检查矿工是否运行真实模型
        is_running_model = self.miner_model_status.get(uid, False)
        
        if not is_running_model:
            # 不运行真实模型的矿工得分为0，不再是0.5
            bt.logging.info(f"UID={uid} is not running a real model, setting score to 0.0")
            return 0.0
            
        # 对于运行模型的矿工，根据性能调整分数
        recent_results = [entry['result'] for entry in self.miner_history[uid][-10:]]
        accuracy = sum(recent_results) / len(recent_results)
        
        # 性能调整：0.8到1.0基于准确率
        performance_score = 0.8 + 0.2 * accuracy
        
        bt.logging.info(f"UID={uid} is running a real model with accuracy {accuracy}, setting score to {performance_score}")
        return float(performance_score)

    def save_state(self):
        try:
            state = {
                'miner_history': self.miner_history,
                'miner_model_status': self.miner_model_status,
                'special_aware_miners': list(self._special_aware_miners),
                'miner_dtao_balance': self.miner_dtao_balance,
                'last_balance_check': self.last_balance_check,
                'scores': self.scores_dict,  # Save scores as dictionary
                'test_round': self.test_round,
                'last_weight_set_time': self.last_weight_set_time
            }
            
            with open('validator_state.json', 'w') as f:
                json.dump(state, f)
                
            bt.logging.info("Validator state saved successfully")
        except Exception as e:
            bt.logging.error(f"Error saving validator state: {e}")

    def load_state(self):
        try:
            if os.path.exists('validator_state.json'):
                with open('validator_state.json', 'r') as f:
                    state = json.load(f)
                    
                self.miner_history = state.get('miner_history', {})
                self.miner_model_status = state.get('miner_model_status', {})
                self._special_aware_miners = set(state.get('special_aware_miners', []))
                self.miner_dtao_balance = state.get('miner_dtao_balance', {})
                self.last_balance_check = state.get('last_balance_check', {})
                
                # 保证scores_dict键值都是字符串
                scores = state.get('scores', {})
                self.scores_dict = {str(k): float(v) for k, v in scores.items()}
                
                self.test_round = state.get('test_round', 0)
                self.last_weight_set_time = state.get('last_weight_set_time', 0)
                
                bt.logging.info("Validator state loaded successfully")
        except Exception as e:
            bt.logging.error(f"Error loading validator state: {e}")

    def set_weights(self):
        """修改后的set_weights方法，直接使用计算好的分数，不做归一化"""
        try:
            # 获取矿工数量
            n = self.metagraph.n
            
            # 创建权重张量
            weights = torch.zeros(n)
            
            # 直接从scores_dict填充权重，不做归一化
            for uid_str, score in self.scores_dict.items():
                try:
                    uid = int(uid_str)
                    if 0 <= uid < n:
                        weights[uid] = float(score)
                except (ValueError, TypeError) as e:
                    bt.logging.error(f"Error converting UID {uid_str}: {e}")
            
            # 打印详细的权重信息，帮助调试
            zero_count = torch.sum(weights == 0).item()
            non_zero_count = torch.sum(weights > 0).item()
            bt.logging.info(f"Setting direct weights: zeros={zero_count}, non-zeros={non_zero_count}")
            
            # 打印部分非零分数
            non_zero_uids = [uid for uid in range(n) if weights[uid] > 0]
            if non_zero_uids:
                sample_uids = random.sample(non_zero_uids, min(5, len(non_zero_uids)))
                for uid in sample_uids:
                    bt.logging.info(f"Sample weight - UID={uid}: {weights[uid].item()}")
            
            # 非活跃矿工强制设为0
            active = self.metagraph.active.to(torch.bool)
            weights[~active] = 0.0
            
            # 如果所有权重都是0，给一个随机矿工设置权重，避免全0
            if torch.sum(weights) == 0:
                if torch.sum(active) > 0:
                    # 从活跃矿工中随机选择一个
                    active_uids = torch.nonzero(active).squeeze()
                    random_uid = active_uids[random.randint(0, len(active_uids)-1)]
                    weights[random_uid] = 1.0
                    bt.logging.info(f"All weights are zero, setting random active miner (UID={random_uid}) to 1.0")
                else:
                    # 如果没有活跃矿工，随机选择一个
                    random_uid = random.randint(0, n-1)
                    weights[random_uid] = 1.0
                    bt.logging.info(f"No active miners, setting random miner (UID={random_uid}) to 1.0")
            
            # 直接设置权重，不做归一化
            bt.logging.info(f"Setting weights on chain (sum={weights.sum().item():.4f})")
            result = self.subtensor.set_weights(
                netuid=self.config.netuid,
                wallet=self.wallet,
                uids=torch.arange(0, n),
                weights=weights,
                wait_for_inclusion=False
            )
            
            if result:
                bt.logging.info(f"Successfully set weights directly")
                self.last_weight_set_time = time.time()
            else:
                bt.logging.error("Failed to set weights")
                
        except Exception as e:
            bt.logging.error(f"Error in set_weights: {str(e)}")
            # 即使发生错误，也更新时间，避免频繁重试
            self.last_weight_set_time = time.time()

    async def run_step(self):
        """改进后的run_step方法，更加稳定"""
        # 如果已经在运行，跳过
        if self.is_running_step:
            await asyncio.sleep(5)
            return
        
        try:
            # 设置运行标志
            self.is_running_step = True
            
            # 在开始添加短暂休眠，避免立即重试
            await asyncio.sleep(2)
            
            # 同步元图
            bt.logging.info(f"Syncing metagraph...")
            self.metagraph.sync(subtensor=self.subtensor)
            bt.logging.info(f"Metagraph updated with {self.metagraph.n} total miners")
            
            # 控制权重设置频率
            current_time = time.time()
            should_set_weights = (current_time - self.last_weight_set_time) > 1800  # 30分钟
            
            if should_set_weights:
                bt.logging.info("Setting weights due to time interval")
                self.set_weights()  # 使用修改后的set_weights方法
                self.save_state()
            elif self.test_round % 10 == 0:
                self.save_state()
                
            # 增加计数
            self.test_round += 1
            
            # 查询矿工
            await self.query_miners()
            
            # 结束时较长的休眠，避免频繁循环
            await asyncio.sleep(max(30, self.config.neuron.validation_interval))
            
        except Exception as e:
            bt.logging.error(f"Error in run_step: {e}")
            # 错误后更长的休眠
            await asyncio.sleep(60)
        finally:
            # 清除运行标志
            self.is_running_step = False

    async def query_miners(self):
        # 查询前确保有足够的矿工可以抽样
        if self.metagraph.n <= 0:
            bt.logging.warning("No miners available in the metagraph")
            return
            
        # 选择sample_size个随机矿工进行查询
        sample_size = min(self.config.neuron.sample_size, self.metagraph.n)
        uids = random.sample(range(self.metagraph.n), sample_size)
        
        bt.logging.info(f"Querying {len(uids)} random miners")
        
        for uid in uids:
            try:
                # 获取矿工端点
                axon = self.metagraph.axons[uid]
                
                # 创建带有测试样本的synapse
                test_id, is_cat = self.select_test_sample()
                
                if random.random() < 0.3:  # 30%特殊测试概率
                    encoded = self._create_special_test(test_id)
                else:
                    encoded = base64.b64encode(f"TEST:{test_id}".encode()).decode('utf-8')
                
                synapse = CatSoundProtocol(
                    audio_data=encoded,
                    sample_id=test_id,
                    ground_truth=is_cat
                )
                
                # 查询矿工
                bt.logging.info(f"Querying UID={uid}")
                
                # 设置超时，防止无响应的矿工导致程序卡住
                try:
                    response = await asyncio.wait_for(
                        self.dendrite.forward(axon, synapse, deserialize=True),
                        timeout=10.0  # 10秒超时
                    )
                except asyncio.TimeoutError:
                    bt.logging.warning(f"Query to UID={uid} timed out")
                    # 为超时的矿工设置0分
                    self.scores_dict[str(uid)] = 0.0
                    continue
                
                # 处理结果
                if response and hasattr(response, 'predictions'):
                    bt.logging.info(f"Got response from UID={uid}")
                    
                    # 处理响应
                    pred = response.predictions[0] if isinstance(response.predictions, list) and len(response.predictions) > 0 else None
                    
                    if pred and isinstance(pred, dict) and 'is_cat' in pred:
                        result = pred['is_cat'] == is_cat
                        
                        # 更新矿工历史
                        if uid not in self.miner_history:
                            self.miner_history[uid] = []
                            
                        # 检查是否为特殊测试
                        is_special_test = False
                        if hasattr(response, 'hidden_marker_response') and response.hidden_marker_response:
                            is_special_test = self._verify_special_response(response.hidden_marker_response, test_id)
                            if is_special_test:
                                bt.logging.info(f"Miner {uid} passed special test")
                                self._special_aware_miners.add(uid)
                        
                        # 记录结果
                        self.miner_history[uid].append({
                            'timestamp': time.time(),
                            'result': result,
                            'is_special_test': is_special_test
                        })
                        
                        # 更新模型状态
                        model_usage = self.detect_model_usage(uid)
                        self.miner_model_status[uid] = model_usage
                        
                        # 检查dTAO余额
                        hotkey = self.metagraph.hotkeys[uid]
                        current_time = time.time()
                        if uid not in self.last_balance_check or (current_time - self.last_balance_check.get(uid, 0)) > 300:
                            balance = self.check_dtao_balance(hotkey)
                            self.miner_dtao_balance[uid] = balance
                            self.last_balance_check[uid] = current_time
                            
                        # 计算分数
                        score = self.calculate_score(uid, result)
                        self.scores_dict[str(uid)] = float(score)
                        
                        bt.logging.info(f"Score for UID={uid}: {score}")
                    else:
                        # 无有效预测，设置分数为0
                        bt.logging.warning(f"Invalid prediction from UID={uid}")
                        self.scores_dict[str(uid)] = 0.0
                else:
                    # 无响应，设置分数为0
                    bt.logging.warning(f"No valid response from UID={uid}")
                    self.scores_dict[str(uid)] = 0.0
                
            except Exception as e:
                bt.logging.error(f"Error querying miner {uid}: {e}")
                # 查询出错的矿工得分为0
                self.scores_dict[str(uid)] = 0.0

def get_config():
    parser = argparse.ArgumentParser()
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    parser.add_argument('--netuid', type=int, default=86, help="Subnet UID")
    parser.add_argument('--neuron.validation_interval', type=int, default=60, help="Validation interval in seconds")
    parser.add_argument('--neuron.sample_size', type=int, default=2, help="Number of miners to sample each round")
    return bt.config(parser)

if __name__ == "__main__":
    # 确保导入argparse
    import argparse
    
    # 获取配置
    config = get_config()
    
    # 设置完整路径
    config.neuron.full_path = os.path.expanduser(os.path.dirname(os.path.abspath(__file__)))
    
    # 创建并运行验证器
    validator = Validator(config=config)
    
    # 使用标准启动方法
    validator.run()
