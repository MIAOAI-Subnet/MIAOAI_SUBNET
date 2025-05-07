# MIT License
# Copyright © 2025 MIAO

import time
import random
import base64
import os
import asyncio
import bittensor as bt
import torch
from template.base.validator import BaseValidatorNeuron
from template.protocol import CatSoundProtocol

def get_config():
    import argparse
    parser = argparse.ArgumentParser()
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    parser.add_argument("--netuid", type=int, default=86, help="Subnet ID")
    parser.add_argument("--neuron.validation_interval", type=int, default=5, help="验证间隔（秒）")
    parser.add_argument("--neuron.sample_size",        type=int, default=10, help="每轮并发样本数")
    return bt.config(parser)

class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        # 如果外部不传 config，用 get_config() 
        if config is None:
            config = get_config()
        super(Validator, self).__init__(config=config)

        # 分数表：uid -> score
        self.scores = {}
        # dTAO 最低余额要求
        self.min_dtao_balance = 50.0
        # 用于生成测试样本
        self.test_database = [
            ("cat_easy",     True),
            ("cat_medium",   True),
            ("cat_hard",     True),
            ("not_cat_easy", False),
            ("not_cat_med",  False),
            ("not_cat_hard", False),
        ]
        self._state = {}
        self.load_state()
        bt.logging.info("Validator 初始化完成")

    def select_test_sample(self):
        # 随机挑选一条测试（仅 ID 和 ground_truth）
        tid, is_cat = random.choice(self.test_database)
        return tid, is_cat

    def _encode_test(self, test_id: str, special: bool):
        payload = f"{'SPECIAL:' if special else ''}TEST:{test_id}"
        return base64.b64encode(payload.encode()).decode("utf-8")

    async def create_synapse(self) -> CatSoundProtocol:
        """创建一个新的synapse对象，供concurrent_forward使用"""
        synapse = CatSoundProtocol()
        # 随机 20% special test（可检测作弊）
        special = (random.random() < 0.2)
        tid, is_cat = self.select_test_sample()
        encoded = self._encode_test(tid, special)
        
        synapse.audio_data = encoded
        synapse.sample_id = tid
        synapse.ground_truth = is_cat
        
        return synapse

    async def forward(self, synapse: CatSoundProtocol) -> CatSoundProtocol:
        """
        1) 第一次调用：打包 test 发给矿工；
        2) 再次调用：收集上次结果，score -> self.scores，然后发新包。
        """
        try:
            # 如果没有hotkey，我们创建一个新的synapse
            if not hasattr(synapse, 'dendrite') or not hasattr(synapse.dendrite, 'hotkey'):
                # 这是一个新的请求，随机选择一个矿工
                if self.metagraph.n == 0:
                    bt.logging.warning("没有找到矿工")
                    return synapse
                    
                # 随机选择一个活跃矿工
                active_uids = torch.where(self.metagraph.active)[0].tolist()
                if not active_uids:
                    bt.logging.warning("没有活跃矿工")
                    return synapse
                    
                uid = random.choice(active_uids)
                axon = self.metagraph.axons[uid]
                
                # 使用dendrite转发请求
                try:
                    # 设置超时
                    response = await asyncio.wait_for(
                        self.dendrite.forward(axon, synapse, deserialize=True),
                        timeout=10.0
                    )
                    return response
                except asyncio.TimeoutError:
                    bt.logging.warning(f"查询 UID={uid} 超时")
                    self.scores[uid] = 0.0
                    return synapse
                except Exception as e:
                    bt.logging.error(f"查询 UID={uid} 错误: {e}")
                    self.scores[uid] = 0.0
                    return synapse
            
            # 已经收到响应，处理结果
            hotkey = getattr(synapse.dendrite, 'hotkey', None)
            uuid = getattr(synapse.dendrite, 'uuid', 'Unknown')
            bt.logging.debug(f"收到验证请求 UUID={uuid}")
            
            # 处理响应
            try:
                uid = self.metagraph.hotkeys.index(hotkey)
            except ValueError:
                bt.logging.warning(f"未在 metagraph 中找到 hotkey={hotkey}")
                return synapse
                
            # 获取响应并检查结果
            if hasattr(synapse, 'response') and hasattr(synapse.response, 'predictions'):
                preds = synapse.response.predictions
                if isinstance(preds, list) and len(preds) > 0:
                    pred = preds[0]
                    if isinstance(pred, dict) and 'is_cat' in pred:
                        is_cat_sound = pred['is_cat']
                        correct = (is_cat_sound == synapse.ground_truth)
                        
                        # 检查余额
                        bal = float(self.metagraph.S[uid]) * 1000
                        if bal < self.min_dtao_balance:
                            bt.logging.info(f"UID={uid} 余额不足 {bal:.1f} < {self.min_dtao_balance}")
                            score = 0.0
                        else:
                            score = 1.0 if correct else 0.0
                        self.scores[uid] = score
                        bt.logging.info(f"UID={uid} 验证 {'通过' if correct else '失败'}，得分={score}")
            else:
                bt.logging.warning(f"无效响应 from UID={uid}")
                self.scores[uid] = 0.0
                
            # 准备下一次请求的synapse
            special = (random.random() < 0.2)
            tid, is_cat = self.select_test_sample()
            encoded = self._encode_test(tid, special)
            
            synapse.audio_data = encoded
            synapse.sample_id = tid
            synapse.ground_truth = is_cat
            
            return synapse
            
        except Exception as e:
            bt.logging.error(f"forward错误: {e}")
            return synapse

    async def concurrent_forward(self):
        """
        每一轮：
        1) 清空 scores
        2) 自动 fan-out N=lambda(self.config.neuron.sample_size) 个 forward() 请求
        3) 等待所有完成后，调用 set_weights()
        4) 每 10 轮持久化一次状态
        """
        self.scores.clear()
        
        # 创建forward请求列表
        n = min(self.config.neuron.sample_size, self.metagraph.n)
        if n <= 0:
            bt.logging.warning("没有矿工可以查询")
            return
            
        # 随机选择n个不同的活跃矿工
        active_uids = torch.where(self.metagraph.active)[0].tolist()
        if not active_uids:
            bt.logging.warning("没有活跃矿工")
            return
            
        selected_uids = random.sample(active_uids, min(n, len(active_uids)))
        
        # 创建tasks
        tasks = []
        for uid in selected_uids:
            synapse = await self.create_synapse()
            axon = self.metagraph.axons[uid]
            
            # 添加超时处理
            try:
                bt.logging.info(f"查询 UID={uid}")
                task = asyncio.create_task(
                    asyncio.wait_for(
                        self.dendrite.forward(axon, synapse, deserialize=True),
                        timeout=10.0
                    )
                )
                tasks.append(task)
            except Exception as e:
                bt.logging.error(f"创建查询任务错误 UID={uid}: {e}")
                self.scores[uid] = 0.0
        
        # 并发执行所有请求
        if tasks:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            for i, response in enumerate(responses):
                uid = selected_uids[i]
                if isinstance(response, Exception):
                    bt.logging.warning(f"查询 UID={uid} 失败: {response}")
                    self.scores[uid] = 0.0
                    continue
                    
                # 处理正常响应
                if hasattr(response, 'predictions'):
                    preds = response.predictions
                    if isinstance(preds, list) and len(preds) > 0:
                        pred = preds[0]
                        if isinstance(pred, dict) and 'is_cat' in pred:
                            is_cat_sound = pred['is_cat']
                            sample_id = getattr(response, 'sample_id', None)
                            ground_truth = None
                            
                            # 查找对应的ground_truth
                            for sample in self.test_database:
                                if sample[0] == sample_id:
                                    ground_truth = sample[1]
                                    break
                                    
                            if ground_truth is not None:
                                correct = (is_cat_sound == ground_truth)
                                
                                # 检查余额
                                bal = float(self.metagraph.S[uid]) * 1000
                                if bal < self.min_dtao_balance:
                                    bt.logging.info(f"UID={uid} 余额不足 {bal:.1f} < {self.min_dtao_balance}")
                                    score = 0.0
                                else:
                                    score = 1.0 if correct else 0.0
                                self.scores[uid] = score
                                bt.logging.info(f"UID={uid} 验证 {'通过' if correct else '失败'}，得分={score}")
                            else:
                                bt.logging.warning(f"无法找到 sample_id={sample_id} 的ground_truth")
                                self.scores[uid] = 0.0
                        else:
                            bt.logging.warning(f"无效预测格式 from UID={uid}")
                            self.scores[uid] = 0.0
                    else:
                        bt.logging.warning(f"无效预测列表 from UID={uid}")
                        self.scores[uid] = 0.0
                else:
                    bt.logging.warning(f"无效响应 from UID={uid}")
                    self.scores[uid] = 0.0
        
        # 设置权重
        self.set_weights()
        
        # 每10轮保存一次状态
        if self._state.get("round", 0) % 10 == 0:
            self.save_state()
        self._state["round"] = self._state.get("round", 0) + 1

    def set_weights(self):
        """
        Bittensor 要求：
          ∑ weights[active_uids] == #active_uids
        我们将 0 分者权重设 0，得分为 1 的矿工均匀分配权重。
        """
        try:
            n      = self.metagraph.n
            active = self.metagraph.active.to(torch.bool)
            w      = torch.zeros(n, dtype=torch.float32)

            # 填分
            for uid, score in self.scores.items():
                if 0 <= uid < n and active[uid]:
                    w[uid] = float(score)

            # 若全为 0，给每个活跃矿工平均 1.0
            if w[active].sum().item() == 0:
                w[active] = 1.0

            # 归一化：sum(active_w) == 活跃矿工数
            total_active = float(active.sum().item())
            if total_active > 0 and w[active].sum().item() > 0:
                w[active] = w[active] / w[active].sum() * total_active

            # 上链
            bt.logging.info(f"正在上链设置权重，活跃矿工={int(total_active)}，权重和={w.sum().item():.3f}")
            
            # 打印一些非零权重样本
            non_zeros = [(i, w[i].item()) for i in range(n) if w[i] > 0]
            if non_zeros:
                samples = random.sample(non_zeros, min(5, len(non_zeros)))
                for uid, weight in samples:
                    bt.logging.info(f"样本权重 UID={uid}: {weight:.4f}")
            
            result = self.subtensor.set_weights(
                netuid             = self.config.netuid,
                wallet             = self.wallet,
                uids               = torch.arange(n, dtype=torch.long),
                weights            = w,
                wait_for_inclusion = True
            )
            
            if result:
                bt.logging.info(f"权重设置成功")
            else:
                bt.logging.error(f"权重设置失败")
        except Exception as e:
            bt.logging.error(f"set_weights 错误: {e}")

    def save_state(self):
        try:
            state = {
                "scores":      self.scores,
                "round":       self._state.get("round", 0),
            }
            path = os.path.join(self.config.neuron.full_path, "validator_state.json")
            with open(path, "w") as f:
                import json
                json.dump(state, f)
            bt.logging.info("Validator state 已保存")
        except Exception as e:
            bt.logging.error(f"save_state 错误: {e}")

    def load_state(self):
        try:
            self._state = {}
            path = os.path.join(self.config.neuron.full_path, "validator_state.json")
            if os.path.isfile(path):
                with open(path, "r") as f:
                    import json
                    self._state = json.load(f)
                bt.logging.info("Validator state 已加载")
        except Exception as e:
            bt.logging.error(f"load_state 错误: {e}")

if __name__ == "__main__":
    config = get_config()
    config.neuron.full_path = os.path.expanduser(os.path.dirname(os.path.abspath(__file__)))
    validator = Validator(config=config)
    validator.run()
