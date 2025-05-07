# MIT License
# Copyright © 2025 MIAO

import time
import random
import base64
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
        bt.logging.info("Validator 初始化完成")

    def select_test_sample(self):
        # 随机挑选一条测试（仅 ID 和 ground_truth）
        tid, is_cat = random.choice(self.test_database)
        return tid, is_cat

    def _encode_test(self, test_id: str, special: bool):
        payload = f"{'SPECIAL:' if special else ''}TEST:{test_id}"
        return base64.b64encode(payload.encode()).decode("utf-8")

    async def forward(self, synapse: CatSoundProtocol) -> CatSoundProtocol:
        """
        1) 第一次调用：打包 test 发给矿工；
        2) 再次调用：收集上次结果，score -> self.scores，然后发新包。
        """
        hotkey = getattr(synapse.dendrite, 'hotkey', None)
        uuid   = getattr(synapse.dendrite, 'uuid',   'Unknown')
        bt.logging.debug(f"收到验证请求 UUID={uuid}")

        # 随机 20% special test（可检测作弊）
        special = (random.random() < 0.2)
        tid, is_cat = self.select_test_sample()
        encoded = self._encode_test(tid, special)

        if not hasattr(synapse, "sent_once"):
            # 第一次发包
            synapse.audio_data   = encoded
            synapse.sent_once    = True
            synapse.sample_id    = tid
            synapse.ground_truth = is_cat
        else:
            # 已经有响应，处理上次结果
            # 注意：CatSoundProtocol 会把返回字段填在 synapse 上
            # 假设 synapse.is_cat_sound 为 bool，response_time 为 float
            correct = getattr(synapse, "is_cat_sound", None) == synapse.ground_truth
            try:
                uid = self.metagraph.hotkeys.index(hotkey)
            except ValueError:
                bt.logging.warning(f"未在 metagraph 中找到 hotkey={hotkey}")
            else:
                # 检查余额
                bal = float(self.metagraph.S[uid]) * 1000
                if bal < self.min_dtao_balance:
                    bt.logging.info(f"UID={uid} 余额不足 {bal:.1f} < {self.min_dtao_balance}")
                    score = 0.0
                else:
                    score = 1.0 if correct else 0.0
                self.scores[uid] = score
                bt.logging.info(f"UID={uid} 验证 {'通过' if correct else '失败'}，得分={score}")

            # 准备下一轮
            synapse.audio_data   = encoded
            synapse.sample_id    = tid
            synapse.ground_truth = is_cat

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
        await super(Validator, self).concurrent_forward()
        self.set_weights()
        if self._state.get("round", 0) % 10 == 0:
            self.save_state()
        self._state["round"] = self._state.get("round", 0) + 1

    def set_weights(self):
        """
        Bittensor 要求：
          ∑ weights[active_uids] == #active_uids
        我们将 0 分者权重设 0，得分为 1 的矿工均匀分配权重。
        """
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
        w[active] = w[active] / w[active].sum() * total_active

        # 上链
        bt.logging.info(f"正在上链设置权重，活跃矿工={int(total_active)}，权重和={w.sum().item():.3f}")
        self.subtensor.set_weights(
            netuid             = self.config.netuid,
            wallet             = self.wallet,
            uids               = torch.arange(n, dtype=torch.long),
            weights            = w,
            wait_for_inclusion = True
        )

    def save_state(self):
        state = {
            "scores":      self.scores,
            "round":       self._state.get("round", 0),
        }
        path = os.path.join(self.config.neuron.full_path, "validator_state.json")
        with open(path, "w") as f:
            import json; json.dump(state, f)
        bt.logging.info("Validator state 已保存")

    def load_state(self):
        self._state = {}
        path = os.path.join(self.config.neuron.full_path, "validator_state.json")
        if os.path.isfile(path):
            with open(path, "r") as f:
                import json; self._state = json.load(f)
            bt.logging.info("Validator state 已加载")

if __name__ == "__main__":
    config    = get_config()
    config.neuron.full_path = os.path.expanduser(os.path.dirname(__file__))
    validator = Validator(config=config)
    # 这会启动 BaseValidatorNeuron 内部的 asyncio 循环并调用 concurrent_forward()
    validator.run()
