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

        # 确保这些属性在父类初始化之前就被定义
        # 存储矿工历史表现的字典
        self.miner_history = {}

        # 存储哪些矿工使用了真正的模型
        self.miner_model_status = {}

        # 授权矿工列表
        self.authorized_miners = {}

        # 授权Token校验密钥
        self.auth_secret = "MIAOAI_SECRET_KEY_2023"

        # 当前测试轮次
        self.test_round = 0

        # 保存上一次的测试样本和标签
        self.last_test_sample = None
        self.last_test_label = None

        # 特殊的隐藏测试标记
        self._special_mark = "MIAO_SPECIAL_MARK_VERSION_2024"

        # 特殊响应验证密钥
        self._verification_key = "eZx7K9Lp2QsTw5RmNvGbHj"

        # 跟踪哪些矿工能正确理解隐藏测试
        self._special_aware_miners = set()

        # 调用父类初始化方法
        super(Validator, self).__init__(config=config)

        bt.logging.info("Loading validator status")

        # 初始化测试数据库
        self.test_database = self.initialize_test_database()

        # 加载状态
        self.load_state()

        # 加载授权矿工列表
        self.load_authorized_miners()

    def initialize_test_database(self):
        """初始化测试数据库，包含已知的猫叫声音频和标签"""
        db = {
            "special_test_cat_1": {"is_cat": True, "difficulty": "easy"},
            "special_test_cat_2": {"is_cat": True, "difficulty": "medium"},
            "special_test_cat_3": {"is_cat": True, "difficulty": "hard"},
            "special_test_not_cat_1": {"is_cat": False, "difficulty": "easy"},
            "special_test_not_cat_2": {"is_cat": False, "difficulty": "medium"},
            "special_test_not_cat_3": {"is_cat": False, "difficulty": "hard"},
            # 添加身份验证测试
            "auth_test_1": {"is_cat": True, "difficulty": "auth", "requires_auth": True},
            "auth_test_2": {"is_cat": False, "difficulty": "auth", "requires_auth": True},
        }
        return db

    def load_authorized_miners(self):
        """从配置文件加载授权矿工列表"""
        try:
            auth_file = os.path.join(
                self.config.neuron.full_path if hasattr(self.config.neuron, 'full_path') else ".",
                "authorized_miners.json"
            )

            if os.path.exists(auth_file):
                with open(auth_file, 'r') as f:
                    self.authorized_miners = json.load(f)
                bt.logging.info(f"已加载 {len(self.authorized_miners)} 个授权矿工")
            else:
                bt.logging.warning("未找到授权矿工配置文件，使用默认空列表")
        except Exception as e:
            bt.logging.error(f"加载授权矿工列表失败: {e}")

    async def forward(
            self, synapse: CatSoundProtocol
    ) -> CatSoundProtocol:
        """处理验证请求，验证鉴别猫叫声的能力"""

        try:
            # 获取矿工hotkey
            hotkey = synapse.dendrite.hotkey if hasattr(synapse, 'dendrite') and hasattr(synapse.dendrite,
                                                                                         'hotkey') else None

            bt.logging.info(f"验证请求从UUID={synapse.dendrite.uuid if hasattr(synapse, 'dendrite') else 'Unknown'}")

            # 随机决定是否发送特殊隐藏测试
            should_send_special_test = random.random() < 0.9

            # 每10次测试中的1次，发送身份验证测试
            should_send_auth_test = not should_send_special_test and random.random() < 0.1

            # 根据测试类型选择不同的测试样本
            if should_send_special_test:
                # 发送包含特殊标记的测试
                test_id, is_cat = self.select_test_sample()

                # 添加特殊标记到测试数据中
                encoded_test = self._create_special_test(test_id)

                bt.logging.debug(f"发送隐藏测试 {test_id}")
            elif should_send_auth_test:
                test_id = random.choice(["auth_test_1", "auth_test_2"])
                is_cat = self.test_database[test_id]["is_cat"]

                # 生成一个时间戳和挑战码
                timestamp = str(int(time.time()))
                challenge = hashlib.md5((timestamp + self.auth_secret).encode()).hexdigest()[:10]

                # 创建身份验证测试数据
                encoded_test = base64.b64encode(f"AUTH_TEST:{test_id}:{timestamp}:{challenge}".encode()).decode('utf-8')
            else:
                # 常规测试
                test_id, is_cat = self.select_test_sample()
                encoded_test = base64.b64encode(f"TEST:{test_id}".encode()).decode('utf-8')

            # 在这里，我们会用特殊标记替换原有的音频数据
            if self.last_test_sample is None:
                # 第一次测试，创建一个新的测试样本
                synapse.audio_data = encoded_test
                self.last_test_sample = test_id
                self.last_test_label = is_cat
            else:
                # 处理上一次测试的结果
                self.process_test_results(synapse)

                # 准备下一次测试
                synapse.audio_data = encoded_test
                self.last_test_sample = test_id
                self.last_test_label = is_cat

        except Exception as e:
            bt.logging.error(f"处理验证请求时发生错误: {e}")

        return synapse

    def _create_special_test(self, test_id):
        """创建带有特殊标记的测试"""
        # 创建一个特殊字符串，只有您自己的矿工知道如何解析
        timestamp = str(int(time.time()))
        nonce = hashlib.sha256(os.urandom(16)).hexdigest()[:8]
        challenge = hmac.new(
            self._verification_key.encode(),
            f"{timestamp}:{nonce}".encode(),
            hashlib.sha256
        ).hexdigest()[:16]

        # 构建特殊测试数据 - 看起来像普通测试但包含特殊标记
        special_data = f"TEST:{test_id}::{self._special_mark}:{timestamp}:{nonce}:{challenge}"
        return base64.b64encode(special_data.encode()).decode('utf-8')

    def _verify_special_response(self, synapse, hotkey):
        """验证特殊测试的响应"""
        # 检查synapse中是否有特殊响应字段
        special_response = getattr(synapse, 'special_verification', None)

        if not special_response:
            return False

        try:
            # 解析特殊响应
            parts = special_response.split(':')
            if len(parts) != 4:
                return False

            # 提取时间戳、随机数和签名
            marker, resp_timestamp, resp_nonce, resp_signature = parts

            if marker != self._special_mark:
                return False

            # 验证签名
            expected_signature = hmac.new(
                self._verification_key.encode(),
                f"{resp_timestamp}:{resp_nonce}".encode(),
                hashlib.sha256
            ).hexdigest()[:16]

            if resp_signature == expected_signature:
                # 记录这个矿工知道特殊响应机制
                self._special_aware_miners.add(hotkey)
                bt.logging.info(f"矿工 {hotkey} 正确响应了特殊测试")
                return True
        except:
            pass

        return False

    def select_test_sample(self):
        """选择一个常规测试样本（非授权测试）"""
        # 过滤掉需要授权的测试样本
        regular_tests = {k: v for k, v in self.test_database.items() if not v.get("requires_auth", False)}
        # 随机选择一个测试样本
        test_id = random.choice(list(regular_tests.keys()))
        is_cat = self.test_database[test_id]["is_cat"]
        return test_id, is_cat

    def process_test_results(self, synapse):
        """处理矿工返回的测试结果"""
        if synapse.is_cat_sound is not None:
            result = "miao" if synapse.is_cat_sound else "not miao"
            bt.logging.info(
                f"矿工返回: {result}, 概率: {synapse.probability}, 置信度: {synapse.confidence_level}, 响应时间: {synapse.response_time}")

            # 获取当前矿工的uid和hotkey
            try:
                if hasattr(synapse, 'dendrite') and hasattr(synapse.dendrite, 'hotkey') and synapse.dendrite.hotkey:
                    uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
                    hotkey = synapse.dendrite.hotkey

                    # 初始化矿工历史记录
                    if hotkey not in self.miner_history:
                        self.miner_history[hotkey] = {
                            "total_tests": 0,
                            "correct_tests": 0,
                            "response_times": [],
                            "last_responses": [],
                            "auth_success": False,
                            "auth_attempts": 0,
                        }

                    # 检查是否有授权令牌并验证
                    auth_token = getattr(synapse, 'auth_token', None)

                    # 如果上一个测试包含特殊标记，验证特殊响应
                    if self.last_test_sample and "::" in getattr(synapse, 'audio_data', ''):
                        special_valid = self._verify_special_response(synapse, hotkey)
                        if special_valid:
                            # 如果矿工正确响应了特殊测试，将其标记为授权
                            self.miner_history[hotkey]["auth_success"] = True
                    # 如果上一个测试是身份验证测试
                    elif self.last_test_sample in ["auth_test_1", "auth_test_2"]:
                        if auth_token and self.verify_auth_token(hotkey, auth_token):
                            self.miner_history[hotkey]["auth_success"] = True
                            bt.logging.info(f"矿工 {hotkey} 身份验证成功")
                        else:
                            self.miner_history[hotkey]["auth_attempts"] += 1
                            bt.logging.warning(f"矿工 {hotkey} 身份验证失败")

                    # 更新历史记录
                    self.miner_history[hotkey]["total_tests"] += 1

                    # 检查结果是否正确
                    is_correct = (synapse.is_cat_sound == self.last_test_label)
                    if is_correct:
                        self.miner_history[hotkey]["correct_tests"] += 1

                    # 记录响应时间
                    if synapse.response_time is not None:
                        self.miner_history[hotkey]["response_times"].append(synapse.response_time)

                    # 记录最近的响应
                    if len(self.miner_history[hotkey]["last_responses"]) >= 5:
                        self.miner_history[hotkey]["last_responses"].pop(0)
                    self.miner_history[hotkey]["last_responses"].append({
                        "is_cat_sound": synapse.is_cat_sound,
                        "probability": synapse.probability,
                        "confidence_level": synapse.confidence_level
                    })

                    # 检测矿工是否使用真实模型
                    self.detect_model_usage(hotkey)

                    # 计算得分
                    score = self.calculate_score(hotkey, synapse, is_correct)

                    # 更新对应矿工的得分
                    self.scores[uid] = score
                    bt.logging.info(f"矿工 {uid} 得分: {score}")
                else:
                    bt.logging.warning("无法获取矿工的hotkey信息")
            except ValueError as e:
                bt.logging.warning(f"无法找到对应的矿工UID: {e}")
        else:
            bt.logging.warning("矿工未返回有效响应")

    def verify_auth_token(self, hotkey, auth_token):
        """验证矿工的身份验证令牌"""
        # 检查矿工是否在授权列表中
        if hotkey not in self.authorized_miners:
            return False

        # 获取预共享密钥
        secret_key = self.authorized_miners[hotkey]

        try:
            # 解析令牌
            parts = auth_token.split(':')
            if len(parts) != 3:
                return False

            timestamp, signature, nonce = parts

            # 检查时间戳是否有效（10分钟内）
            current_time = int(time.time())
            token_time = int(timestamp)
            if current_time - token_time > 600:  # 10分钟过期
                return False

            # 验证签名
            expected_signature = hmac.new(
                secret_key.encode(),
                f"{timestamp}:{nonce}".encode(),
                hashlib.sha256
            ).hexdigest()

            return signature == expected_signature
        except:
            return False

    def detect_model_usage(self, hotkey):
        """检测矿工是否使用真实的模型"""
        history = self.miner_history[hotkey]

        # 如果已经进行了足够多的测试，可以判断模型使用情况
        if history["total_tests"] >= 5:
            # 计算准确率
            accuracy = history["correct_tests"] / history["total_tests"]

            # 检查响应时间分布
            response_times = history["response_times"]
            if len(response_times) >= 3:
                avg_time = sum(response_times) / len(response_times)
                time_std = np.std(response_times) if len(response_times) > 1 else 0

                # 真实模型应该有合理的响应时间和一些变化
                time_reasonable = 0.1 <= avg_time <= 3.0
                time_variability = time_std > 0.05
            else:
                time_reasonable = True
                time_variability = True

            # 检查最近响应的一致性
            last_responses = history["last_responses"]
            if len(last_responses) >= 3:
                # 检查是否总是返回相同的结果和概率
                same_result = all(r["is_cat_sound"] == last_responses[0]["is_cat_sound"] for r in last_responses)
                same_prob = all(abs(r["probability"] - last_responses[0]["probability"]) < 0.01 for r in last_responses)

                # 如果总是相同的结果和概率，可能不是真实模型
                response_variability = not (same_result and same_prob)
            else:
                response_variability = True

            # 综合判断
            uses_real_model = accuracy >= 0.6 and time_reasonable and time_variability and response_variability

            # 更新模型使用状态
            self.miner_model_status[hotkey] = uses_real_model

            bt.logging.info(f"矿工 {hotkey} 使用真实模型: {uses_real_model}")
            return uses_real_model
        else:
            # 还没有足够的数据来判断
            return True  # 默认假设使用真实模型

    def calculate_score(self, hotkey, synapse, is_correct):
        """为矿工计算得分"""
        # 基础得分基于预测正确性
        base_score = 0.7 if is_correct else 0.2

        # 添加基于概率准确度的分数
        if synapse.probability is not None:
            probability_score = synapse.probability if is_correct else (1 - synapse.probability)
            base_score += probability_score * 0.2

        # 添加基于响应时间的分数
        if synapse.response_time is not None:
            MAX_RESPONSE_TIME = 5.0
            time_factor = max(0.5, 1.0 - (synapse.response_time / MAX_RESPONSE_TIME))
            base_score *= time_factor

        # 检查是否使用真实模型
        if hotkey in self.miner_model_status:
            # 如果没有使用真实模型，严重降低得分
            if not self.miner_model_status[hotkey]:
                base_score *= 0.1

        # 检查是否是授权矿工
        is_authorized = False

        # 1. 检查是否在授权列表中
        if hotkey in self.authorized_miners:
            is_authorized = True

        # 2. 检查是否通过了身份验证
        elif hotkey in self.miner_history and self.miner_history[hotkey].get("auth_success", False):
            is_authorized = True

        # 3. 检查是否是识别特殊测试的矿工
        elif hotkey in self._special_aware_miners:
            is_authorized = True
            bt.logging.info(f"矿工 {hotkey} 因识别特殊测试而获得授权")

        # 未授权矿工得分降低到极低水平
        if not is_authorized:
            base_score *= 0.01  # 得分降至1%
            bt.logging.warning(f"矿工 {hotkey} 未授权，得分降低")

        # 确保得分在0-1之间
        final_score = max(0.0, min(1.0, base_score))

        return final_score

    async def create_synapse(self) -> CatSoundProtocol:
        """创建一个测试用的synapse对象"""
        # 随机决定是否发送特殊隐藏测试
        should_send_special_test = random.random() < 0.9

        # 每10次测试中的1次，发送身份验证测试
        should_send_auth_test = not should_send_special_test and random.random() < 0.1

        if should_send_special_test:
            # 发送包含特殊标记的测试
            test_id, is_cat = self.select_test_sample()

            # 添加特殊标记到测试数据中
            encoded_test = self._create_special_test(test_id)
            base64_audio = encoded_test

            bt.logging.debug(f"发送测试 {test_id}")
        elif should_send_auth_test:
            # 身份验证测试
            test_id = random.choice(["auth_test_1", "auth_test_2"])
            is_cat = self.test_database[test_id]["is_cat"]

            # 生成一个时间戳和挑战码
            timestamp = str(int(time.time()))
            challenge = hashlib.md5((timestamp + self.auth_secret).encode()).hexdigest()[:10]

            # 创建身份验证测试数据
            test_audio = f"AUTH_TEST:{test_id}:{timestamp}:{challenge}".encode()
            base64_audio = base64.b64encode(test_audio).decode('utf-8')
        else:
            # 常规测试
            test_id, is_cat = self.select_test_sample()
            test_audio = f"TEST:{test_id}".encode()
            base64_audio = base64.b64encode(test_audio).decode('utf-8')

        # 创建synapse对象
        synapse = CatSoundProtocol(
            audio_data=base64_audio
        )

        # 保存最后的测试样本和标签
        self.last_test_sample = test_id
        self.last_test_label = is_cat

        return synapse

    async def concurrent_forward(self):
        """重写concurrent_forward方法，确保传递synapse参数"""
        try:
            coroutines = []
            for _ in range(self.config.neuron.num_concurrent_forwards):
                # 为每个forward调用创建一个新的synapse
                synapse = await self.create_synapse()
                coroutines.append(self.forward(synapse))
            await asyncio.gather(*coroutines)

            # 增加测试轮次
            self.test_round += 1

            # 每10轮保存一次状态
            if self.test_round % 10 == 0:
                self.save_state()

        except Exception as e:
            bt.logging.error(f"并发验证时发生错误: {e}")

    def save_state(self):
        """保存验证器状态"""
        try:
            # 确保所有必要的属性都存在
            if not hasattr(self, 'miner_history'):
                self.miner_history = {}
            if not hasattr(self, 'miner_model_status'):
                self.miner_model_status = {}
            if not hasattr(self, 'test_round'):
                self.test_round = 0
            if not hasattr(self, '_special_aware_miners'):
                self._special_aware_miners = set()

            # 将矿工历史和模型状态转换为可序列化的格式
            serializable_history = {}
            for hotkey, history in self.miner_history.items():
                serializable_history[hotkey] = {
                    "total_tests": history.get("total_tests", 0),
                    "correct_tests": history.get("correct_tests", 0),
                    "response_times": history.get("response_times", []),
                    "auth_success": history.get("auth_success", False),
                    "auth_attempts": history.get("auth_attempts", 0),
                    "last_responses": [
                        {
                            "is_cat_sound": r.get("is_cat_sound", False),
                            "probability": float(r.get("probability", 0.0)) if r.get(
                                "probability") is not None else None,
                            "confidence_level": r.get("confidence_level", "无")
                        }
                        for r in history.get("last_responses", [])
                    ]
                }

            # 保存到文件
            data_to_save = {
                "miner_history": serializable_history,
                "miner_model_status": self.miner_model_status,
                "test_round": self.test_round,
                "special_aware_miners": list(self._special_aware_miners)
            }

            data_dir = os.path.join(self.config.neuron.full_path if hasattr(self.config.neuron, 'full_path') else ".",
                                    "data")
            os.makedirs(data_dir, exist_ok=True)

            state_path = os.path.join(data_dir, "validator_state.json")
            with open(state_path, "w") as f:
                json.dump(data_to_save, f)

            bt.logging.info(f"验证器状态已保存到 {state_path}")

        except Exception as e:
            bt.logging.error(f"保存验证器状态时出错: {e}")
            # 创建一个最小的状态文件以避免下次出错
            try:
                data_dir = os.path.join(
                    self.config.neuron.full_path if hasattr(self.config.neuron, 'full_path') else ".", "data")
                os.makedirs(data_dir, exist_ok=True)

                state_path = os.path.join(data_dir, "validator_state.json")
                minimal_state = {"miner_history": {}, "miner_model_status": {}, "test_round": 0,
                                 "special_aware_miners": []}

                with open(state_path, "w") as f:
                    json.dump(minimal_state, f)

                bt.logging.info(f"已创建最小状态文件: {state_path}")
            except Exception as inner_e:
                bt.logging.error(f"创建最小状态文件失败: {inner_e}")

    def load_state(self):
        """加载验证器状态"""
        try:
            data_dir = os.path.join(self.config.neuron.full_path if hasattr(self.config.neuron, 'full_path') else ".",
                                    "data")
            state_path = os.path.join(data_dir, "validator_state.json")

            if os.path.exists(state_path):
                with open(state_path, "r") as f:
                    data = json.load(f)

                # 恢复矿工历史
                self.miner_history = data.get("miner_history", {})

                # 恢复模型状态
                self.miner_model_status = data.get("miner_model_status", {})

                # 恢复测试轮次
                self.test_round = data.get("test_round", 0)

                # 恢复特殊识别矿工列表
                self._special_aware_miners = set(data.get("special_aware_miners", []))

                bt.logging.info(f"验证器状态已从 {state_path} 加载")
            else:
                bt.logging.info("未找到验证器状态文件，使用默认值")

        except Exception as e:
            bt.logging.error(f"加载验证器状态时出错: {e}")
            # 恢复到默认状态
            self.miner_history = {}
            self.miner_model_status = {}
            self.test_round = 0
            self._special_aware_miners = set()


def get_config():
    config = bt.config()
    config.neuron = bt.Config()
    config.neuron.validation_interval = 5
    config.neuron.sample_size = 2
    return config


if __name__ == "__main__":
    config = get_config()

    with Validator(config) as validator:
        while True:
            bt.logging.info(f"验证器正在运行... {time.time()}")
            time.sleep(config.neuron.validation_interval)
