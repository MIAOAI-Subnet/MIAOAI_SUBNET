# MIT License
# Copyright © 2025 MIAO

import time
import bittensor as bt
import asyncio
import base64
import numpy as np
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

        super(Validator, self).__init__(config=config)

        bt.logging.info("Loading validator status")
        self.load_state()
        self.scores = {}

    async def forward(
        self, synapse: CatSoundProtocol
    ) -> CatSoundProtocol:
        """
        处理验证请求，验证鉴别猫叫声的能力
        """
        
        try:
            bt.logging.info(f"验证请求从UUID={synapse.dendrite.uuid if hasattr(synapse, 'dendrite') else 'Unknown'}")
            
            # 这里应该放置验证逻辑
            # 在实际的验证器中，应该实现真实的逻辑来测试矿工的能力
            
            # 适配矿工的响应格式
            if synapse.is_cat_sound is not None:
                result = "miao" if synapse.is_cat_sound else "not miao"
                bt.logging.info(f"矿工返回: {result}, 概率: {synapse.probability}, 置信度: {synapse.confidence_level}, 响应时间: {synapse.response_time}")
                
                # 获取当前矿工的uid
                try:
                    if hasattr(synapse, 'dendrite') and hasattr(synapse.dendrite, 'hotkey') and synapse.dendrite.hotkey:
                        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
                        
                        # 计算得分
                        score = 0.0
                        # 根据概率和置信度评分
                        if synapse.probability > 0.8 and synapse.confidence_level == "高":
                            score = 1.0
                        elif synapse.probability > 0.5 and synapse.confidence_level in ["中", "高"]:
                            score = 0.8
                        elif synapse.probability > 0.2:
                            score = 0.5
                        else:
                            score = 0.2
                            
                        # 响应时间也纳入评分
                        if synapse.response_time < 1.0:  # 响应时间小于1秒
                            score *= 1.0
                        elif synapse.response_time < 2.0:  # 响应时间小于2秒
                            score *= 0.9
                        else:  # 响应时间大于2秒
                            score *= 0.8
                            
                        # 更新对应矿工的得分
                        self.scores[uid] = score
                        bt.logging.info(f"矿工 {uid} 得分: {score}")
                    else:
                        bt.logging.warning("无法获取矿工的hotkey信息")
                except ValueError as e:
                    bt.logging.warning(f"无法找到对应的矿工UID: {e}")
                
            else:
                bt.logging.warning("矿工未返回有效响应")
        except Exception as e:
            bt.logging.error(f"处理验证请求时发生错误: {e}")
        
        return synapse

    async def create_synapse(self) -> CatSoundProtocol:
        """创建一个测试用的synapse对象"""
        # 创建一个空的音频数据（实际使用时应该使用真实的音频数据）
        dummy_audio = b"dummy_audio_data"
        base64_audio = base64.b64encode(dummy_audio).decode('utf-8')
        
        # 创建synapse对象
        synapse = CatSoundProtocol(
            audio_data=base64_audio
        )
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
        except Exception as e:
            bt.logging.error(f"并发验证时发生错误: {e}")

    def score(self, uid: int) -> float:

        return self.scores.get(uid, 0.0)

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
