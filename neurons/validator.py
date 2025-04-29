# MIT License
# Copyright © 2025 MIAO

import time
import bittensor as bt
import template
from template.base.validator import BaseValidatorNeuron
from template.validator import forward
from template.protocol import CatSoundProtocol
import base64
import asyncio

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

    async def forward(
        self, synapse: template.protocol.CatSoundProtocol
    ) -> template.protocol.CatSoundProtocol:
        """
        处理验证请求，验证鉴别猫叫声的能力
        """
        
        bt.logging.info(f"验证请求从UUID={synapse.dendrite.uuid if hasattr(synapse, 'dendrite') else 'Unknown'}")
        
        # 这里应该放置验证逻辑
        # 在实际的验证器中，应该实现真实的逻辑来测试矿工的能力
        
        if synapse.is_cat_sound is not None:
            bt.logging.info(f"矿工发现了猫叫声: {synapse.is_cat_sound}, 概率: {synapse.probability}")
        else:
            bt.logging.warning("矿工未返回有效响应")
        
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
        coroutines = []
        for _ in range(self.config.neuron.num_concurrent_forwards):
            # 为每个forward调用创建一个新的synapse
            synapse = await self.create_synapse()
            coroutines.append(self.forward(synapse))
        await asyncio.gather(*coroutines)

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
            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(config.neuron.validation_interval)
