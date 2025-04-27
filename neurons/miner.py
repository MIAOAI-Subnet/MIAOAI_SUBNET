# MIT License
# Copyright © 2025 MIAO
import os
import time
import typing
import bittensor as bt
import requests
import base64
import template
from template.base.miner import BaseMinerNeuron

class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        
        # 更新为连接猫叫声识别系统的URL
        self.api_url = os.getenv('AI_API_URL', 'http://localhost:8080') + "/predict"
        bt.logging.info(f"正在使用API端点: {self.api_url}")

    async def forward(
        self, synapse: template.protocol.CatSoundProtocol
    ) -> template.protocol.CatSoundProtocol:

        try:
            
            start_time = time.time()
            
            # 解码音频数据
            audio_binary = base64.b64decode(synapse.audio_data)
            
            # 发送到猫叫声识别API
            files = {
                'file': ('recording.wav', audio_binary, 'audio/wav')
            }
            
            # 获取响应
            response = requests.post(self.api_url, files=files)
            response_json = response.json()
            
            # 计算响应时间
            response_time = time.time() - start_time
            
            # 根据结果判断是否为猫叫声
            # 适配新的API响应格式
            is_cat_sound = response_json['result'] == "miao"
            
            # 填充同步数据
            synapse.is_cat_sound = is_cat_sound
            synapse.probability = response_json['probability']
            synapse.confidence_level = response_json.get('confidence_level', '无')
            synapse.response_time = response_time
            
            bt.logging.debug(f"结果: {response_json['result']}, 概率: {response_json['probability']}, 置信度: {response_json.get('confidence_level', '无')}")
            
        except Exception as e:
            bt.logging.error(f"错误: {str(e)}")
            # 出错时返回默认值
            synapse.is_cat_sound = False
            synapse.probability = 0.0
            synapse.confidence_level = '无'
            synapse.response_time = 0.0
            
        return synapse

    async def blacklist(
        self, synapse: template.protocol.CatSoundProtocol
    ) -> typing.Tuple[bool, str]:

        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            return True, "Missing dendrite or hotkey"

        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        
        # 黑名单验证逻辑
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            return True, "Unregistered hotkeys"


        if self.config.blacklist.force_validator_permit:
            if not self.metagraph.validator_permit[uid]:
                return True, "Non-validator hotkeys"

        return False, "Certified Hotkeys"

    async def priority(self, synapse: template.protocol.CatSoundProtocol) -> float:
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            return 0.0

        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority = float(self.metagraph.S[caller_uid])
        
        bt.logging.trace(f"Request Priority {synapse.dendrite.hotkey}: {priority}")
        return priority


if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"矿工正在运行... {time.time()}")
            time.sleep(5)
