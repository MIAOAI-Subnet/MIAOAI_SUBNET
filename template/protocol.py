# MIT License
# Copyright © 2023 Yuma Rao

import typing
import bittensor as bt
import base64

class CatSoundProtocol(bt.Synapse):
    """猫叫声识别协议"""

    # 输入字段 - base64编码的音频数据
    audio_data: str

    # 输出字段 - 识别结果
    is_cat_sound: typing.Optional[bool] = None
    probability: typing.Optional[float] = None
    confidence_level: typing.Optional[str] = None
    response_time: typing.Optional[float] = None

    def deserialize(self) -> dict:
        """返回可序列化的字典"""
        return {
            'is_cat_sound': self.is_cat_sound,
            'probability': self.probability,
            'confidence_level': self.confidence_level,
            'response_time': self.response_time
        }
