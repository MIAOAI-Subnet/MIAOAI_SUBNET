# MIT License
# Copyright © 2023 Yuma Rao

import typing
import bittensor as bt
import base64

class CatSoundProtocol(bt.Synapse):
    """Cat Sound Recognition Protocol"""

    # Input field - base64 encoded audio data
    audio_data: str

    # Output fields - recognition results
    is_cat_sound: typing.Optional[bool] = None
    probability: typing.Optional[float] = None
    confidence_level: typing.Optional[str] = None
    response_time: typing.Optional[float] = None

    def deserialize(self) -> dict:
        """Return serializable dictionary"""
        return {
            'is_cat_sound': self.is_cat_sound,
            'probability': self.probability,
            'confidence_level': self.confidence_level,
            'response_time': self.response_time
        }
