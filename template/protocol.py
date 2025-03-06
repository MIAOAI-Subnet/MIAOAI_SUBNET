# MIT License
# Copyright © 2023 Yuma Rao

import typing
import bittensor as bt
import base64

class DogSoundProtocol(bt.Synapse):

    audio_data: str

    is_dog_sound: typing.Optional[bool] = None
    probability: typing.Optional[float] = None
    response_time: typing.Optional[float] = None

    def deserialize(self) -> dict:
        return {
            'is_dog_sound': self.is_dog_sound,
            'probability': self.probability,
            'response_time': self.response_time
        }
