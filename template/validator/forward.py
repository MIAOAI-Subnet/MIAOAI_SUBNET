# MIT License
# Copyright © 2023 Yuma Rao

import time
import bittensor as bt
import base64
import os
import random

from template.protocol import CatSoundProtocol
from template.validator.reward import get_rewards
from template.utils.uids import get_random_uids

class TestData:

    def __init__(self):

        self.test_data_dir = "test_data"

        os.makedirs(self.test_data_dir, exist_ok=True)
        

        self.test_dataset = {
            "cat1.wav": True,
            "cat2.wav": True,
            "not_cat1.wav": False,
            "not_cat2.wav": False,
        }
    
    def get_random_test_case(self):

        filename = random.choice(list(self.test_dataset.keys()))
        filepath = os.path.join(self.test_data_dir, filename)
        

        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                audio_data = f.read()
            return {
                'audio_data': base64.b64encode(audio_data).decode('utf-8'),
                'is_cat_sound': self.test_dataset[filename]
            }
        else:
            bt.logging.warning(f"The test file does not exist: {filepath}")
            return None

async def forward(self):
    test_data = TestData()
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)
    test_case = test_data.get_random_test_case()
    if test_case is None:
        bt.logging.error("Unable to obtain test cases")
        return
    
    synapse = CatSoundProtocol(
        audio_data=test_case['audio_data']
    )
    

    responses = await self.dendrite(
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        synapse=synapse,
        deserialize=True,
    )
    
    bt.logging.info(f"Receive response: {responses}")
    
    rewards = get_rewards(
        self,
        responses=responses,
        ground_truth=test_case['is_cat_sound']
    )
    
    bt.logging.info(f"Rating results: {rewards}")

    self.update_scores(rewards, miner_uids)
    time.sleep(self.config.neuron.validation_interval)
