# MIT License
# Copyright © 2025 MIAO

import time
import bittensor as bt
from template.base.validator import BaseValidatorNeuron
from template.validator import forward

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

    async def forward(self):
        return await forward(self)

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
