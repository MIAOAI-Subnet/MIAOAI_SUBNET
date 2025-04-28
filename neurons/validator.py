# MIT License
# Copyright © 2025 MIAO

import time
import bittensor as bt
from MIAOAI_SUBNET.template.base.validator import BaseValidatorNeuron
from MIAOAI_SUBNET.template.validator import forward

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
        
        bt.logging.info(f"验证请求从UID={synapse.dendrite.uid}")
        
        # 这里应该放置验证逻辑
        # 在实际的验证器中，应该实现真实的逻辑来测试矿工的能力
        
        if synapse.is_cat_sound is not None:
            bt.logging.info(f"矿工发现了猫叫声: {synapse.is_cat_sound}, 概率: {synapse.probability}")
        else:
            bt.logging.warning("矿工未返回有效响应")
        
        return synapse

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
