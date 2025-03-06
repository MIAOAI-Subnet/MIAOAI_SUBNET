import torch
import bittensor as bt
from bittensor.utils.balance import tao, rao

def get_rewards(self, responses: list, ground_truth: bool) -> torch.FloatTensor:


    rewards = torch.zeros(len(responses))
    

    MAX_RESPONSE_TIME = 5.0
    
    for i, response in enumerate(responses):
        try:
            if response is None or not isinstance(response, dict):
                bt.logging.warning(f"Invalid response #{i}")
                continue
                
            score = 0.0
            
        
            if response.get('is_dog_sound') == ground_truth:
                score += 0.6
    
            probability = response.get('probability')
            confidence_score = 0.0
            if probability is not None:
                if ground_truth:
          
                    confidence_score = min(float(probability), 1.0) * 0.2
                else:
                
                    confidence_score = (1 - min(float(probability), 1.0)) * 0.2
            score += confidence_score
            
   
            response_time = response.get('response_time')
            time_score = 0.0
            if response_time is not None:
                time_score = max(0, 1 - (float(response_time) / MAX_RESPONSE_TIME)) * 0.2
            score += time_score
            
            bt.logging.debug(
                f"miner #{i} Rating Details:\n"
                f"- accuracy: {1.0 if response.get('is_dog_sound') == ground_truth else 0.0}\n"
                f"- Confidence: {confidence_score/0.2}\n"
                f"- Response time: {time_score/0.2}\n"
                f"Total score: {score}"
            )
            
            rewards[i] = score
            
        except Exception as e:
            bt.logging.error(f"Error calculating reward #{i}: {str(e)}")
            continue
    
 
    if torch.sum(rewards) > 0:
        rewards = rewards / torch.sum(rewards)
    
    return rewards
