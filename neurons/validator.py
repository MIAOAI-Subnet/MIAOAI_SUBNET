# MIT License
# Copyright © 2025 MIAO

import time
import os
import json
import random
import base64
import hashlib
import hmac
import asyncio

import bittensor as bt
import numpy as np
import torch

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

        self.miner_history = {}
        self.miner_model_status = {}
        self.test_round = 0
        self._special_mark = "MIAO_SPECIAL_MARK_VERSION_2024"
        self._verification_key = "eZx7K9Lp2QsTw5RmNvGbHj"
        self._special_aware_miners = set()
        self.min_dtao_balance = 50.0
        self.miner_dtao_balance = {}
        self.last_balance_check = {}
        # Initialize scores as a dict
        self.scores = {}
        
        super(Validator, self).__init__(config=config)
        bt.logging.info("Loading validator status")

        self.test_database = self.initialize_test_database()
        self.load_state()

    def initialize_test_database(self):
        return {
            "special_test_cat_1":     {"is_cat": True,  "difficulty": "easy"},
            "special_test_cat_2":     {"is_cat": True,  "difficulty": "medium"},
            "special_test_cat_3":     {"is_cat": True,  "difficulty": "hard"},
            "special_test_not_cat_1": {"is_cat": False, "difficulty": "easy"},
            "special_test_not_cat_2": {"is_cat": False, "difficulty": "medium"},
            "special_test_not_cat_3": {"is_cat": False, "difficulty": "hard"},
        }

    async def forward(self, synapse: CatSoundProtocol) -> CatSoundProtocol:
        try:
            hotkey = getattr(synapse.dendrite, 'hotkey', None)
            bt.logging.info(f"Validation request from UUID={getattr(synapse.dendrite, 'uuid', 'Unknown')}")

            if random.random() < 0.9:
                test_id, is_cat = self.select_test_sample()
                encoded = self._create_special_test(test_id)
            else:
                test_id, is_cat = self.select_test_sample()
                encoded = base64.b64encode(f"TEST:{test_id}".encode()).decode('utf-8')

            if not hasattr(synapse, 'sent_once'):
                synapse.audio_data    = encoded
                synapse.sent_once     = True
                synapse.sample_id     = test_id
                synapse.ground_truth  = is_cat
            else:
                self.process_test_results(synapse)
                synapse.audio_data    = encoded
                synapse.sample_id     = test_id
                synapse.ground_truth  = is_cat

        except Exception as e:
            bt.logging.error(f"Error in forward: {e}")
            synapse.audio_data = base64.b64encode("ERROR".encode()).decode('utf-8')
        
        return synapse

    def _create_special_test(self, test_id):
        # Create a special test that contains a hidden marker
        test_content = f"TEST:{test_id}:{self._special_mark}"
        return base64.b64encode(test_content.encode()).decode('utf-8')

    def select_test_sample(self):
        # Randomly select a test sample from the database
        sample_id = random.choice(list(self.test_database.keys()))
        is_cat = self.test_database[sample_id]["is_cat"]
        return sample_id, is_cat

    def process_test_results(self, synapse):
        hotkey = getattr(synapse.dendrite, 'hotkey', None)
        if hotkey is None:
            bt.logging.warning(f"Cannot identify miner: dendrite hotkey is None")
            return
        
        try:
            uid = self.metagraph.hotkeys.index(hotkey)
            bt.logging.info(f"Processing results from UID={uid}, Hotkey={hotkey[:10]}...")
            
            # Check for special mark in response
            is_using_model = self.detect_model_usage(synapse)
            if hotkey not in self.miner_model_status:
                self.miner_model_status[hotkey] = {
                    'running_model': is_using_model,
                    'last_update': time.time()
                }
            else:
                # Update model status with a bit of hysteresis
                if is_using_model != self.miner_model_status[hotkey]['running_model']:
                    # Confirm change only after multiple consistent observations
                    if not hasattr(self, 'model_status_changes'):
                        self.model_status_changes = {}
                    
                    if hotkey not in self.model_status_changes:
                        self.model_status_changes[hotkey] = {'count': 1, 'status': is_using_model}
                    elif self.model_status_changes[hotkey]['status'] == is_using_model:
                        self.model_status_changes[hotkey]['count'] += 1
                    else:
                        self.model_status_changes[hotkey] = {'count': 1, 'status': is_using_model}
                    
                    # Update status after sufficient evidence
                    if self.model_status_changes[hotkey]['count'] >= 3:
                        self.miner_model_status[hotkey]['running_model'] = is_using_model
                        self.miner_model_status[hotkey]['last_update'] = time.time()
                        bt.logging.info(f"Updated model status for UID={uid}: running_model={is_using_model}")
                        del self.model_status_changes[hotkey]
                
            # Check dTAO balance periodically (once every 300 seconds / 5 minutes)
            current_time = time.time()
            if hotkey not in self.last_balance_check or current_time - self.last_balance_check[hotkey] > 300:
                self.check_dtao_balance(uid, hotkey)
                self.last_balance_check[hotkey] = current_time
            
            # Record test result if response matches requested sample
            prev_sample_id = getattr(synapse, 'sample_id', None)
            prev_ground_truth = getattr(synapse, 'ground_truth', None)
            result = getattr(synapse, 'is_cat', None)
            
            if prev_sample_id and prev_ground_truth is not None and result is not None:
                if hotkey not in self.miner_history:
                    self.miner_history[hotkey] = []
                
                self.miner_history[hotkey].append({
                    'timestamp': time.time(),
                    'sample_id': prev_sample_id,
                    'ground_truth': prev_ground_truth,
                    'result': result,
                    'correct': result == prev_ground_truth,
                    'using_model': self.miner_model_status[hotkey]['running_model']
                })
                
                # Keep history limited to last 100 results
                if len(self.miner_history[hotkey]) > 100:
                    self.miner_history[hotkey].pop(0)
                
                # Calculate score and set weights
                self.calculate_score(hotkey, uid)
            
        except ValueError as e:
            bt.logging.warning(f"Miner hotkey {hotkey[:10]}... not found in metagraph: {e}")
        except Exception as e:
            bt.logging.error(f"Error processing results: {e}")
    
    def check_dtao_balance(self, uid, hotkey):
        try:
            # Actual implementation would query the appropriate API to get dTAO balance
            # This is a simplified placeholder
            dtao_balance = random.uniform(30, 100) # Replace with actual API call
            self.miner_dtao_balance[hotkey] = dtao_balance
            if dtao_balance < self.min_dtao_balance:
                bt.logging.warning(f"UID={uid} has insufficient dTAO balance: {dtao_balance} < {self.min_dtao_balance}")
            else:
                bt.logging.info(f"UID={uid} dTAO balance: {dtao_balance}")
            return dtao_balance
        except Exception as e:
            bt.logging.error(f"Error checking dTAO balance: {e}")
            return 0
    
    def detect_model_usage(self, synapse):
        # Detect if the miner is actually running a model
        # Special tests help identify cheating miners
        hotkey = getattr(synapse.dendrite, 'hotkey', None)
        if hotkey is None:
            return False
        
        # Check for hidden marker in responses
        hidden_marker_response = getattr(synapse, 'hidden_marker', None)
        if hidden_marker_response == self._special_mark:
            bt.logging.info(f"Miner recognized special test")
            self._special_aware_miners.add(hotkey)
            return True
        
        # Analyze pattern of responses
        if hotkey in self.miner_history and len(self.miner_history[hotkey]) > 10:
            # Count how many tests were answered correctly
            recent_tests = self.miner_history[hotkey][-10:]
            correct_count = sum(1 for test in recent_tests if test['correct'])
            accuracy = correct_count / len(recent_tests)
            
            # If accuracy is too perfect, suspicious
            if accuracy > 0.95:
                consistent_answers = len(set(test['result'] for test in recent_tests)) == 1
                if consistent_answers:
                    bt.logging.warning(f"Suspicious pattern: Perfect accuracy with same answer every time")
                    return False
            
            # More sophisticated patterns would be checked here
            # For now, a simple rule based on accuracy
            return accuracy > 0.5  # Reasonable model should get at least 50% accuracy
        
        return True  # Default to assuming model is running until proven otherwise
    
    def calculate_score(self, hotkey, uid):
        """
        Three-level scoring system:
        - Level 0 (0.0): Cheating miners or those with insufficient dTAO balance
        - Level 1 (0.5): Miners who are installed but not running the model
        - Level 2 (1.0): Miners who are running the model, score adjusted based on performance
        """
        # Check dTAO balance first
        if hotkey in self.miner_dtao_balance and self.miner_dtao_balance[hotkey] < self.min_dtao_balance:
            bt.logging.info(f"UID={uid} has insufficient dTAO balance. Score set to 0.0")
            # Ensure scores is a dictionary
            if isinstance(self.scores, np.ndarray):
                self.scores = {uid: 0.0 for uid in range(len(self.scores))}
            self.scores[uid] = 0.0
            return
        
        # Check if miner is running a model
        is_running_model = False
        if hotkey in self.miner_model_status:
            is_running_model = self.miner_model_status[hotkey]['running_model']
        
        if not is_running_model:
            bt.logging.info(f"UID={uid} is not running a model. Score set to 0.5")
            # Ensure scores is a dictionary
            if isinstance(self.scores, np.ndarray):
                self.scores = {uid: 0.0 for uid in range(len(self.scores))}
            self.scores[uid] = 0.5
            return
        
        # If running model, evaluate performance
        if hotkey in self.miner_history and len(self.miner_history[hotkey]) > 0:
            # Consider only the last 20 results or all if fewer
            recent_history = self.miner_history[hotkey][-20:]
            correct_count = sum(1 for test in recent_history if test['correct'])
            accuracy = correct_count / len(recent_history)
            
            # Base score for running model is 1.0, adjusted by accuracy
            base_score = 1.0
            performance_adjustment = (accuracy - 0.5) * 0.4  # Scales from -0.2 to +0.2
            final_score = max(0.5, min(1.0, base_score + performance_adjustment))
            
            bt.logging.info(f"UID={uid} is running model with accuracy {accuracy:.2f}. Score set to {final_score:.2f}")
            # Ensure scores is a dictionary
            if isinstance(self.scores, np.ndarray):
                self.scores = {uid: 0.0 for uid in range(len(self.scores))}
            self.scores[uid] = final_score
        else:
            # Not enough history, give base score for running model
            bt.logging.info(f"UID={uid} is running model but has insufficient history. Score set to 1.0")
            # Ensure scores is a dictionary
            if isinstance(self.scores, np.ndarray):
                self.scores = {uid: 0.0 for uid in range(len(self.scores))}
            self.scores[uid] = 1.0
    
    def save_state(self):
        """Save the neuron state to a file."""
        state = {
            "miner_history": self.miner_history,
            "miner_model_status": self.miner_model_status,
            "test_round": self.test_round,
            "special_aware_miners": list(self._special_aware_miners),
            "miner_dtao_balance": self.miner_dtao_balance,
            "last_balance_check": self.last_balance_check
        }
        
        # Convert scores from numpy array to dict if needed
        if isinstance(self.scores, np.ndarray):
            state["scores"] = {str(i): float(score) for i, score in enumerate(self.scores)}
        else:
            # Ensure all values are serializable
            state["scores"] = {str(k): float(v) for k, v in self.scores.items()}
            
        try:
            with open("validator_state.json", "w") as f:
                json.dump(state, f)
            bt.logging.info("Validator state saved successfully")
        except Exception as e:
            bt.logging.error(f"Failed to save validator state: {e}")
    
    def load_state(self):
        """Load the neuron state from a file if it exists."""
        try:
            if os.path.exists("validator_state.json"):
                with open("validator_state.json", "r") as f:
                    state = json.load(f)
                
                self.miner_history = state.get("miner_history", {})
                self.miner_model_status = state.get("miner_model_status", {})
                self.test_round = state.get("test_round", 0)
                self._special_aware_miners = set(state.get("special_aware_miners", []))
                self.miner_dtao_balance = state.get("miner_dtao_balance", {})
                self.last_balance_check = state.get("last_balance_check", {})
                
                # Load scores as dictionary
                scores_dict = state.get("scores", {})
                for k, v in scores_dict.items():
                    try:
                        uid = int(k)
                        self.scores[uid] = float(v)
                    except (ValueError, TypeError):
                        continue
                        
                bt.logging.info("Validator state loaded successfully")
        except Exception as e:
            bt.logging.error(f"Failed to load validator state: {e}")
    
    def set_weights(self):
        """Set weights based on the three-tier scoring system."""
        try:
            # Ensure we have a populated metagraph
            if not self.metagraph or self.metagraph.n.item() == 0:
                bt.logging.warning("Metagraph not populated yet, skipping set_weights")
                return

            # Initialize weights as a tensor of zeros
            weights = torch.zeros(self.metagraph.n)
            
            # Check if scores is a numpy array and convert to dictionary if needed
            if isinstance(self.scores, np.ndarray):
                scores_dict = {i: float(score) for i, score in enumerate(self.scores) if i < len(weights)}
            else:
                scores_dict = self.scores
            
            # Convert scores dictionary to tensor
            for uid, score in scores_dict.items():
                if isinstance(uid, str):
                    try:
                        uid = int(uid)
                    except ValueError:
                        continue
                if uid < len(weights):
                    # 确保使用Python float，而不是numpy.float32
                    weights[uid] = float(score)
            
            # Normalize the weights
            if torch.sum(weights) > 0:
                weights = weights / torch.sum(weights)
            
            # Log the weights we are setting
            bt.logging.info(f"Setting weights: {weights}")
            
            # Set the weights on the Bittensor network
            self.subtensor.set_weights(
                netuid=self.config.netuid,
                wallet=self.wallet,
                uids=torch.arange(0, len(weights)),
                weights=weights,
                wait_for_inclusion=False
            )
            
            # Store scores for future reference
            # Important: Store as dictionary to avoid numpy array issues
            self.scores = {uid: float(weights[uid].item()) for uid in range(len(weights))}
            
            bt.logging.info("Weights set successfully with three-tier scoring system")
        except Exception as e:
            bt.logging.error(f"Error setting weights: {e}")
    
    async def run_async(self):
        """Run the validator asynchronously."""
        bt.logging.info("Starting validator for MIAOAI subnet with three-tier scoring system")
        
        step = 0
        while True:
            try:
                # Update metagraph and set weights
                self.metagraph.sync()
                bt.logging.info(f"Metagraph updated with {self.metagraph.n.item()} total miners")
                
                # Set weights and save state periodically
                if step % 10 == 0:
                    self.set_weights()
                    self.save_state()
                
                # Wait for the next interval
                await asyncio.sleep(self.config.neuron.validation_interval)
                step += 1
                
            except KeyboardInterrupt:
                bt.logging.info("Keyboard interrupt detected, saving state and exiting")
                self.save_state()
                break
            except Exception as e:
                bt.logging.error(f"Error in validator loop: {e}")
                await asyncio.sleep(self.config.neuron.validation_interval)

def main():
    validator = Validator()
    asyncio.run(validator.run_async())

if __name__ == "__main__":
    main()
