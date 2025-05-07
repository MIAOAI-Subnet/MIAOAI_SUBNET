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

        # Initialize tracking variables
        self.miner_history = {}
        self.miner_model_status = {}
        self.test_round = 0
        self._special_mark = "MIAO_SPECIAL_MARK_VERSION_2024"
        self._verification_key = "eZx7K9Lp2QsTw5RmNvGbHj"
        self._special_aware_miners = set()
        self.min_dtao_balance = 50.0
        self.miner_dtao_balance = {}
        self.last_balance_check = {}
        self.scores_dict = {}  # Dictionary format for scores

        super(Validator, self).__init__(config=config)
        bt.logging.info("Loading validator status")

        self.test_database = self.initialize_test_database()
        self.load_state()
        bt.logging.info("Starting validator for MIAOAI subnet with three-tier scoring system")

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

            if random.random() < 0.3:  # 30% chance of special test
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
            synapse.audio_data = "ERROR"

        return synapse

    def select_test_sample(self):
        test_id = random.choice(list(self.test_database.keys()))
        is_cat = self.test_database[test_id]["is_cat"]
        return test_id, is_cat

    def _create_special_test(self, test_id):
        # Add special marker for testing if miners run real model
        special_data = f"TEST:{test_id}:{self._special_mark}"
        return base64.b64encode(special_data.encode()).decode('utf-8')

    def _verify_special_response(self, response, expected_id):
        if not response or not isinstance(response, str):
            return False
        
        try:
            # Check if response contains awareness of special marker
            return self._special_mark in response
        except:
            return False

    def process_test_results(self, synapse):
        hotkey = getattr(synapse.dendrite, 'hotkey', None)
        if not hotkey:
            bt.logging.warning("No hotkey found in synapse")
            return
            
        uid = None
        try:
            uid = self.metagraph.hotkeys.index(hotkey)
        except ValueError:
            bt.logging.warning(f"Hotkey {hotkey} not found in metagraph")
            return

        bt.logging.info(f"Processing results from UID={uid}")
        
        # Process response based on test type
        if synapse.sample_id:
            expected_result = synapse.ground_truth
            
            # Check if this was a special test
            is_special_test = False
            if hasattr(synapse, 'response') and synapse.response:
                if hasattr(synapse.response, 'hidden_marker_response'):
                    hidden_response = synapse.response.hidden_marker_response
                    if hidden_response:
                        is_special_test = self._verify_special_response(hidden_response, synapse.sample_id)
                        if is_special_test:
                            bt.logging.info(f"Miner {uid} passed special test")
                            if uid is not None:
                                self._special_aware_miners.add(uid)
            
            # Record prediction accuracy
            if hasattr(synapse, 'response') and synapse.response:
                if hasattr(synapse.response, 'predictions') and synapse.response.predictions:
                    pred = synapse.response.predictions[0] if isinstance(synapse.response.predictions, list) and len(synapse.response.predictions) > 0 else None
                    if pred and isinstance(pred, dict) and 'is_cat' in pred:
                        result = pred['is_cat'] == expected_result
                        
                        # Track miner response history
                        if uid not in self.miner_history:
                            self.miner_history[uid] = []
                        
                        self.miner_history[uid].append({
                            'timestamp': time.time(),
                            'result': result,
                            'is_special_test': is_special_test
                        })
                        
                        # Check if the miner is using a real model
                        model_usage = self.detect_model_usage(uid)
                        self.miner_model_status[uid] = model_usage
                        
                        # Check dTAO balance periodically (every 300 seconds)
                        current_time = time.time()
                        if uid not in self.last_balance_check or (current_time - self.last_balance_check.get(uid, 0)) > 300:
                            balance = self.check_dtao_balance(hotkey)
                            self.miner_dtao_balance[uid] = balance
                            self.last_balance_check[uid] = current_time
                            bt.logging.info(f"Updated dTAO balance for UID={uid}: {balance}")
                            
                        # Calculate score
                        score = self.calculate_score(uid, result)
                        self.scores_dict[uid] = float(score)  # Store score in dictionary format
                        
                        bt.logging.info(f"Score for UID={uid}: {score}")
                        
            else:
                bt.logging.warning(f"No response from miner {uid}")

    def detect_model_usage(self, uid):
        # Check if miner is aware of special tests
        if uid in self._special_aware_miners:
            return True
            
        # Check recent history
        if uid in self.miner_history:
            history = self.miner_history[uid]
            
            # Only consider miners with sufficient history
            if len(history) >= 5:
                # Check if miner has consistent performance
                results = [entry['result'] for entry in history[-10:]]
                if sum(results) / len(results) > 0.7:
                    # Miners with consistently good performance likely run real models
                    return True
                
                # Check if there's variance in success rate
                if 0.3 < sum(results) / len(results) < 0.9:
                    # Some variance indicates real model behavior
                    return True
                    
                # Check for patterns that indicate artificial responses
                consecutive_same = 1
                max_consecutive = 1
                for i in range(1, len(results)):
                    if results[i] == results[i-1]:
                        consecutive_same += 1
                        max_consecutive = max(max_consecutive, consecutive_same)
                    else:
                        consecutive_same = 1
                
                # Long streaks of identical results suggest fake responses
                if max_consecutive > 7:
                    return False
                    
        # Default to assuming model is NOT running
        return False

    def check_dtao_balance(self, hotkey):
        try:
            # Get the actual dTAO balance from the blockchain
            subtensor = bt.subtensor()
            balance = subtensor.get_balance(hotkey)
            bt.logging.info(f"Checked real dTAO balance for {hotkey}: {balance}")
            return float(balance)
        except Exception as e:
            bt.logging.error(f"Error checking dTAO balance: {e}")
            return 0.0

    def calculate_score(self, uid, result):
        # Three-tier scoring system
        # 0.0 - Cheating or insufficient dTAO balance
        # 0.5 - Installed but not running a model
        # 1.0 - Running a model, adjusted by performance
        
        # Check if miner has no history (new miner)
        if uid not in self.miner_history or len(self.miner_history[uid]) < 3:
            bt.logging.info(f"UID={uid} has insufficient history, setting score to 0.0")
            return 0.0
            
        # Check dTAO balance requirement
        if uid in self.miner_dtao_balance and self.miner_dtao_balance[uid] < self.min_dtao_balance:
            bt.logging.info(f"UID={uid} has insufficient dTAO balance ({self.miner_dtao_balance[uid]} < {self.min_dtao_balance})")
            return 0.0
            
        # Check if miner is running a real model
        is_running_model = self.miner_model_status.get(uid, False)
        
        if not is_running_model:
            bt.logging.info(f"UID={uid} is not running a real model, setting score to 0.5")
            return 0.5
            
        # For miners running a model, adjust score based on performance
        recent_results = [entry['result'] for entry in self.miner_history[uid][-10:]]
        accuracy = sum(recent_results) / len(recent_results)
        
        # Performance adjustment: 0.8 to 1.0 based on accuracy
        performance_score = 0.8 + 0.2 * accuracy
        
        bt.logging.info(f"UID={uid} is running a real model with accuracy {accuracy}, setting score to {performance_score}")
        return float(performance_score)

    def save_state(self):
        try:
            state = {
                'miner_history': self.miner_history,
                'miner_model_status': self.miner_model_status,
                'special_aware_miners': list(self._special_aware_miners),
                'miner_dtao_balance': self.miner_dtao_balance,
                'last_balance_check': self.last_balance_check,
                'scores': self.scores_dict,  # Save scores as dictionary
                'test_round': self.test_round
            }
            
            with open('validator_state.json', 'w') as f:
                json.dump(state, f)
                
            bt.logging.info("Validator state saved successfully")
        except Exception as e:
            bt.logging.error(f"Error saving validator state: {e}")

    def load_state(self):
        try:
            if os.path.exists('validator_state.json'):
                with open('validator_state.json', 'r') as f:
                    state = json.load(f)
                    
                self.miner_history = state.get('miner_history', {})
                self.miner_model_status = state.get('miner_model_status', {})
                self._special_aware_miners = set(state.get('special_aware_miners', []))
                self.miner_dtao_balance = state.get('miner_dtao_balance', {})
                self.last_balance_check = state.get('last_balance_check', {})
                self.scores_dict = state.get('scores', {})  # Load scores as dictionary
                self.test_round = state.get('test_round', 0)
                
                bt.logging.info("Validator state loaded successfully")
        except Exception as e:
            bt.logging.error(f"Error loading validator state: {e}")

    def set_weights(self):
        try:
            # Convert scores dictionary to numpy array aligned with UIDs
            scores = torch.zeros(self.metagraph.n)
            
            for uid, score in self.scores_dict.items():
                if isinstance(uid, str):
                    uid = int(uid)
                if uid < self.metagraph.n:
                    scores[uid] = float(score)
            
            bt.logging.info(f"Setting weights for {len(self.scores_dict)} miners")
            
            # Normalize weights
            scores = torch.nn.functional.normalize(scores, p=1, dim=0)
            
            # Set weights on the blockchain
            result = self.subtensor.set_weights(
                netuid=self.config.netuid,
                wallet=self.wallet,
                uids=torch.arange(0, self.metagraph.n),
                weights=scores,
                wait_for_inclusion=False
            )
            
            if result:
                bt.logging.info("Successfully set weights")
            else:
                bt.logging.error("Failed to set weights")
                
        except Exception as e:
            bt.logging.error(f"Error setting weights: {str(e)}")

    async def run_step(self):
        try:
            # Resync the metagraph
            self.metagraph.sync()
            bt.logging.info(f"Metagraph updated with {self.metagraph.n} total miners")
            
            if self.test_round % 10 == 0:
                # Update weights every 10 rounds
                self.set_weights()
                
                # Save state
                self.save_state()
                
            self.test_round += 1
            
            # Query random miners
            await self.query_miners()
            
        except Exception as e:
            bt.logging.error(f"Error in run_step: {e}")

    async def query_miners(self):
        # Select sample_size random miners to query
        uids = random.sample(range(self.metagraph.n), min(self.config.neuron.sample_size, self.metagraph.n))
        
        bt.logging.info(f"Querying {len(uids)} random miners")
        
        for uid in uids:
            try:
                # Get the miner endpoint
                axon = self.metagraph.axons[uid]
                
                # Create the synapse with testing sample
                test_id, is_cat = self.select_test_sample()
                
                if random.random() < 0.3:  # 30% special test probability
                    encoded = self._create_special_test(test_id)
                else:
                    encoded = base64.b64encode(f"TEST:{test_id}".encode()).decode('utf-8')
                
                synapse = CatSoundProtocol(
                    audio_data=encoded,
                    sample_id=test_id,
                    ground_truth=is_cat
                )
                
                # Query the miner
                bt.logging.info(f"Querying UID={uid}")
                response = await self.dendrite.forward(axon, synapse, deserialize=True)
                
                # Process the results
                if response and hasattr(response, 'predictions'):
                    bt.logging.info(f"Got response from UID={uid}")
                    
                    # Process the response
                    pred = response.predictions[0] if isinstance(response.predictions, list) and len(response.predictions) > 0 else None
                    
                    if pred and isinstance(pred, dict) and 'is_cat' in pred:
                        result = pred['is_cat'] == is_cat
                        
                        # Update miner history
                        if uid not in self.miner_history:
                            self.miner_history[uid] = []
                            
                        # Check if this was a special test
                        is_special_test = False
                        if hasattr(response, 'hidden_marker_response') and response.hidden_marker_response:
                            is_special_test = self._verify_special_response(response.hidden_marker_response, test_id)
                            if is_special_test:
                                bt.logging.info(f"Miner {uid} passed special test")
                                self._special_aware_miners.add(uid)
                        
                        # Record the result
                        self.miner_history[uid].append({
                            'timestamp': time.time(),
                            'result': result,
                            'is_special_test': is_special_test
                        })
                        
                        # Update model status
                        model_usage = self.detect_model_usage(uid)
                        self.miner_model_status[uid] = model_usage
                        
                        # Check dTAO balance
                        hotkey = self.metagraph.hotkeys[uid]
                        current_time = time.time()
                        if uid not in self.last_balance_check or (current_time - self.last_balance_check.get(uid, 0)) > 300:
                            balance = self.check_dtao_balance(hotkey)
                            self.miner_dtao_balance[uid] = balance
                            self.last_balance_check[uid] = current_time
                            
                        # Calculate score
                        score = self.calculate_score(uid, result)
                        self.scores_dict[uid] = float(score)
                        
                        bt.logging.info(f"Score for UID={uid}: {score}")
                
            except Exception as e:
                bt.logging.error(f"Error querying miner {uid}: {e}")
