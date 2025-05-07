# MIT License
# Copyright © 2025 MIAO

import time
import random
import base64
import os
import asyncio
import bittensor as bt
import numpy as np
import torch
from template.base.validator import BaseValidatorNeuron
from template.protocol import CatSoundProtocol

def get_config():
    import argparse
    parser = argparse.ArgumentParser()
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    parser.add_argument("--netuid", type=int, default=86, help="Subnet ID")
    parser.add_argument("--neuron.validation_interval", type=int, default=60, help="Validation interval (seconds)")
    parser.add_argument("--neuron.sample_size", type=int, default=10, help="Number of miners to sample per round")
    return bt.config(parser)

class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        # If no config is provided, use get_config()
        if config is None:
            config = get_config()
            
        # Initialize tracking variables before super().__init__
        self.miner_history = {}
        self.miner_model_status = {}
        self.min_dtao_balance = 50.0
        self._special_mark = "MIAO_SPECIAL_MARK_2025"
        self._special_aware_miners = set()
        self.last_weight_set_time = 0
        self._state = {"round": 0}
        
        # Important: Initialize scores as numpy array for compatibility
        self.scores = torch.zeros(256, dtype=torch.float32)
        
        # Initialize base class
        super(Validator, self).__init__(config=config)
        
        # Test samples database
        self.test_database = [
            ("cat_easy",     True),
            ("cat_medium",   True),
            ("cat_hard",     True),
            ("not_cat_easy", False),
            ("not_cat_med",  False),
            ("not_cat_hard", False),
        ]
        
        self.load_state()
        bt.logging.info("Validator initialization complete")

    def select_test_sample(self):
        # Randomly select a test sample (ID and ground truth)
        tid, is_cat = random.choice(self.test_database)
        return tid, is_cat

    def _encode_test(self, test_id: str, special: bool):
        if special:
            # Add special marker to detect if miners understand this marker
            payload = f"SPECIAL:TEST:{test_id}:{self._special_mark}"
        else:
            payload = f"TEST:{test_id}"
        return base64.b64encode(payload.encode()).decode("utf-8")

    async def create_synapse(self) -> CatSoundProtocol:
        """Create a new synapse object for concurrent_forward"""
        # Create synapse with required audio_data field
        synapse = CatSoundProtocol(audio_data="")
        
        # 20% chance of special test to detect cheating
        special = (random.random() < 0.2)
        tid, is_cat = self.select_test_sample()
        encoded = self._encode_test(tid, special)
        
        # Set the audio_data field (REQUIRED by protocol)
        synapse.audio_data = encoded
        
        # Store ground truth in metadata
        synapse._ground_truth = is_cat
        synapse._test_id = tid
        synapse._is_special = special
        
        return synapse

    def detect_model_usage(self, uid):
        """Determine if a miner is actually running a model"""
        # If miner can recognize special markers, they're running a real model
        if uid in self._special_aware_miners:
            return True
            
        # Check response history
        if uid in self.miner_history and len(self.miner_history[uid]) >= 5:
            history = self.miner_history[uid]
            
            # Use most recent 10 records
            recent = history[-10:] if len(history) > 10 else history
            
            # Calculate accuracy
            correct_count = sum(1 for h in recent if h['correct'])
            accuracy = correct_count / len(recent)
            
            # If accuracy is reasonable (not too low or perfect), likely a real model
            if 0.3 < accuracy < 0.9:
                return True
                
            # If accuracy is very high, might be a real model
            if accuracy >= 0.7:
                return True
                
            # Check response patterns - too regular responses might be fake
            consecutive = 1
            max_consecutive = 1
            for i in range(1, len(recent)):
                if recent[i]['correct'] == recent[i-1]['correct']:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    consecutive = 1
                    
            # If too many consecutive identical results, might be fake
            if max_consecutive > 7:
                return False
                
        # Default assumption: not a real model
        return False

    def calculate_score(self, uid, is_correct, uses_model, balance):
        """Calculate miner score: 0 or 1"""
        # Insufficient balance
        if balance < self.min_dtao_balance:
            bt.logging.info(f"UID={uid} insufficient balance {balance:.1f} < {self.min_dtao_balance}, score=0.0")
            return 0.0
            
        # Not running a model
        if not uses_model:
            bt.logging.info(f"UID={uid} not running a real model, score=0.0")
            return 0.0
            
        # Score based on correctness
        score = 1.0 if is_correct else 0.0
        bt.logging.info(f"UID={uid} answered {'correctly' if is_correct else 'incorrectly'}, score={score}")
        return score

    async def forward(self, synapse: CatSoundProtocol) -> CatSoundProtocol:
        """Standard forward method called by BaseValidatorNeuron"""
        try:
            # Check if we have a hotkey (processing previous response)
            hotkey = getattr(synapse.dendrite, 'hotkey', None)
            if hotkey:
                try:
                    uid = self.metagraph.hotkeys.index(hotkey)
                    
                    # Get the is_cat_sound field from response
                    is_cat_pred = getattr(synapse, 'is_cat_sound', None)
                    ground_truth = getattr(synapse, '_ground_truth', None)
                    
                    if is_cat_pred is not None and ground_truth is not None:
                        # Determine correctness
                        is_correct = (is_cat_pred == ground_truth)
                        
                        # Check special marker
                        is_special = getattr(synapse, '_is_special', False)
                        identified_special = False
                        
                        # Look for special marker in response
                        if is_special:
                            # app_cat_fixed.py might return special marker in different fields
                            special_response = ""
                            if hasattr(synapse, 'special_marker'):
                                special_response = synapse.special_marker
                            elif hasattr(synapse, 'confidence_level'):
                                special_response = str(synapse.confidence_level)
                                
                            if special_response and self._special_mark in special_response:
                                identified_special = True
                                self._special_aware_miners.add(uid)
                                bt.logging.info(f"UID={uid} identified special marker")
                        
                        # Update history
                        if uid not in self.miner_history:
                            self.miner_history[uid] = []
                            
                        self.miner_history[uid].append({
                            'timestamp': time.time(),
                            'correct': is_correct,
                            'special': is_special,
                            'identified_special': identified_special
                        })
                        
                        # Limit history length
                        if len(self.miner_history[uid]) > 100:
                            self.miner_history[uid] = self.miner_history[uid][-100:]
                        
                        # Detect model usage
                        uses_model = self.detect_model_usage(uid)
                        self.miner_model_status[uid] = uses_model
                        
                        # Get balance
                        balance = float(self.metagraph.S[uid]) * 1000
                        
                        # Calculate score and update tensor directly
                        score = self.calculate_score(uid, is_correct, uses_model, balance)
                        if uid < len(self.scores):
                            self.scores[uid] = float(score)
                        
                        bt.logging.info(f"UID={uid} result: correct={is_correct}, model={uses_model}, score={score}")
                    else:
                        bt.logging.warning(f"UID={uid} missing prediction or ground truth")
                        if uid < len(self.scores):
                            self.scores[uid] = 0.0
                
                except ValueError:
                    bt.logging.warning(f"Hotkey={hotkey} not found in metagraph")
                except Exception as e:
                    bt.logging.error(f"Error processing response: {e}")
            
            # Prepare new request
            special = (random.random() < 0.2)
            tid, is_cat = self.select_test_sample()
            encoded = self._encode_test(tid, special)
            
            # Set the required field
            synapse.audio_data = encoded
            
            # Store metadata for next round
            synapse._test_id = tid
            synapse._ground_truth = is_cat
            synapse._is_special = special
            
            # Reset previous response fields
            synapse.is_cat_sound = None
            synapse.probability = None
            synapse.confidence_level = None
            synapse.response_time = None
            
            return synapse
            
        except Exception as e:
            bt.logging.error(f"Forward error: {e}")
            # Ensure audio_data is always set
            synapse.audio_data = base64.b64encode(b"ERROR").decode("utf-8")
            return synapse

    async def run_step(self):
        """Standard run step for BaseValidatorNeuron"""
        try:
            # Sync metagraph
            self.metagraph.sync(subtensor=self.subtensor)
            bt.logging.info(f"Metagraph synced, miners={self.metagraph.n}")
            
            # Check if weights should be set (every 30 minutes)
            current_time = time.time()
            should_set_weights = (current_time - self.last_weight_set_time) > 1800
            
            if should_set_weights or self._state.get("round", 0) % 10 == 0:
                # Use standard weights API of BaseValidatorNeuron
                self.set_weights()
                self.save_state()
                self.last_weight_set_time = time.time()
                
            # Query miners
            await self.query_miners()
            
            # Update round counter
            self._state["round"] = self._state.get("round", 0) + 1
            
            # Add longer sleep to avoid rate limits
            await asyncio.sleep(self.config.neuron.validation_interval)
            
        except Exception as e:
            bt.logging.error(f"Run step error: {e}")
            # Sleep longer on error to avoid rapid restart loops
            await asyncio.sleep(60)

    async def query_miners(self):
        """Query a batch of random miners"""
        try:
            # Determine query count
            n = min(self.config.neuron.sample_size, self.metagraph.n)
            if n <= 0:
                bt.logging.warning("No miners available to query")
                return
                
            # Find active miners
            active = self.metagraph.active
            if isinstance(active, np.ndarray):
                active = torch.tensor(active, dtype=torch.bool)
            else:
                active = active.to(torch.bool)
                
            active_indices = torch.nonzero(active).flatten().tolist()
            if not active_indices:
                bt.logging.warning("No active miners")
                return
                
            # Randomly select miners
            selected_uids = random.sample(active_indices, min(n, len(active_indices)))
            bt.logging.info(f"Selected {len(selected_uids)} miners for this round")
            
            # Create query tasks
            tasks = []
            for uid in selected_uids:
                synapse = await self.create_synapse()
                axon = self.metagraph.axons[uid]
                
                # Add task with timeout
                task = asyncio.create_task(
                    self._query_single_miner(uid, axon, synapse)
                )
                tasks.append(task)
            
            # Wait for all tasks to complete
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                
        except Exception as e:
            bt.logging.error(f"Query miners error: {e}")

    async def _query_single_miner(self, uid, axon, synapse):
        """Query a single miner with timeout handling"""
        try:
            bt.logging.info(f"Querying UID={uid}")
            
            # Set timeout
            response = await asyncio.wait_for(
                self.dendrite.forward(axon, synapse, deserialize=True),
                timeout=10.0
            )
            
            # Process response using CatSoundProtocol fields
            is_cat_pred = getattr(response, 'is_cat_sound', None)
            ground_truth = getattr(synapse, '_ground_truth', None)
            
            if is_cat_pred is not None and ground_truth is not None:
                # Determine correctness
                is_correct = (is_cat_pred == ground_truth)
                
                # Check special marker
                is_special = getattr(synapse, '_is_special', False)
                identified_special = False
                
                # Check for special marker in different fields
                if is_special:
                    special_response = ""
                    if hasattr(response, 'special_marker'):
                        special_response = response.special_marker
                    elif hasattr(response, 'confidence_level'):
                        special_response = str(response.confidence_level)
                        
                    if special_response and self._special_mark in special_response:
                        identified_special = True
                        self._special_aware_miners.add(uid)
                        bt.logging.info(f"UID={uid} identified special marker")
                
                # Update history
                if uid not in self.miner_history:
                    self.miner_history[uid] = []
                    
                self.miner_history[uid].append({
                    'timestamp': time.time(),
                    'correct': is_correct,
                    'special': is_special,
                    'identified_special': identified_special
                })
                
                # Limit history length
                if len(self.miner_history[uid]) > 100:
                    self.miner_history[uid] = self.miner_history[uid][-100:]
                
                # Detect model usage
                uses_model = self.detect_model_usage(uid)
                self.miner_model_status[uid] = uses_model
                
                # Get balance
                balance = float(self.metagraph.S[uid]) * 1000
                
                # Calculate score
                score = self.calculate_score(uid, is_correct, uses_model, balance)
                
                # Update score in tensor directly
                if uid < len(self.scores):
                    self.scores[uid] = float(score)
                
                bt.logging.info(f"UID={uid} query result: correct={is_correct}, model={uses_model}, score={score}")
            else:
                bt.logging.warning(f"UID={uid} invalid response format")
                if uid < len(self.scores):
                    self.scores[uid] = 0.0
            
        except asyncio.TimeoutError:
            bt.logging.warning(f"UID={uid} query timeout")
            if uid < len(self.scores):
                self.scores[uid] = 0.0
        except Exception as e:
            bt.logging.error(f"UID={uid} query error: {e}")
            if uid < len(self.scores):
                self.scores[uid] = 0.0

    def set_weights(self):
        """Override BaseValidatorNeuron's set_weights to ensure compatibility"""
        try:
            n = self.metagraph.n
            
            # Resize scores tensor if needed
            if len(self.scores) != n:
                new_scores = torch.zeros(n, dtype=torch.float32)
                # Copy existing scores
                for i in range(min(len(self.scores), n)):
                    new_scores[i] = self.scores[i]
                self.scores = new_scores
            
            # Convert active to tensor if needed
            active = self.metagraph.active
            if isinstance(active, np.ndarray):
                active = torch.tensor(active, dtype=torch.bool)
            else:
                active = active.to(torch.bool)
                
            # Ensure all inactive miners get 0 weight
            inactive_mask = ~active
            self.scores[inactive_mask] = 0.0
            
            # Print weight statistics
            zeros = torch.sum(self.scores == 0).item()
            ones = torch.sum(self.scores == 1.0).item()
            bt.logging.info(f"Weight stats: zeros={zeros}, ones={ones}, active={active.sum().item()}")
            
            # If all weights are 0, randomly set one to 1
            if self.scores.sum().item() == 0:
                active_indices = torch.nonzero(active).flatten().tolist()
                if active_indices:
                    random_uid = random.choice(active_indices)
                    self.scores[random_uid] = 1.0
                    bt.logging.info(f"All weights are 0, randomly setting UID={random_uid} to 1.0")
                
            # Set weights to chain
            bt.logging.info(f"Setting weights to chain, sum={self.scores.sum().item()}")
            
            # Create proper uids tensor
            uids = torch.arange(0, n, dtype=torch.long)
            
            # Note: w are already properly computed in self.scores
            result = self.subtensor.set_weights(
                netuid=self.config.netuid,
                wallet=self.wallet,
                uids=uids,
                weights=self.scores,
                wait_for_inclusion=False
            )
            
            if result:
                bt.logging.info(f"Weight setting successful")
                self.last_weight_set_time = time.time()
            else:
                bt.logging.error(f"Weight setting failed")
                
        except Exception as e:
            bt.logging.error(f"set_weights error: {e}")
            # Update timestamp even on error to prevent rapid retries
            self.last_weight_set_time = time.time()

    def save_state(self):
        """Save state to file"""
        try:
            # Convert tensor to list for saving
            scores_dict = {}
            for i in range(len(self.scores)):
                if self.scores[i] > 0:  # Only save non-zero scores to save space
                    scores_dict[str(i)] = float(self.scores[i])
            
            state = {
                "scores": scores_dict,
                "miner_history": self.miner_history,
                "miner_model_status": self.miner_model_status,
                "special_aware_miners": list(self._special_aware_miners),
                "round": self._state.get("round", 0),
                "last_weight_set_time": self.last_weight_set_time
            }
            path = os.path.join(self.config.neuron.full_path, "validator_state.json")
            with open(path, "w") as f:
                import json
                json.dump(state, f, default=str)  # Use default=str for non-serializable objects
            bt.logging.info("State saved successfully")
        except Exception as e:
            bt.logging.error(f"save_state error: {e}")

    def load_state(self):
        """Load state from file"""
        try:
            path = os.path.join(self.config.neuron.full_path, "validator_state.json")
            if os.path.isfile(path):
                with open(path, "r") as f:
                    import json
                    state = json.load(f)
                
                # Load scores back into tensor
                scores_dict = state.get("scores", {})
                for uid_str, score in scores_dict.items():
                    try:
                        uid = int(uid_str)
                        if uid < len(self.scores):
                            self.scores[uid] = float(score)
                    except (ValueError, IndexError):
                        continue
                
                self.miner_history = state.get("miner_history", {})
                self.miner_model_status = state.get("miner_model_status", {})
                self._special_aware_miners = set(state.get("special_aware_miners", []))
                self._state["round"] = state.get("round", 0)
                self.last_weight_set_time = state.get("last_weight_set_time", 0)
                
                bt.logging.info("State loaded successfully")
        except Exception as e:
            bt.logging.error(f"load_state error: {e}")

    async def concurrent_forward(self):
        """
        This method is required by BaseValidatorNeuron
        """
        try:
            # Just call run_step which contains our main logic
            await self.run_step()
        except Exception as e:
            bt.logging.error(f"concurrent_forward error: {e}")
            await asyncio.sleep(30)

if __name__ == "__main__":
    # Ensure required modules are imported
    config = get_config()
    config.neuron.full_path = os.path.expanduser(os.path.dirname(os.path.abspath(__file__)))
    validator = Validator(config=config)
    validator.run()
