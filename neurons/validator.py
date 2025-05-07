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
    parser.add_argument("--neuron.validation_interval", type=int, default=30, help="Validation interval (seconds)")
    parser.add_argument("--neuron.sample_size", type=int, default=10, help="Number of miners to sample per round")
    return bt.config(parser)

class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        # If no config is provided, use get_config()
        if config is None:
            config = get_config()
            
        super(Validator, self).__init__(config=config)

        # Score table: uid -> score
        self.scores = {}
        # Miner state tracking
        self.miner_history = {}  # {uid: [{'timestamp':, 'correct':, 'special':}]}
        self.miner_model_status = {}  # {uid: bool} - whether running a real model
        # Minimum dTAO balance requirement
        self.min_dtao_balance = 50.0
        # Special marker to detect if miners are running real models
        self._special_mark = "MIAO_SPECIAL_MARK_2025"
        self._special_aware_miners = set()  # miners that recognize the special marker
        # Last time weights were set
        self.last_weight_set_time = 0
        
        # Test samples database
        self.test_database = [
            ("cat_easy",     True),
            ("cat_medium",   True),
            ("cat_hard",     True),
            ("not_cat_easy", False),
            ("not_cat_med",  False),
            ("not_cat_hard", False),
        ]
        
        self._state = {"round": 0}
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
        synapse = CatSoundProtocol()
        # 20% chance of special test to detect cheating
        special = (random.random() < 0.2)
        tid, is_cat = self.select_test_sample()
        encoded = self._encode_test(tid, special)
        
        synapse.audio_data = encoded
        synapse.sample_id = tid
        synapse.ground_truth = is_cat
        synapse.is_special = special
        
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
        """
        Standard forward method called by BaseValidatorNeuron:
        1) Process previous response
        2) Prepare and send new request
        """
        try:
            # Check if we have a hotkey (processing previous response)
            hotkey = getattr(synapse.dendrite, 'hotkey', None)
            if hotkey:
                try:
                    uid = self.metagraph.hotkeys.index(hotkey)
                    
                    # Check for prediction results
                    if hasattr(synapse, 'response') and hasattr(synapse.response, 'predictions'):
                        preds = synapse.response.predictions
                        if isinstance(preds, list) and len(preds) > 0:
                            pred = preds[0]
                            if isinstance(pred, dict) and 'is_cat' in pred:
                                # Get prediction and ground truth
                                is_cat_pred = pred['is_cat']
                                is_cat_true = getattr(synapse, 'ground_truth', None)
                                
                                if is_cat_true is not None:
                                    # Determine correctness
                                    is_correct = (is_cat_pred == is_cat_true)
                                    
                                    # Check special marker
                                    is_special = getattr(synapse, 'is_special', False)
                                    identified_special = False
                                    
                                    if is_special and hasattr(synapse.response, 'special_marker'):
                                        special_marker = synapse.response.special_marker
                                        if special_marker and self._special_mark in special_marker:
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
                                    self.scores[str(uid)] = score
                                    
                                    bt.logging.info(f"UID={uid} result: correct={is_correct}, model={uses_model}, score={score}")
                                else:
                                    bt.logging.warning(f"UID={uid} response missing ground_truth")
                                    self.scores[str(uid)] = 0.0
                            else:
                                bt.logging.warning(f"UID={uid} invalid prediction format")
                                self.scores[str(uid)] = 0.0
                        else:
                            bt.logging.warning(f"UID={uid} invalid predictions list")
                            self.scores[str(uid)] = 0.0
                    else:
                        bt.logging.warning(f"UID={uid} response missing predictions")
                        self.scores[str(uid)] = 0.0
                
                except ValueError:
                    bt.logging.warning(f"Hotkey={hotkey} not found in metagraph")
                except Exception as e:
                    bt.logging.error(f"Error processing response: {e}")
            
            # Prepare new request
            special = (random.random() < 0.2)
            tid, is_cat = self.select_test_sample()
            encoded = self._encode_test(tid, special)
            
            synapse.audio_data = encoded
            synapse.sample_id = tid
            synapse.ground_truth = is_cat
            synapse.is_special = special
            
            return synapse
            
        except Exception as e:
            bt.logging.error(f"Forward error: {e}")
            return synapse

    async def run_step(self):
        """Standard run step for BaseValidatorNeuron"""
        try:
            # Sync metagraph
            self.metagraph.sync(subtensor=self.subtensor)
            bt.logging.info(f"Metagraph synced, miners={self.metagraph.n}")
            
            # Check if weights should be set
            current_time = time.time()
            should_set_weights = (current_time - self.last_weight_set_time) > 1800  # 30 minutes
            
            if should_set_weights or self._state.get("round", 0) % 10 == 0:
                # Set weights
                self.set_weights_direct()
                self.save_state()
                
            # Query miners
            await self.query_miners()
            
            # Update round counter
            self._state["round"] = self._state.get("round", 0) + 1
            
            # Add appropriate sleep
            await asyncio.sleep(max(15, self.config.neuron.validation_interval))
            
        except Exception as e:
            bt.logging.error(f"Run step error: {e}")
            await asyncio.sleep(30)  # Longer sleep to prevent rapid restart loops

    async def query_miners(self):
        """Query a batch of random miners"""
        try:
            # Determine query count
            n = min(self.config.neuron.sample_size, self.metagraph.n)
            if n <= 0:
                bt.logging.warning("No miners available to query")
                return
                
            # Find active miners - convert numpy array to tensor if needed
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
                
                # Set timeout mechanism
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
            
            # Process response
            if hasattr(response, 'predictions'):
                preds = response.predictions
                if isinstance(preds, list) and len(preds) > 0:
                    pred = preds[0]
                    if isinstance(pred, dict) and 'is_cat' in pred:
                        # Get prediction
                        is_cat_pred = pred['is_cat']
                        is_cat_true = synapse.ground_truth
                        is_correct = (is_cat_pred == is_cat_true)
                        
                        # Check special marker
                        is_special = getattr(synapse, 'is_special', False)
                        identified_special = False
                        
                        if is_special and hasattr(response, 'special_marker'):
                            special_marker = response.special_marker
                            if special_marker and self._special_mark in special_marker:
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
                        self.scores[str(uid)] = score
                        
                        bt.logging.info(f"UID={uid} query result: correct={is_correct}, model={uses_model}, score={score}")
                    else:
                        bt.logging.warning(f"UID={uid} invalid prediction format")
                        self.scores[str(uid)] = 0.0
                else:
                    bt.logging.warning(f"UID={uid} invalid predictions list")
                    self.scores[str(uid)] = 0.0
            else:
                bt.logging.warning(f"UID={uid} response missing predictions")
                self.scores[str(uid)] = 0.0
            
        except asyncio.TimeoutError:
            bt.logging.warning(f"UID={uid} query timeout")
            self.scores[str(uid)] = 0.0
        except Exception as e:
            bt.logging.error(f"UID={uid} query error: {e}")
            self.scores[str(uid)] = 0.0

    def set_weights_direct(self):
        """Set weights directly to chain without standard normalization"""
        try:
            n = self.metagraph.n
            
            # Convert active to tensor if it's numpy array
            active = self.metagraph.active
            if isinstance(active, np.ndarray):
                active = torch.tensor(active, dtype=torch.bool)
            else:
                active = active.to(torch.bool)
                
            w = torch.zeros(n, dtype=torch.float32)

            # Set weights directly from scores, no normalization
            bt.logging.info("Setting direct weights (no normalization):")
            
            # Ensure all inactive miners get 0 weight
            inactive_mask = ~active
            w[inactive_mask] = 0.0
            
            # Set active miner weights
            for uid_str, score in self.scores.items():
                try:
                    uid = int(uid_str)
                    if 0 <= uid < n and active[uid]:
                        w[uid] = float(score)
                except (ValueError, TypeError):
                    continue
                    
            # Print weight statistics
            zeros = torch.sum(w == 0).item()
            ones = torch.sum(w == 1.0).item()
            bt.logging.info(f"Weight stats: zeros={zeros}, ones={ones}, active={active.sum().item()}")
            
            # Print sample weights
            active_indices = torch.nonzero(active).flatten().tolist()
            if active_indices:
                samples = random.sample(active_indices, min(5, len(active_indices)))
                for uid in samples:
                    bt.logging.info(f"Sample weight UID={uid}: {w[uid].item()}")
                    
            # If all weights are 0, randomly set one to 1
            if w.sum().item() == 0 and len(active_indices) > 0:
                random_uid = random.choice(active_indices)
                w[random_uid] = 1.0
                bt.logging.info(f"All weights are 0, randomly setting UID={random_uid} to 1.0")
                
            # Set weights to chain
            bt.logging.info(f"Setting weights to chain, sum={w.sum().item()}")
            result = self.subtensor.set_weights(
                netuid=self.config.netuid,
                wallet=self.wallet,
                uids=torch.arange(n, dtype=torch.long),
                weights=w,
                wait_for_inclusion=False
            )
            
            if result:
                bt.logging.info(f"Weight setting successful")
                self.last_weight_set_time = time.time()
            else:
                bt.logging.error(f"Weight setting failed")
                
        except Exception as e:
            bt.logging.error(f"set_weights_direct error: {e}")
            # Update timestamp even on error to prevent rapid retries
            self.last_weight_set_time = time.time()

    def save_state(self):
        """Save state to file"""
        try:
            state = {
                "scores": self.scores,
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
                    
                self.scores = state.get("scores", {})
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
        This method is called by BaseValidatorNeuron's run() method.
        We implement it to avoid compatibility issues with the base class.
        """
        try:
            # In case metagraph hasn't been synced yet
            if not hasattr(self.metagraph, 'n') or self.metagraph.n == 0:
                self.metagraph.sync(subtensor=self.subtensor)
                
            # Just delegate to run_step which has our main logic
            await self.run_step()
            
        except Exception as e:
            bt.logging.error(f"concurrent_forward error: {e}")
            await asyncio.sleep(30)  # Add delay to prevent rapid restarts

if __name__ == "__main__":
    # Ensure required modules are imported
    config = get_config()
    config.neuron.full_path = os.path.expanduser(os.path.dirname(os.path.abspath(__file__)))
    validator = Validator(config=config)
    validator.run()
