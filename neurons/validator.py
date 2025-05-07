#!/usr/bin/env python3
# MIT License
# Copyright © 2025 MIAO

import time
import random
import base64
import os
import asyncio
import traceback
import bittensor as bt
import numpy as np
import torch
from typing import List, Dict, Tuple, Union
from template.base.validator import BaseValidatorNeuron
from template.protocol import CatSoundProtocol

def get_config():
    """
    Get the configuration for the validator.
    """
    import argparse
    parser = argparse.ArgumentParser()
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    parser.add_argument("--netuid", type=int, default=86, help="Subnet ID")
    parser.add_argument("--neuron.validation_interval", type=int, default=60, help="Validation interval (seconds)")
    parser.add_argument("--neuron.sample_size", type=int, default=10, help="Number of miners to sample per round")
    parser.add_argument("--audio.miao_dir", type=str, default="audio/miao", help="Directory containing miao sound samples")
    parser.add_argument("--audio.other_dir", type=str, default="audio/other", help="Directory containing other sound samples")
    return bt.config(parser)

class Validator(BaseValidatorNeuron):
    """
    Validator for the MIAO subnet.
    """
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
        
        # Initialize base class - Important: this will set self.scores as numpy array
        super(Validator, self).__init__(config=config)
        
        # Ensure dendrite is properly initialized
        if not hasattr(self, 'dendrite') or self.dendrite is None:
            bt.logging.warning("Dendrite not initialized in base class, creating manually...")
            self.dendrite = bt.dendrite(wallet=self.wallet)
        
        # Ensure subtensor is properly initialized
        if not hasattr(self, 'subtensor') or self.subtensor is None:
            bt.logging.info(f"Initializing subtensor connection to network: {self.config.netuid.network}")
            self.subtensor = bt.subtensor(network=self.config.netuid.network)
        
        # Test samples database
        self.test_database = [
            ("cat_easy",     True),
            ("cat_medium",   True),
            ("cat_hard",     True),
            ("not_cat_easy", False),
            ("not_cat_med",  False),
            ("not_cat_hard", False),
        ]
        
        # Try to load audio samples if available
        try:
            # Check if audio directories exist
            miao_dir = self.config.audio.miao_dir
            other_dir = self.config.audio.other_dir
            
            if os.path.isdir(miao_dir) and os.path.isdir(other_dir):
                bt.logging.info(f"Loading audio samples from {miao_dir} and {other_dir}")
                # Would implement audio sample loading here
                self.has_audio_samples = True
            else:
                bt.logging.warning(f"Audio directories not found: {miao_dir} or {other_dir}")
                self.has_audio_samples = False
        except Exception as e:
            bt.logging.warning(f"Could not load audio samples: {e}")
            self.has_audio_samples = False
        
        self.load_state()
        bt.logging.info("Validator initialization complete")
        
        # Ensure metagraph is synchronized
        if not hasattr(self, 'metagraph'):
            bt.logging.info("Initializing metagraph...")
            self.metagraph = bt.metagraph(netuid=self.config.netuid, sync=False)
        
        bt.logging.info("Starting initial metagraph sync...")
        try:
            self.metagraph.sync(subtensor=self.subtensor)
            bt.logging.info(f"Initial sync completed: total nodes={self.metagraph.n}, current block={self.metagraph.block}")
        except Exception as e:
            bt.logging.warning(f"Initial sync failed, will retry during runtime: {e}")

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
                        
                        # Calculate score and update scores array directly
                        score = self.calculate_score(uid, is_correct, uses_model, balance)
                        
                        # Update score in numpy array
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
        """Standard run step, ensuring proper metagraph synchronization"""
        try:
            # Ensure subtensor is correctly initialized
            if self.subtensor is None:
                self.subtensor = bt.subtensor(network=self.config.netuid.network)
                bt.logging.info(f"Initialized subtensor connection to network: {self.config.netuid.network}")
            
            # Comprehensive metagraph synchronization
            bt.logging.info("Starting metagraph synchronization...")
            try:
                # Ensure using correct subnet ID
                self.metagraph.sync(subtensor=self.subtensor)
                
                # Get detailed sync status information
                active_count = torch.sum(self.metagraph.active).item() if isinstance(self.metagraph.active, torch.Tensor) else sum(self.metagraph.active)
                bt.logging.info(f"Metagraph sync completed: total nodes={self.metagraph.n}, active nodes={active_count}, current block={self.metagraph.block}")
                
                # Check if first few axon information is valid
                if self.metagraph.n > 0:
                    for i in range(min(3, self.metagraph.n)):
                        axon = self.metagraph.axons[i]
                        hotkey = axon.hotkey if hasattr(axon, 'hotkey') and axon.hotkey else 'None'
                        bt.logging.info(f"Axon example UID={i}: ip={axon.ip}, port={axon.port}, hotkey={hotkey[:10]}...")
            except Exception as e:
                bt.logging.error(f"Metagraph sync failed: {e}")
                # Wait and continue rather than exiting directly if sync fails
                await asyncio.sleep(10)
                
            # Ensure weights are set
            current_time = time.time()
            should_set_weights = (current_time - self.last_weight_set_time) > 1800
            
            if should_set_weights or self._state.get("round", 0) % 10 == 0:
                self._ensure_scores_are_valid()
                self.set_weights()
                self.save_state()
                self.last_weight_set_time = time.time()
                
            # Query miner nodes
            await self.query_miners()
            
            # Update round counter
            self._state["round"] = self._state.get("round", 0) + 1
            
            # Wait for next round
            await asyncio.sleep(self.config.neuron.validation_interval)
            
        except Exception as e:
            bt.logging.error(f"Run step error: {e}")
            # Wait longer on error to avoid rapid restarts
            await asyncio.sleep(60)

    def _ensure_scores_are_valid(self):
        """Ensure scores are in the correct format before setting weights"""
        n = self.metagraph.n
        
        # Ensure scores is a numpy array of the correct shape
        if not isinstance(self.scores, np.ndarray):
            bt.logging.warning(f"Converting scores from {type(self.scores)} to numpy array")
            
            # If it's a torch tensor, convert to numpy
            if isinstance(self.scores, torch.Tensor):
                self.scores = self.scores.cpu().detach().numpy()
            else:
                # Create new numpy array
                new_scores = np.zeros(n, dtype=np.float32)
                
                # If it's a dictionary, try to extract values
                if isinstance(self.scores, dict):
                    for uid_str, score in self.scores.items():
                        try:
                            uid = int(uid_str)
                            if 0 <= uid < n:
                                new_scores[uid] = float(score)
                        except (ValueError, IndexError):
                            continue
                self.scores = new_scores
        
        # Ensure scores array has the correct shape
        if len(self.scores) != n:
            bt.logging.warning(f"Resizing scores from {len(self.scores)} to {n}")
            new_scores = np.zeros(n, dtype=np.float32)
            # Copy existing scores
            for i in range(min(len(self.scores), n)):
                new_scores[i] = self.scores[i]
            self.scores = new_scores
        
        # Ensure all scores are properly bounded
        self.scores = np.clip(self.scores, 0.0, 1.0)
        
        # Apply inactive mask - ensure all inactive miners have 0 score
        active = self.metagraph.active
        if isinstance(active, torch.Tensor):
            active = active.cpu().numpy()
        
        # Set scores for inactive miners to 0
        inactive_mask = ~active
        self.scores[inactive_mask] = 0.0
        
        # Log score statistics
        num_zeros = np.sum(self.scores == 0.0)
        num_ones = np.sum(self.scores == 1.0)
        num_active = np.sum(active)
        num_scored = np.sum(self.scores > 0.0)
        
        bt.logging.info(f"Score statistics: zeros={num_zeros}, ones={num_ones}, active nodes={num_active}, scored nodes={num_scored}")

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
                
            active_count = torch.sum(active).item()
            bt.logging.info(f"Metagraph status: total nodes={self.metagraph.n}, active nodes={active_count}")
            
            # If no active nodes, log warning and return
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
                
                # Log the miner that will be queried
                bt.logging.info(f"Preparing to query UID={uid}: ip={axon.ip}, port={axon.port}, hotkey={axon.hotkey[:10] if axon.hotkey else 'None'}...")
                
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
            bt.logging.info(f"Starting query to UID={uid}, address={axon.ip}:{axon.port}")
            
            # Use dendrite.forward to send request to miner's axon
            response = await asyncio.wait_for(
                self.dendrite.forward(axon, synapse, deserialize=True),
                timeout=10.0
            )
            
            bt.logging.info(f"Successfully received response from UID={uid}")
            
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
                
                # Update score in numpy array
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
            bt.logging.error(f"UID={uid} query error: {e}, axon={axon}")
            if uid < len(self.scores):
                self.scores[uid] = 0.0

    def set_weights(self):
        """Sets weights on chain using standard methods."""
        try:
            # Ensure scores are valid before setting weights
            self._ensure_scores_are_valid()
            
            # Log weight statistics
            nonzero_count = np.sum(self.scores > 0)
            bt.logging.info(f"Setting weights: {nonzero_count} non-zero weights out of {len(self.scores)}")
            
            # Standard bittensor weights setting with retries
            try:
                # Try normalized weights for robustness
                success = self.subtensor.set_weights(
                    netuid=self.config.netuid,
                    wallet=self.wallet,
                    uids=self.metagraph.uids,
                    weights=self.scores,
                    wait_for_inclusion=False,
                    version_key=getattr(self, 'spec_version', None)
                )
                if success:
                    bt.logging.success(f"Successfully set {nonzero_count} weights on chain")
                    self.last_weight_set_time = time.time()
                else:
                    bt.logging.warning("Failed to set weights on chain")
            except Exception as e:
                bt.logging.error(f"Error setting weights: {e}")
                
        except Exception as e:
            bt.logging.error(f"set_weights error: {e}")

    def save_state(self):
        """Save state to file"""
        try:
            # Convert to simple types for saving
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
            bt.logging.error(f"Save state error: {e}")

    def load_state(self):
        """Load state from file"""
        try:
            path = os.path.join(self.config.neuron.full_path, "validator_state.json")
            if os.path.isfile(path):
                with open(path, "r") as f:
                    import json
                    state = json.load(f)
                
                # Load scores back into numpy array
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
            bt.logging.error(f"Load state error: {e}")

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
    
    # Create and run the validator
    bt.logging.info("Starting MIAO validator...")
    validator = Validator(config=config)
    validator.run()
