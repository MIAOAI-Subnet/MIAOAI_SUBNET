import argparse
import os
from typing import Optional

import logging as python_logging

from bittensor import Subtensor, config, logging, axon
from bittensor_wallet import Wallet
from tabulate import tabulate


from miaoai.core.constants import BLOCK_TIME, DEFAULT_BLACKLIST, TESTNET_NETUID

from miaoai.validator.storage import (
    RedisValidatorStorage,
    get_validator_storage,
)

class BaseValidator:
    def __init__(self):
        self.config = self.get_config()

        blacklist_str = os.getenv("VALIDATOR_BLACKLIST", "")
        self.config.blacklist = [addr.strip() for addr in
                                 blacklist_str.split(",")] if blacklist_str else DEFAULT_BLACKLIST

        self.setup_logging_path()
        self.setup_logging()
        self.storage = get_validator_storage(
            storage_type=self.config.storage, config=self.config
        )

        self.subtensor = None
        self.wallet = None
        self.metagraph = None
        self.metagraph_info = None
        self.tempo = None
        self.uid = None
        self.weights_interval = None

        self.eval_interval = self.config.eval_interval

        self.last_update = 0
        self.current_block = 0
        self.scores = []
        self.moving_avg_scores = []
        self.hotkeys = []
        self.block_at_registration = []

    def get_config(self):
        parser = argparse.ArgumentParser()
        self.add_args(parser)
        return config(parser)

    def add_args(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--worker_prefix",
            required=False,
            default="",
            help="A prefix for the workers names miners will use.",
        )
        parser.add_argument(
            "--netuid",
            type=int,
            default=os.getenv("NETUID", TESTNET_NETUID),
            help="The chain subnet uid.",
        )
        parser.add_argument(
            "--subtensor.network",
            default=os.getenv("SUBTENSOR_NETWORK", "finney"),
            help="The chain subnet network.",
        )
        parser.add_argument(
            "--axon.ip",
            type=str,
            default=os.getenv("AXON_IP", "0.0.0.0"),
            help="The IP address to serve the axon on.",
        )
        parser.add_argument(
            "--axon.port",
            type=int,
            default=int(os.getenv("AXON_PORT", "8092")),
            help="The port to serve the axon on.",
        )
        parser.add_argument(
            "--min_blocks_per_validator",
            type=int,
            default=int(os.getenv("MIN_BLOCKS_PER_VALIDATOR", 40)),
            help="Minimum blocks required per validator",
        )
        parser.add_argument(
            "--eval_interval",
            type=int,
            default=30,
            help="The interval on which to run evaluation across the metagraph.",
        )
        parser.add_argument(
            "--state",
            type=str,
            choices=["restore", "fresh"],
            default="restore",
            help="Whether to restore previous validator state ('restore') or start fresh ('fresh').",
        )
        parser.add_argument(
            "--storage",
            type=str,
            choices=["json", "redis"],
            default=os.getenv("STORAGE_TYPE", "redis"),
            help="Storage type to use (json or redis)",
        )

        parser.add_argument(
            "--logging.info",
            action="store_true",
            help="Enable debug logging.",
            default=False
        )
        parser.add_argument(
            "--logging.record_log",
            action="store_true",
            help="Enable recording logs to file.",
            default=True
        )
        parser.add_argument(
            "--logging.level",
            type=str,
            choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            default=os.getenv("BT_LOGGING_INFO", "INFO"),
            help="Logging level."
        )

        parser.add_argument(
            "--neuron.axon_off",
            "--axon_off",
            action="store_true",
            help="Set this flag to not attempt to serve an Axon.",
            default=False,
        )

        Subtensor.add_args(parser)
        logging.add_args(parser)
        Wallet.add_args(parser)
        RedisValidatorStorage.add_args(parser)

    def setup_logging_path(self) -> None:
        self.config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                self.config.logging.logging_dir,
                self.config.wallet.name,
                self.config.wallet.hotkey,
                self.config.netuid,
                "validator",
            )
        )
        os.makedirs(self.config.full_path, exist_ok=True)

    def setup_logging(self) -> None:
        self.config.logging.logging_dir = self.config.full_path
        self.config.logging.record_log = True
        logging(config=self.config)

        if self.config.logging.level == "TRACE":
            logging.set_trace(True)
        else:
            logging.set_trace(False)
            if self.config.logging.level == "DEBUG":
                logging.debug()
            elif self.config.logging.level == "INFO":
                logging.info()
            elif self.config.logging.level == "WARNING":
                logging.warning()
            elif self.config.logging.level == "ERROR":
                logging.error()
            elif self.config.logging.level == "CRITICAL":
                logging.critical()
        
        python_logging.getLogger("bittensor").setLevel(
            getattr(python_logging, self.config.logging.level)
        )

    def setup_bittensor_objects(self) -> None:

        self.wallet = Wallet(config=self.config)

        self.subtensor = Subtensor(config=self.config)

        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        self.metagraph_info = self.subtensor.get_metagraph_info(self.config.netuid)
        self.metagraph.sync(subtensor=self.subtensor)  # 同步最新状态

        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            logging.error(
                f"\nYour validator: {self.wallet}"
                f" is not registered to chain connection: {self.subtensor}"
                f"\nRun 'btcli register' and try again."
            )
            exit()
        else:
            self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        self.current_block = self.metagraph.block
        self.hotkeys = self.metagraph.hotkeys
        self.block_at_registration = self.metagraph.block_at_registration
        self.scores = [0.0] * len(self.metagraph.total_stake)
        self.moving_avg_scores = [0.0] * len(self.metagraph.total_stake)
        self.tempo = self.subtensor.tempo(self.config.netuid)

        if not self.config.neuron.axon_off:
            self.serve_axon()
        else:
            logging.warning("axon off, not serving ip to chain.")

    def save_state(self) -> None:
        state = {
            "scores": self.scores,
            "moving_avg_scores": self.moving_avg_scores,
            "hotkeys": self.hotkeys,
            "block_at_registration": self.block_at_registration,
            "current_block": self.current_block,
        }
        self.storage.save_state(state)

    def resync_metagraph(self) -> None:

        previous_hotkeys = self.hotkeys

        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        self.metagraph.sync(subtensor=self.subtensor)  # 同步最新状态
        self.current_block = self.metagraph.block

        if previous_hotkeys == self.metagraph.hotkeys:
            return

        for uid, hotkey in enumerate(previous_hotkeys):
            if (
                uid < len(self.metagraph.hotkeys)
                and hotkey != self.metagraph.hotkeys[uid]
            ):
                self.scores[uid] = 0.0
                self.moving_avg_scores[uid] = 0.0

        if len(previous_hotkeys) < len(self.metagraph.hotkeys):
            old_size = len(previous_hotkeys)
            new_size = len(self.metagraph.hotkeys)

            new_scores = [0.0] * new_size
            new_moving_avg = [0.0] * new_size

            for i in range(min(old_size, len(self.scores))):
                new_scores[i] = self.scores[i]
                new_moving_avg[i] = self.moving_avg_scores[i]

            self.scores = new_scores
            self.moving_avg_scores = new_moving_avg

        self.hotkeys = self.metagraph.hotkeys
        self.block_at_registration = self.metagraph.block_at_registration

    def get_burn_uid(self) -> Optional[int]:
        sn_owner_hotkey = self.subtensor.query_subtensor(
            "SubnetOwnerHotkey",
            params=[self.config.netuid],
        )
        owner_uid = self.metagraph.hotkeys.index(sn_owner_hotkey)
        return owner_uid

    def get_next_sync_block(self) -> tuple[int, str]:
        sync_reason = "Regular sync"
        next_sync = self.current_block + self.eval_interval

        blocks_since_last_weights = self.subtensor.blocks_since_last_update(
            self.config.netuid, self.uid
        )
        blocks_until_weights = self.weights_interval - blocks_since_last_weights
        next_weights_block = self.current_block + blocks_until_weights + 1

        if blocks_since_last_weights >= self.weights_interval:
            sync_reason = "Weights due"
            return self.current_block + 1, sync_reason

        elif next_weights_block <= next_sync:
            sync_reason = "Weights due"
            return next_weights_block, sync_reason

        return next_sync, sync_reason

    def ensure_validator_permit(self) -> None:

        validator_permits = self.subtensor.query_subtensor(
            "ValidatorPermit",
            params=[self.config.netuid],
        ).value
        if not validator_permits[self.uid]:
            blocks_since_last_step = self.subtensor.query_subtensor(
                "BlocksSinceLastStep",
                block=self.current_block,
                params=[self.config.netuid],
            ).value
            time_to_wait = (self.tempo - blocks_since_last_step) * BLOCK_TIME + 0.1
            logging.error(
                f"Validator permit not found. Waiting {time_to_wait} seconds."
            )
            target_block = self.current_block + (self.tempo - blocks_since_last_step)
            self.subtensor.wait_for_block(target_block)

    def serve_axon(self):

        try:
            if not self.config.axon.ip:
                self.config.axon.ip = "0.0.0.0"

            if not self.config.axon.port:
                self.config.axon.port = 8091

            self.axon = axon(wallet=self.wallet, config=self.config)
            self.axon.start()

            try:
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )

            except Exception as e:
                logging.error(f"Failed to serve Axon with exception: {e}")
                pass

        except Exception as e:
            logging.error(
                f"Failed to create Axon initialize with exception: {e}"
            )
            pass
    def _log_weights_and_scores(self, weights: list[float]) -> None:
        rows = []
        headers = ["UID", "Hotkey", "Moving Avg", "Weight", "Normalized (%)"]

        sorted_indices = sorted(
            range(len(weights)), key=lambda w: weights[w], reverse=True
        )

        for i in sorted_indices:
            if weights[i] > 0 or self.moving_avg_scores[i] > 0:
                hotkey = self.metagraph.hotkeys[i]
                rows.append(
                    [
                        i,
                        f"{hotkey}",
                        f"{self.moving_avg_scores[i]:.8f}",
                        f"{weights[i]:.8f}",
                        f"{weights[i] * 100:.2f}%",
                    ]
                )

        if not rows:
            return

        table = tabulate(
            rows, headers=headers, tablefmt="grid", numalign="right", stralign="left"
        )
        title = f"Weights set at Block: {self.current_block}"
        # logging.info(f"{title}\n{table}")

    def _log_scores(self, coin: str, hash_price: float) -> None:
        rows = []
        headers = ["UID", "Hotkey", "Score", "Moving Avg"]

        sorted_indices = sorted(
            range(len(self.scores)), key=lambda s: self.scores[s], reverse=True
        )

        for i in sorted_indices:
            if self.scores[i] > 0 or self.moving_avg_scores[i] > 0:
                hotkey = self.metagraph.hotkeys[i]
                rows.append(
                    [
                        i,
                        f"{hotkey}",
                        f"{self.scores[i]:.8f}",
                        f"{self.moving_avg_scores[i]:.8f}",
                    ]
                )

        if not rows:
            return

        table = tabulate(
            rows, headers=headers, tablefmt="grid", numalign="right", stralign="left"
        )

        title = f"Current Mining Scores - Block {self.current_block} - {coin.upper()} (Hash Price: ${hash_price:.8f})"
