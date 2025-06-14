import argparse
import os
import logging as python_logging

from bittensor import Subtensor, config, logging, Dendrite, axon
from bittensor_wallet.bittensor_wallet import Wallet

from miaoai.miner.storage import get_miner_storage,  BaseRedisStorage

DEFAULT_SYNC_FREQUENCY = 6

class BaseMiner:
    def __init__(self):
        self.config = self.get_config()
        self.setup_logging_path()
        self.setup_logging()

        self.subtensor = None
        self.wallet = None
        self.metagraph = None
        self.metagraph_info = None
        self.uid = None
        self._dendrite = None

        self.setup_bittensor_objects()
        self.storage = get_miner_storage(storage_type=self.config.storage, config=self.config)
        self.worker_id = self.create_worker_id()
        self.tempo = self.subtensor.tempo(self.config.netuid)
        self.current_block = 0
        self.blocks_per_sync = self.tempo // self.config.sync_frequency

        self._first_sync = True
        self._recover_schedule = self.config.recover_schedule

    def get_config(self):
        parser = argparse.ArgumentParser()
        self.add_args(parser)
        return config(parser)

    def setup_logging_path(self) -> None:
        self.config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                self.config.logging.logging_dir,
                self.config.wallet.name,
                self.config.wallet.hotkey,
                self.config.netuid,
                "miner",
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

    def add_args(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--netuid",
            type=int,
            default=os.getenv("NETUID", 115),
            help="The chain subnet uid.",
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
            default=int(os.getenv("AXON_PORT", "8091")),
            help="The port to serve the axon on.",
        )
        parser.add_argument(
            "--subtensor.network",
            default=os.getenv("SUBTENSOR_NETWORK", "finney"),
            help="The chain subnet network.",
        )
        parser.add_argument(
            "--sync_frequency",
            type=int,
            default=os.getenv("SYNC_FREQUENCY", DEFAULT_SYNC_FREQUENCY),
            help=f"Number of times to sync and update pool info per epoch (1-359). Default is {DEFAULT_SYNC_FREQUENCY} times per epoch.",
        )
        parser.add_argument(
            "--no-recover_schedule",
            action="store_false",
            dest="recover_schedule",
            default=os.getenv("RECOVER_SCHEDULE", "true").lower() == "true",
            help="Disable schedule recovery between restarts.",
        )
        parser.add_argument(
            "--blacklist",
            type=str,
            nargs="+",
            default=os.getenv("BLACKLIST", "").split(",")
            if os.getenv("BLACKLIST")
            else [],
            help="List of validator hotkeys to exclude from mining",
        )

        parser.add_argument(
            "--logging.info",
            action="store_true",
            help="Enable debug logging output",
        )
        parser.add_argument(
            "--logging.level",
            type=str,
            choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            default=os.getenv("BT_LOGGING_INFO", "INFO"),
            help="Logging level."
        )
        parser.add_argument(
            "--storage",
            type=str,
            choices=["json", "redis"],
            default=os.getenv("STORAGE_TYPE", "redis"),
            help="Storage type to use (json or redis)",
        )
        parser.add_argument(
            "--neuron.axon_off",
            "--axon_off",
            action="store_true",
            help="Set this flag to not attempt to serve an Axon.",
            default=False,
        )

        BaseRedisStorage.add_args(parser)
        Subtensor.add_args(parser)
        logging.add_args(parser)
        Wallet.add_args(parser)

    def setup_bittensor_objects(self) -> "Subtensor":

        self.wallet = Wallet(config=self.config)

        self.subtensor = Subtensor(config=self.config)

        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        self.metagraph_info = self.subtensor.get_metagraph_info(netuid=self.config.netuid)

        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            logging.error(
                f"\nYour miner: {self.wallet} is not registered to chain connection: {self.subtensor}\n"
                f"Run 'btcli subnet register' and try again."
            )
            exit()

        self._dendrite = Dendrite(wallet=self.wallet)

        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        self.current_block = self.metagraph.block

        if not self.config.neuron.axon_off:
            self.serve_axon()
        else:
            logging.warning("axon off, not serving ip to chain.")


    def create_worker_id(self) -> str:
        hotkey = self.wallet.hotkey.ss58_address
        return hotkey[:4] + hotkey[-4:]

    def blocks_until_next_epoch(self) -> int:
        blocks = self.subtensor.subnet(self.config.netuid).blocks_since_last_step
        return self.tempo - blocks

    def serve_axon(self):

        try:
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