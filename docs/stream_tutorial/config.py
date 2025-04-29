import bittensor as bt
import argparse
import os


def check_config(cls, config: "bt.Config"):
    bt.axon.check_config(config)
    bt.logging.check_config(config)
    full_path = os.path.expanduser(
        "{}/{}/{}/{}".format(
            config.logging.logging_dir,
            config.wallet.get("name", bt.defaults.wallet.name),
            config.wallet.get("hotkey", bt.defaults.wallet.hotkey),
            config.miner.name,
        )
    )
    config.miner.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.miner.full_path):
        os.makedirs(config.miner.full_path)


def get_config() -> "bt.Config":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--axon.port", type=int, default=8098, help="Ports running Axon."
    )

    parser.add_argument(
        "--subtensor.network",
        default="finney",
        help="Bittensor network to be connected.",
    )

    parser.add_argument(
        "--subtensor.chain_endpoint",
        default="wss://entrypoint-finney.opentensor.ai:443",
        help="The endpoint of the chain to be connected.",
    )

    parser.add_argument(
        "--netuid", type=int, default=1, help="Chain.com uid."
    )

    parser.add_argument(
        "--miner.root",
        type=str,
        help="The trials for this miner will be in miner.root / (wallet_cold - wallet_hot) / miner.name ",
        default="~/.bittensor/miners/",
    )
    parser.add_argument(
        "--miner.name",
        type=str,
        help="The trials for this miner will be in miner.root / (wallet_cold - wallet_hot) / miner.name ",
        default="Bittensor miner",
    )

    parser.add_argument(
        "--miner.blocks_per_epoch",
        type=str,
        help="Block until the miner re-pulls the meta-map from the chain",
        default=100,
    )

    parser.add_argument(
        "--miner.no_serve",
        action="store_true",
        help="If true, the miner does not provide Axon services.",
        default=False,
    )
    parser.add_argument(
        "--miner.no_start_axon",
        action="store_true",
        help="If true, the miner does not start Axon.",
        default=False,
    )

    parser.add_argument(
        "--miner.mock_subtensor",
        action="store_true",
        help="If true, the miner will allow non-registered hotkeys to be mined.",
        default=False,
    )

    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.axon.add_args(parser)
    config = bt.config(parser)

    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            "miner",
        )
    )
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)
    return config