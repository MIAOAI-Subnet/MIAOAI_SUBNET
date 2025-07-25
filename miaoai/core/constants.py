from pathlib import Path
import os

BLOCK_TIME = 12

MAIN_PATH = Path(__file__).parent.parent.parent

VERSION_KEY = 2208

U16_MAX = 65535

DEFAULT_ALLOCATION_STRATEGY = "equal"  # Options: "stake", "equal"

DEFAULT_MODEL_NAME = "Qwen/Qwen3-32B"

DEFAULT_BLACKLIST = []

BAD_COLDKEYS = []

RAO_TO_TAO = 1_000_000_000

OWNER_DEFAULT_SCORE = 0.2
FINAL_MIN_SCORE = 0.8
DEFAULT_PENALTY_COEFFICIENT= 0.000000001

MIN_VALIDATOR_STAKE_DTAO = float(os.getenv("MIN_VALIDATOR_STAKE_DTAO", "1000.0"))
MIN_MINER_STAKE_DTAO = float(os.getenv("MIN_MINER_STAKE_DTAO", "50.0"))

TESTNET_NETUID = 356

MIN_BLOCKS_PER_VALIDATOR = 10

MAX_TASK_POOL_SIZE = 1000
DEFAULT_TASK_POOL_SIZE = 1000

DEFAULT_LOG_PATH = "logs"
MAX_VALIDATOR_BLOCKS = int(os.getenv("MAX_VALIDATOR_BLOCKS", "7200"))
CHECK_NODE_ACTIVE = os.getenv("CHECK_NODE_ACTIVE", "false").lower() == "true"
