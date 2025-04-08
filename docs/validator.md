# Full instructions for setup

Welcome to MIAOAI Validator 🔥

## Setup environment
```bash
git clone https://github.com/MIAO-AI/MIAOAI_Subnet
```

### Install system dependencies
You can directly use our one-click installation script to create the environment
```bash
cd script
bash setup_btcli.sh
```
Or you can install the dependencies manually
```bash
# creat venv 
python3 -m venv btcli_venv
source btcli_venv/bin/activate

# setup bittensor sdk
pip install bittensor
pip install -e .
```

### Get hot and coldkeys onto your machine
Securely move them onto your machine as usual. Either with the btcli or with a secure method of your choosing
```bash
btcli wallet create
```
## Register your validator
```bash
btcli subnet register --netuid 84  --wallet.name validator --wallet.hotkey validator
```

## Start validator
```bash
python neurons/miner.py --netuid 84  --wallet.name miner --wallet.hotkey miner --logging.debug
```

