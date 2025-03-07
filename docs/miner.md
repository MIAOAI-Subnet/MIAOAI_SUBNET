# Full instructions for setup

Welcome to MIAOAI Mining 🔥

# Overview
A miner consists of several parts, fitting into two categories:

- Proxy
- Workers

Because of the existence of JIN, you don’t need to pay attention to the proxy, you only need to install the environment to start mining! ! !

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
## Register your miner
```bash
btcli subnet register --netuid 319 --subtensor.network test --wallet.name miner --wallet.hotkey miner
```
## Start up your model
```bash
git clone https://github.com/MIAO-AI/MIAO-recognition
cd MIAO-recognition
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py 
```

## Export your model service url
```bash
export AI_API_URL="your_model_url"
```
## Start miners
```bash
python neurons/miner.py --netuid 319 --subtensor.network test --wallet.name miner --wallet.hotkey miner --logging.debug
```

