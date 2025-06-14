# Miaoai Validator Setup

- [miaoai Validator Setup ](#miaoai-validator-setup)
  - [Prerequisites](#prerequisites)
  - [Setup Steps](#setup-steps)
    - [1. Bittensor Wallet Setup](#1-bittensor-wallet-setup)
    - [2. Install Redis](#2-install-redis)
    - [3. Install Miaoai](#3-install-miaoai)
    - [4. Configuration Methods](#4-configuration-methods)
      - [Option A: Using a `.env` File (Recommended)](#option-a-using-a-env-file-recommended)
      - [Option B: Using Command-Line Arguments](#option-b-using-command-line-arguments)
    - [5. Running the Validator](#5-running-the-validator)
      - [Option A: Using PM2 (Recommended)](#option-a-using-pm2-recommended)
      - [Option B: Direct Python Execution](#option-b-direct-python-execution)
  - [Important Parameters](#important-parameters)
  - [Validator evaluation](#validator-evaluation)
  - [Troubleshooting](#troubleshooting)
  - [Security Notes](#security-notes)
  - [PM2 Management Guide](#pm2-management-guide)
    - [Process Management](#process-management)
  - [Support](#support)

This guide will walk you through setting up and running a Miaoai validator on the Bittensor network.

## Prerequisites

1. A Bittensor wallet
2. Python environment (Python 3.10 or higher recommended)

## Setup Steps

### 1. Bittensor Wallet Setup

Ensure you have created a Bittensor wallet. If you haven't, you can create one using:
```bash
pip install bittensor-cli
btcli wallet create
```
## 2. Install Redis

### Ubuntu/Debian
```bash
sudo apt update && sudo apt install redis-server -y
sudo systemctl enable redis-server
sudo systemctl start redis-server
```
### MacOS
```bash
brew install redis
brew services start redis
```
### Verify installation
```bash
redis-cli ping
```
You should receive PONG as a response.

### 3. Install Miaoai

First, let's set up the repository and Python environment:

1. Clone the repository:
```bash
git clone https://github.com/MIAOAI-Subnet/MIAOAI_SUBNET.git
cd MIAOAI_SUBNET
```

2. Set up Python virtual environment:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

3. Install the package:
```bash
pip install -e .
```

### 4. Configuration Methods

You have two ways to configure the validator:

#### Option A: Using a `.env` File (Recommended)

1. Create a `.env` file in the project root based on the provided example:
```bash
cd miaoai/validator
cp .env.validator.example .env
```

2. Edit the `.env` file with your credentials:
```bash
nano .env
cd ../.. # Return to the root dir
```

This approach keeps your credentials secure and simplifies your command-line commands.

#### Option B: Using Command-Line Arguments

Use all required parameters directly in the command line (see "Important Parameters" section below).

### 5. Running the Validator

Now you have two options to run the validator: using PM2 for process management (recommended) or directly with Python.

#### Option A: Using PM2 (Recommended)

1. Install PM2:

   **On Linux**:
   ```bash
   sudo apt update && sudo apt install nodejs npm -y && sudo npm install pm2@latest -g && pm2 update
   ```

   **On macOS**:
   ```bash
   brew update && brew install node && npm install pm2@latest -g && pm2 update
   ```

   Verify installation:
   ```bash
   pm2 --version
   ```

   Setup PM2 startup script:
   ```bash
   pm2 startup
   ```

2. Start the validator with PM2:

   Using `.env` configuration:
   ```bash
   pm2 start python3 --name "miaoai-validator" -- miaoai/validator/miaoai_validator.py run \
       --subtensor.network finney \
       --logging.info
   ```

   Or with command-line arguments:
   ```bash
   pm2 start python3 --name "miaoai-validator" -- miaoai/validator/miaoai_validator.py run \
       --netuid 86 \
       --subtensor.network finney \
       --wallet.name YOUR_WALLET_NAME \
       --wallet.hotkey YOUR_HOTKEY \
       --logging.info
   ```

#### Option B: Direct Python Execution

   Using `.env` configuration:
   ```bash
   python3 miaoai/validator/miaoai_validator.py run \
       --subtensor.network finney \
       --logging.info
   ```

   Or with command-line arguments:
   ```bash
   python3 miaoai/validator/miaoai_validator.py run \
       --netuid 86 \
       --subtensor.network finney \
       --wallet.name YOUR_WALLET_NAME \
       --wallet.hotkey YOUR_HOTKEY \
       --logging.info
   ```

## Important Parameters

- `netuid`: Set to 86 for miaoai subnet
- `subtensor.network`: Set to `finney` for mainnet
- `wallet.name`: Your Bittensor wallet name
- `wallet.hotkey`: Your wallet's hotkey
- `logging.info`: Enables detailed logging

## Validator evaluation

1. Validators submit their evaluations every `blocks_per_weights_set`. Currently set at `2 * tempo`. 
2. They fetch, calculate, and store the model value provided by miners every 5 minutes (25 blocks).
2. After setting weights, moving average is updated and scores are refreshed for next evaluation window. 

## Troubleshooting

If you encounter any issues:
1. Verify all credentials are correct
2. Ensure your wallet is properly set up and registered on the subnet
3. Check logs for detailed error messages

## Security Notes

- Never share your wallet credentials
- Store your wallet password securely

## PM2 Management Guide

### Process Management
```bash
# Save process list (ensures validator restarts on system reboot)
pm2 save

# Basic Commands
pm2 list                    # View all processes
pm2 monit                   # Monitor in real-time
pm2 logs miaoai-validator  # View live logs
pm2 logs miaoai-validator --lines 100  # View last 100 lines of logs
pm2 stop miaoai-validator  # Stop validator
pm2 restart miaoai-validator # Restart validator
pm2 delete miaoai-validator # Remove from PM2

# Log management
# Install log rotation module
pm2 install pm2-logrotate

# Configure log rotation (optional)
pm2 set pm2-logrotate:max_size 10M
pm2 set pm2-logrotate:retain 7
```

## Validator components

### - Bittensor Wallet: 
Manages cryptographic keys and identity for subnet registration and Bittensor operations. 

### - Validator file
The validator serves as the core orchestration component, responsible for assigning AI tasks to miners and evaluating the quality of their task results.

### - Storage
Persistent data layer (Redis) that stores  configurations, schedules, and AI task state between restarts.


## Support

If you need help, you can:
- Join the [Bittensor Discord](https://discord.com/invite/bittensor) and navigate to Subnet 86
- Check the Miaoai documentation
- To run on different environments, check out [Subnet tutorials](https://docs.bittensor.com/tutorials/basic-subnet-tutorials)