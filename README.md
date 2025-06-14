<div align="center">

# **MIAOAI** ![Subnet 86](https://img.shields.io/badge/Subnet-86_%E1%9A%B3-red)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/MIAOAI-Subnet/MIAOAI_SUBNET)

</div>

![logo](./assets/logo.png)

MIAOAI is a high-performance AI subnet built on the Bittensor network, designed for decentralized processing and validation of AI tasks. It features a dual-layer architecture of validators and miners, leveraging intelligent task scheduling and dynamic scoring to form an efficient and trustworthy distributed AI inference network.

In MIAOAI:

Validators are responsible for receiving and dispatching diverse AI tasks—such as e-commerce customer service, text classification, and scene understanding—and for evaluating miner performance in real time. Based on performance, validators dynamically adjust miner weights and rewards.

Miners deploy and run AI models to process tasks assigned by validators. They earn token-based incentives according to the quality and correctness of their outputs.

A hybrid scoring mechanism, combining trust-based evaluation with stake-weighted distribution, ensures secure and accurate task allocation across the network.

By establishing a closed loop for task handling and model evaluation, MIAOAI significantly improves the scalability and reliability of decentralized AI systems, injecting robust computational power into the Bittensor subnet ecosystem.
- [Incentive Design](#incentive-design)
- [Requirements](#requirements)
  - [Miner Requirements](#miner-requirements)
  - [Validator Requirements](#validator-requirements)
- [Installation](#installation)
  - [Common Setup](#common-setup)
  - [Miner Specific Setup](#miner-specific-setup)
  - [Validator Specific Setup](#validator-specific-setup)
- [Get Involved](#get-involved)
---

# Incentive Design
MIAOAI’s incentive mechanism is designed to foster positive collaboration between miners and validators through task-driven performance, building a stable and efficient decentralized AI inference network.

Task-driven reward distribution: Validators assign real AI inference tasks (such as intelligent Q&A, image recognition, etc.) and evaluate the results submitted by miners based on accuracy and response time to generate a task quality score.

Dynamic weight and reward calculation: A miner’s task score affects their weight within each validator pool. The system dynamically calculates reward allocation based on multiple factors, including task performance, stake amount, and historical reputation.

Dual scoring with trust and stake: MIAOAI combines a miner’s historical stability (trust score) and the amount of staked tokens (stake weight) to determine task assignment priority and final reward. This dual mechanism helps prevent manipulation by malicious nodes.

# Requirements

## Miner Requirements
To run a MIAOAI miner, you will need:
- A Bittensor wallet
- Bittensor mining hardware ( GPUs, etc.) 
- A running Redis server for data persistence
- Python 3.10 or higher

## Validator Requirements
To run a MIAOAI validator, you will need:
- A Bittensor wallet
- A running Redis server for data persistence
- Python 3.10 or higher environment

# Installation

## Common Setup
These steps apply to both miners and validators:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/MIAOAI-Subnet/MIAOAI_SUBNET.git](https://github.com/MIAOAI-Subnet/MIAOAI_SUBNET.git)
    cd MIAOAI_SUBNET
    ```

2.  **Set up and activate a Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Upgrade pip:**
    ```bash
    pip install --upgrade pip
    ```

4.  **Install the MIAOAI package:**
    ```bash
    pip install -e .
    ```

## Miner Specific Setup
After completing the common setup, follow the detailed steps in the Miner Guide:

* [Install Redis](docs/running_miner#2-install-redis)
* [Configure your miner (`.env` file or command-line arguments)](docs/running_miner#5-configuration)
* [Run the miner (using PM2 recommended)](docs/running_miner#6-running-the-miner)

For the complete, step-by-step instructions for setting up and running your miner, please refer to the [MIAOAI Miner Setup Guide](docs/running_miner).

## Validator Specific Setup
After completing the common setup, follow the detailed steps in the Validator Guide:

* [Configure your validator (`.env` file or command-line arguments)](docs/running_validator#4-configuration-methods)
* [Run the validator (using PM2 recommended)](docs/running_validator#5-running-the-validator)

For the complete, step-by-step instructions for setting up and running your validator, please refer to the [MIAOAI Validator Setup](docs/running_validator).

# Get Involved

- Join the discussion on the [Bittensor Discord](https://discord.com/invite/bittensor) in the Subnet 86 channels.
- Check out the [Bittensor Documentation](https://docs.bittensor.com/) for general information about running subnets and nodes.
- Contributions are welcome! See the repository's contribution guidelines for details.

---
**Full Guides:**
- [MIAOAI Miner Setup Guide ](docs/running_miner.md)
- [MIAOAI Validator Setup ](docs/running_validator.md) 
