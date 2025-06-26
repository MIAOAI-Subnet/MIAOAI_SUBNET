<div align="center">

# **MIAOAI** ![Subnet 86](https://img.shields.io/badge/Subnet-86_%E1%9A%B3-red)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/MIAOAI-Subnet/MIAOAI_SUBNET)

</div>

# MiaoAI - A Decentralized AI Training Subnet for E-commerce

![MIAOAI Logo](./assets/logo.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MiaoAI is undergoing a major upgrade, fully embracing Bittensor's latest Yuma3 consensus mechanism and strategically transitioning into a foundational training network for AI Agents focused on the e-commerce customer service sector. We are dedicated to leveraging the collective power of a distributed network to provide an efficient, multilingual, scalable, and censorship-resistant AI training platform for the global e-commerce ecosystem.

## Core Upgrades in This Release

This upgrade is deeply optimized around Bittensor's Yuma3 consensus, aiming to build a smarter and more efficient ecosystem for incentives and task allocation.

*   Full Compatibility with Yuma3 Consensus: The subnet has been upgraded to fully support all new features of Yuma3. This lays a solid foundation for us to implement more granular task allocation and dynamic reward mechanisms.

*   Optimized Task Allocation and Incentives: We are enhancing the task allocation and reward models for miners. We encourage validators (clients) to open up their task interfaces to distribute more detailed and valuable training tasks to miners, thereby guiding the entire network to produce higher-quality models. Based on the Yuma3 framework, validators who can effectively distribute high-quality training tasks will also receive preferential rewards.

*   Transition to an E-commerce Customer Service AI Training Network: To enhance the commercial value of the subnet's training, we are positioning it as a foundational training network for e-commerce customer service AI. We provide all validators and miners with a large-scale customer service dialogue dataset from the Alibaba e-commerce platform as a foundational training set to help network participants get started and conduct efficient training.

*   Multilingual Environment Support: The global nature of e-commerce requires AI to have multilingual capabilities. Therefore, this training phase will involve multiple languages. We strongly recommend that network participants properly configure their operating system's multilingual environment when deploying their nodes to ensure the smooth execution of deployment and tasks.

*   Support for New Validators: We welcome new validators to join our ecosystem. If you need technical advice on node configuration, especially regarding port settings, please feel free to contact us through our official channels.

## What is MiaoAI?

MiaoAI is a decentralized AI subnet based on Bittensor, dedicated to training efficient, multilingual customer service AI Agents for the e-commerce domain. We move away from the traditional centralized AI service model by distributing training tasks across a global network of miners to collectively build a powerful foundational AI model for e-commerce. This distributed approach not only ensures high availability and censorship resistance but also continuously drives innovation and efficiency for AI models in e-commerce customer service scenarios through market-based competition and collaboration.

## Key Features


| Feature | Description |
| :--- | :--- |
| E-commerce Specialization | Focuses on e-commerce customer service scenarios, utilizing real-world data for training to make the AI model more practical and commercially valuable. |
| Driven by Yuma3 Consensus | Fully supports the latest Yuma3 consensus to achieve a fairer and smarter dynamic reward and task allocation mechanism. |
| Foundational Training Dataset Provided | Provides a large, high-quality customer service dataset from the Alibaba platform as a starter set, lowering the entry barrier for participants. |
| Multilingual Training Support | The network is designed to support the training of multilingual models to meet the demands of global e-commerce. The system language environment must be configured during deployment. |
| Decentralization & Incentives | Inherits the core advantages of Bittensor, ensuring network stability and censorship resistance through decentralization, and attracting global computing power via an incentive mechanism. |

## Future Outlook: Lucky Cat AI Agent

We are excited to announce that the training outcomes of the MiaoAI subnet will power our upcoming next-generation product: the 'Lucky Cat' AI Agent.

'Lucky Cat' (inspired by the Japanese 'Maneki-neko', symbolizing wealth and good fortune) will be a personalized AI customer service solution designed for e-commerce sellers, which is easy to deploy and configure locally. It can be rapidly configured by loading a local knowledge base (e.g., product information, promotional activities, FAQs), becoming an intelligent customer service expert that truly understands your business.

Stay tuned


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

* [Install Redis](docs/running_miner.md#2-install-redis)
* [Configure your miner (`.env` file or command-line arguments)](docs/running_miner.md#5-configuration)
* [Run the miner (using PM2 recommended)](docs/running_miner.md#6-running-the-miner)

For the complete, step-by-step instructions for setting up and running your miner, please refer to the [MIAOAI Miner Setup Guide](docs/running_miner.md).

## Validator Specific Setup
After completing the common setup, follow the detailed steps in the Validator Guide:

* [Configure your validator (`.env` file or command-line arguments)](docs/running_validator.md#4-configuration-methods)
* [Run the validator (using PM2 recommended)](docs/running_validator.md#5-running-the-validator)

For the complete, step-by-step instructions for setting up and running your validator, please refer to the [MIAOAI Validator Setup](docs/running_validator.md).

# Get Involved

- Join the discussion on the [Bittensor Discord](https://discord.com/invite/bittensor) in the Subnet 86 channels.
- Check out the [Bittensor Documentation](https://docs.bittensor.com/) for general information about running subnets and nodes.
- Contributions are welcome! See the repository's contribution guidelines for details.

---
**Full Guides:**
- [MIAOAI Miner Setup Guide ](docs/running_miner.md)
- [MIAOAI Validator Setup ](docs/running_validator.md)