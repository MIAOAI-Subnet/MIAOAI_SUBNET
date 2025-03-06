
<div align="center">

# **MIAO AI** <!-- omit in toc -->

### Bridging Pet Tech and Blockchain Innovation <!-- omit in toc -->


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

#  Introduction

MIAOAI is a cat-themed AI-MEME subnet on the Bittensor, crafted to build a decentralized AI community centered on cat behavior analysis and bionics. Designed for cat lovers, AI enthusiasts, and developers, MIAOAI harnesses cutting-edge AI to explore the intersection of feline communication, AI innovation, and meme culture, creating a unique and engaging experience.

# What is MIAOAI?
MIAOAI is a cat-themed AI-MEME subnet on the Bittensor ecosystem, designed to create an engaging and entertaining community through cat-inspired applications. We aim to build a vibrant ecosystem where users can:
-  Interact with Digital Cats: Generate digital twin cats from your Miao sounds, enabling immersive experiences like AR games, social platform interactions, or virtual pet simulations where your cat can "respond" to commands or mimic real-life behaviors.
-  Enjoy Cat-Themed Entertainment: Create and share cat-inspired audio memes (e.g., turning songs or speech into cat sounds) on platforms like X and Instagram, earning rewards based on social engagement and spreading the fun of Miao culture.

# Core Algorithm
- Audio Feature Extraction: VGGish + STFT - Initially, the VGGish model is used to extract features from audio data.
- Audio Classification: Audio Spectrogram Transformer (AST) - AST is a Transformer model specifically designed for audio data, utilizing self-attention mechanisms to process spectrograms.
- Sound Event Classification: Our method to differentiate “non-dog bark” sounds uses a dual contrastive learning strategy. This involves a “negative sample provisioning” approach where we select and strengthen learning on sounds that could be mistaken for dog barks, thus improving the model’s ability to discriminate between different sounds.

# Model Performance Comparison

<img width="416" alt="image" src="https://github.com/user-attachments/assets/a25d4cc0-bbca-4f74-b587-852a706e800e">

# Miner and Validator Functionality

# Overview
- ⚖️ [Validator](./docs/validator.md)
- ⛏️ [Miner](./docs/miner.md)

This tutorial shows how to  run incentives on it using the our testnet.
**important**.
- Do not expose your private key.
- Use only your testnet wallet.
- Do not reuse your mainnet wallet password.
- Make sure your incentives are resistant to abuse.

## Preparation
#### prepare subnet
```bash
git clone https://github.com/MIAO-AI/MIAOAI_Subnet
python3 -m venv btcli_venv
source btcli_venv/bin/activate

# setuo bittensor sdk
pip install bittensor
pip install -e .
```
##  1.running MIAO-recognition
```bash
 git clone https://github.com/MIAO-AI/MIAO-recognition

 cd MIAO-recognition

 python -m venv venv
 source venv/bin/activate
 pip install -r requirements.txt
 python app.py 
```

### start miner
```bash
python neurons/miner.py --netuid 248 --subtensor.network test --wallet.name miner --wallet.hotkey miner --logging.debug
```

### start validator
```bash
python neurons/validator.py --netuid 248 --subtensor.network test --wallet.name validator1 --wallet.hotkey validator1 --logging.debug 
```
### check state
```bash
btcli wallet overview --wallet.name miner --netuid 248 --subtensor.network test
btcli wallet overview --wallet.name validator --netuid 248 --subtensor.network test
```

# Notice
The model always stays on your machine and is yours!
The data provided by our community friends and the benefits and efficiency brought by running in the subnet will better help us train the dog model

