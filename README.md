<div align="center">

# **MIAO AI** <!-- omit in toc -->
https://github.com/MIAOAI-Subnet/MIAOAI_SUBNET
![hero](./asset/offline.jpg)
### Bridging Pet Tech and Blockchain Innovation <!-- omit in toc -->


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

#  Introduction

MIAOAI is a cat-themed AI-MEME subnet dedicated to creating and interacting with cat-inspired audio content, fostering a playful and entertaining decentralized community for cat lovers and meme enthusiasts.

# Overview
This repository powers MIAOAI's ability to generate and manipulate cat vocalizations (e.g., meows, purrs, hisses) for interactive and entertainment purposes. Integrated into the MIAOAI DApp, it allows users to create cat sounds, interact with virtual cat agents, and share meme-worthy audio content, all while earning $MIAO tokens as rewards.

# Key Features

- Cat Sound Generation: Synthesize unique cat vocalizations based on user inputs or emotional prompts (e.g., "playful meow," "curious purr"), creating personalized cat audio clips.
-	 Interactive Virtual Cats: Generate digital cat agents that "speak" using synthesized sounds, enabling fun interactions in AR environments, social platforms, or virtual pet games.
-	 Meme Audio Creation: Transform user-uploaded audio (e.g., songs, speech) into cat-themed versions using style transfer, perfect for sharing on platforms like X and Instagram.
-	 Community Rewards: Earn 10-50 $MIAO tokens per generated audio clip or interaction, with additional bonuses for social media engagement (e.g., likes, shares).

# Model Performance Comparison

MIAOAI's audio generation and interaction system is built on advanced audio synthesis and processing techniques:
-	WaveNet: A generative model for producing realistic cat vocalizations, trained on a diverse dataset of cat sounds to capture nuances like pitch and tone.
-	Audio Style Transfer with GANs: Uses Generative Adversarial Networks to transform user audio into cat-like sounds, preserving the rhythm while infusing feline characteristics.
-	Emotion-Driven Synthesis: Incorporates a lightweight emotion encoder to adjust generated sounds based on user-selected moods (e.g., happy, curious, sleepy).

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
python neurons/miner.py --netuid 86  --wallet.name miner --wallet.hotkey miner --logging.debug
```

### start validator
```bash
python neurons/validator.py --netuid 86  --wallet.name validator1 --wallet.hotkey validator1 --logging.debug 
```
### check state
```bash
btcli wallet overview --wallet.name miner --netuid 86
btcli wallet overview --wallet.name validator --netuid 86 
```

# Notice
The model always stays on your machine and is yours!
The data provided by our community friends and the benefits and efficiency brought by running in the subnet will better help us train the cat model

