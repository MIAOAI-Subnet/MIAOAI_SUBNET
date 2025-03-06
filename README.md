
<div align="center">

# **WOOF AI** <!-- omit in toc -->

### Bridging Pet Tech and Blockchain Innovation <!-- omit in toc -->
![hero](./asset/offline.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

#  Introduction

“WOOF” is the expression of a dog’s bark. We are developing an AI project designed to analyze and mimic canine behavior. By collecting dog barks uploaded by users, we aim to build a large-scale model to analyze dogs emotions and predict their behaviors, fostering a closer emotional bond between humans and dogs. We hope to encourage people around the world to form deeper connections with dogs. In the future, we look forward to expanding Woof AI's capabilities to provide more personalized interactions, enhance pet care, and revolutionize how we communicate with our furry companions. Woof woof!

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
git clone https://github.com/Woof-AI/WOOFAI_Subnet
python3 -m venv btcli_venv
source btcli_venv/bin/activate

# setuo bittensor sdk
pip install bittensor
pip install -e .
```
##  1.running WOOF-recognition
```bash
 git clone https://github.com/Woof-AI/WOOF-recognition

 cd WOOF-recognition

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

