# Running a subnetwork on the main network

This tutorial shows how to create a subnetwork using Bittensor's `btcli` and connect your incentives to it.

**Important note:** Before attempting to register on the main network, we strongly recommend that you:
- First run [run subnet locally](running_on_staging.md), and
- then run [running on testnet](running_on_testnet.md).

The incentives to run on the main network are open to anyone. They issue real TAOs. creating these mechanisms generates `lock_cost` TAOs.

**DANGER
- Do not expose your private key.
- Use only your testnet wallet.
- Do not reuse the password for your main net wallet.
- Make sure your incentives are resistant to abuse.

## Pre-requisites

Before proceeding, make sure you have Bittensor installed. see the following instructions:

- [Install `bittensor`](https://github.com/opentensor/bittensor#install).

After installing Bittensor, follow these steps:

## Steps

## 1. Install your subnet template

**Note: Skip this step if you have already done it during local testing and development. **

In the project directory:

```bash
git clone https://github.com/Woof-AI/WOOFAI_Subnet.git
```

Next, ``cd`'' into the ``WOOFAI_Subnet`' repository directory:

``bash
cd WOOFAI_Subnet
```

Install the WOOFAI_Subnet package:

```bash
python -m pip install -e .
```

## 2. Create wallets

Create wallets for subnet owners, subnet verifiers, and subnet miners.

This step creates local cold and hot key pairs for your three identities: subnet owner, subnet verifier, and subnet miner.

The owner will create and control the subnet. The owner must have at least 100 TAO to run the next steps.

Validators and miners will register to the subnet created by the owner. This ensures that validators and miners can run their respective validator and miner scripts.

**Note**: You can also use an existing wallet for registration. Creating a new key is shown here for reference only.

Create a cold key for the owner wallet:

```bash
btcli wallet new_coldkey --wallet.name owner
```

Create cold and hot keys for the subnet miner wallet:
```bash
btcli wallet new_coldkey --wallet.name miner
```

and

```bash
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default
```

Create cold and hot keys for the subnet authenticator wallet:

```bash
btcli wallet new_coldkey --wallet.name validator
```

and

```bash
btcli wallet new_hotkey --wallet.name validator --wallet.hotkey default
```

## 3. Getting prices for subnet creation

Creating subnets on the mainnet is competitive. The cost is determined by the rate at which new subnets are registered on the Bittensor blockchain.

By default, there must be at least 100 TAO in the owner's wallet to create a subnet. However, the exact amount fluctuates based on demand. The following code shows how to get the current price for creating a subnet.

```bash
btcli subnet lock_cost 
```

The above command will show:

```bash
>> subnet lock cost: τ100.000000000
```

## 4. purchasing a slot

Using your TAO balance, you can register your subnet to the primary chain. This will create a new subnet and give you owner rights. The following commands show how to purchase a slot.

**Note**: Slots require a locked TAO as a fee. You will be refunded this TAO when you log out of the subnet.

```bash
btcli subnet create  
```

Enter the owner wallet name. This will grant cold key permissions.

```bash
>> Enter wallet name (default): owner # Enter your owner wallet name
>> Enter password to unlock key: # Enter your wallet password.
>> Register subnet? [y/n]: <y/n> # Select yes (y)
>> ⠇ 📡 Registering subnet...
✅ Subnet registered, netuid: 1 # Your subnet netuid will be displayed here, note down this information for future reference.
```

## 5. (Optional) Registration Key

**Note**: While this is not mandatory, we recommend that subnet owners run a subnet validator and subnet miner on their subnets to demonstrate proper usage to the community.

This step registers your Subnet Verifier and Subnet Miner keys to the subnet, giving them access to the **First two slots**.

Register your miner key to the subnet:

```bash
btcli subnet recycle_register --netuid 1 --subtensor.network finney --wallet.name miner --wallet.hotkey default
```

Follow the prompts below:

```bash
>> Enter netuid [1] (1): # Enter netuid 1 to specify the subnet you just created.
>> Continue to register ?
  hotkey: ...
  coldkey: ...
  network: finney [y/n]: # Select yes (y)
>> ✅ Registered
```bash

btcli wallet new_coldkey --wallet.name miner

```
and
```bash

btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default

```
Create cold and hot keys for the subnet authenticator wallet:
```bash
  hotkey: ...
  coldkey: ...
  network: finney [y/n]: # Select yes (y)
>> ✅ Registered
```
## 6. Check if your key is registered

Check if your subnet authenticator key is registered:

```bash
btcli wallet overview --wallet.name validator 
```

The output will be similar to the following:

``bash
Subnet: 1                                                                                                                                                                
COLDKEY HOTKEY UID ACTIVE STAKE(τ) RANK TRUST CONSENSUS INCENTIVE DIVIDENDS EMISSION(ρ) VTRUST VPERMIT UPDATED AXON HOTKEY_SS58                    
miner default 0 True 0.00000 0.00000 0.00000 0.00000 0.00000 0.00000 0.00000 0 0.00000 14 none 5GTFrsEQfvTsh3WjiEVFeKzFTc2xcf...
1 1 2 τ0.00000 0.00000 0.00000 0.00000 0.00000 0.00000 0.00000 ρ0 0.00000                                                         
                                                                          Wallet balance: τ0.0         
```

Check if your subnet miner is registered:

```bash
btcli wallet overview --wallet.name miner 
```

The output will be similar to the following:

```bash
Subnet: 1                                                                                                                                                                
COLDKEY HOTKEY UID ACTIVE STAKE(τ) RANK TRUST CONSENSUS INCENTIVE DIVIDENDS EMISSION(ρ) VTRUST VPERMIT UPDATED AXON HOTKEY_SS58                    
miner default 1 True 0.00000 0.00000 0.00000 0.00000 0.00000 0.00000 0.00000 0 0.00000 14 none 5GTFrsEQfvTsh3WjiEVFeKzFTc2xcf...
1 1 2 τ0.00000 0.00000 0.00000 0.00000 0.00000 0.00000 0.00000 ρ0 0.00000                                                         
                                                                          Wallet balance: τ0.0   
```

## 7. running subnet miners and subnet validators

Run the subnet miner:

```bash
python neurons/miner.py --netuid 1 --wallet.name miner --wallet.hotkey default --logging.debug
```

You will see the following terminal output:

```bash
>> 2023-08-08 16:58:11.223 | INFO | Being subnetted for: 1 On network: wss://entrypoint-finney.opentensor.ai:443 Running the miner, configuring: ...
```

Run subnet verifier:

```bash
python neurons/validator.py --netuid 1 --wallet.name validator --wallet.hotkey default --logging.debug
```

You will see the following terminal output:

```bash
>> 2023-08-08 16:58:11.223 | INFO | Being subnetted for: 1 On network: wss://entrypoint-finney.opentensor.ai:443 Running the validator with configuration: ...
```

## 8. Making emissions flow

Use ``btcli`` to register to the root subnet:

```bash
btcli root register 
```

Then set the weights for the subnet:

``bash
btcli root weights 
```

## 9. Stopping your node

To stop your node, press CTRL + C in the terminal running the node.

---
