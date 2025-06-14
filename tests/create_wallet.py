import bittensor
from getpass import getpass
from bittensor import Keyfile

wallet_name = "owner86"
wallet_path = "~/.bittensor/wallets"

mnemonic = getpass("请输入助记词（输入时不会显示）：\n")

wallet = bittensor.wallet(name=wallet_name, hotkey="default", path=wallet_path)

coldkey = bittensor.Keypair.create_from_mnemonic(mnemonic)

wallet.set_coldkey(coldkey, encrypt=False)

wallet.set_coldkeypub(coldkey)

print(f"✅ 冷钱包 {wallet_name} 已成功从助记词恢复并保存！")
