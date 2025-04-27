import argparse
import asyncio
import bittensor as bt

from protocol import StreamPrompting
async def query_synapse(my_uid, wallet_name, hotkey, network, netuid):
    syn = StreamPrompting(
        roles=["user"],
        messages=[
            "Hi, this is the test for convective response. lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
        ],
    )

    wallet = bt.wallet(name=wallet_name, hotkey=hotkey)


    metagraph = bt.metagraph(
        netuid=netuid, network=network, sync=True, lite=False
    )


    axon = metagraph.axons[my_uid]


    dendrite = bt.dendrite(wallet=wallet)

    async def main():
        responses = await dendrite(
            [axon], syn, deserialize=False, streaming=True
        )

        for resp in responses:
            i = 0
            async for chunk in resp:
                i += 1
                if i % 5 == 0:
                    print()
                if isinstance(chunk, list):
                    print(chunk[0], end="", flush=True)
                else:
               
                    synapse = chunk
            break


    await main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bittensor synapses were queried using the given parameters."
    )


    parser.add_argument(
        "--my_uid",
        type=int,
        required=True,
        help="Your unique miner ID on the chain",
    )
    parser.add_argument(
        "--netuid", type=int, required=True, help="network only ID"
    )
    parser.add_argument(
        "--wallet_name", type=str, default="default", help="Wallet name"
    )
    parser.add_argument(
        "--hotkey", type=str, default="default", help="Hot keys for wallets"
    )
    parser.add_argument(
        "--network",
        type=str,
        default="test",
        help='Network type，Example "test" or "mainnet"',
    )


    args = parser.parse_args()


    asyncio.run(
        query_synapse(
            args.my_uid,
            args.wallet_name,
            args.hotkey,
            args.network,
            args.netuid,
        )
    )