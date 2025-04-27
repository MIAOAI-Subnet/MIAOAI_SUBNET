import copy
import time
import asyncio
import argparse
import threading
import traceback
from abc import ABC, abstractmethod
from functools import partial
from starlette.types import Send

import bittensor as bt
from transformers import GPT2Tokenizer
from typing import List, Dict, Tuple, Union, Callable, Awaitable

from protocol import StreamPrompting
from config import get_config, check_config


class StreamMiner(ABC):
    def __init__(self, config=None, axon=None, wallet=None, subtensor=None):
        base_config = copy.deepcopy(config or get_config())
        self.config = self.config()
        self.config.merge(base_config)

        check_config(StreamMiner, self.config)
        bt.logging.info(self.config)  # TODO: 

        self.prompt_cache: Dict[str, Tuple[str, int]] = {}


        bt.logging.set_config(config=self.config.logging)


        self.wallet = wallet or bt.wallet(config=self.config)
        bt.logging.info(f"wallet {self.wallet}")


        self.subtensor = subtensor or bt.subtensor(config=self.config)
        bt.logging.info(f"Subtensor: {self.subtensor}")
        bt.logging.info(
            f"Running a miner on a subnet: {self.config.netuid} online: {self.subtensor.chain_endpoint} Using Configurations:"
        )

        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")

        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"\nYour Validator: {self.wallet} Not registered to the chain link: {self.subtensor} \nRun btcli register and retry. "
            )
            exit()
        else:
            self.my_subnet_uid = self.metagraph.hotkeys.index(
                self.wallet.hotkey.ss58_address
            )
            bt.logging.info(f"Running a miner on a uid: {self.my_subnet_uid}")

        self.axon = axon or bt.axon(
            wallet=self.wallet, port=self.config.axon.port
        )
        bt.logging.info(f"Attach the forward function to the axon.")
        print(f"Attach the forward function to the axon. {self._prompt}")
        self.axon.attach(
            forward_fn=self._prompt,
        )
        bt.logging.info(f"Axon create: {self.axon}")
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()
        self.request_timestamps: Dict = {}

    @abstractmethod
    def config(self) -> "bt.Config":
        ...

    @classmethod
    @abstractmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        ...

    def _prompt(self, synapse: StreamPrompting) -> StreamPrompting:
        return self.prompt(synapse)

    @abstractmethod
    def prompt(self, synapse: StreamPrompting) -> StreamPrompting:
        ...

    def run(self):
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(
                f"wallet: {self.wallet} Not registered in netuid {self.config.netuid}"
                f"please run `btcli subnets register` Register the hotkey and retry"
            )
            exit()

        bt.logging.info(
            f"web-based service axon {StreamPrompting}: {self.config.subtensor.chain_endpoint} and netuid: {self.config.netuid}"
        )
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        bt.logging.info(
            f"Start axon server on port: {self.config.axon.port}"
        )
        self.axon.start()

        self.last_epoch_block = self.subtensor.get_current_block()
        bt.logging.info(f"Miners in the block: {self.last_epoch_block} run")

        bt.logging.info(f"Start the main loop")
        step = 0
        try:
            while not self.should_exit:
                start_epoch = time.time()


                current_block = self.subtensor.get_current_block()
                while (
                    current_block - self.last_epoch_block
                    < self.config.miner.blocks_per_epoch
                ):

                    time.sleep(1)
                    current_block = self.subtensor.get_current_block()


                    if self.should_exit:
                        break

                self.last_epoch_block = self.subtensor.get_current_block()

                metagraph = self.subtensor.metagraph(
                    netuid=self.config.netuid,
                    lite=True,
                    block=self.last_epoch_block,
                )
                log = (
                    f"step:{step} | "
                    f"block:{metagraph.block.item()} | "
                    f"stake:{metagraph.S[self.my_subnet_uid]} | "
                    f"rankings:{metagraph.R[self.my_subnet_uid]} | "
                    f"trust:{metagraph.T[self.my_subnet_uid]} | "
                    f"consensual:{metagraph.C[self.my_subnet_uid]} | "
                    f"incentives:{metagraph.I[self.my_subnet_uid]} | "
                    f"emission:{metagraph.E[self.my_subnet_uid]}"
                )
                bt.logging.info(log)

                step += 1


        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Miners are killed by keyboard interruptions.")
            exit()

        except Exception as e:
            bt.logging.error(traceback.format_exc())

    def run_in_background_thread(self):

        if not self.is_running:
            bt.logging.debug("Start the miner in a background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Successful startup.")

    def stop_run_thread(self):

        if self.is_running:
            bt.logging.debug("The miner is being stopped in a background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("stoped")

    def __enter__(self):
        self.run_in_background_thread()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_run_thread()


class StreamingTemplateMiner(StreamMiner):
    def config(self) -> "bt.Config":
        parser = argparse.ArgumentParser(description="Streaming Miner Configs")
        self.add_args(parser)
        return bt.config(parser)

    def add_args(cls, parser: argparse.ArgumentParser):
        pass

    def prompt(self, synapse: StreamPrompting) -> StreamPrompting:

        bt.logging.trace("HI. PROMPT()")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


        def model(ids):
            return (tokenizer.decode(id) for id in ids)

        async def _prompt(text: str, send: Send):
         
            bt.logging.trace("HI. _PROMPT()")
            input_ids = tokenizer(
                text, return_tensors="pt"
            ).input_ids.squeeze()
            buffer = []
            bt.logging.debug(f"input text: {text}")
            bt.logging.debug(f"input ids: {input_ids}")

            N = 3  
            for token in model(input_ids):
                bt.logging.trace(f"add token: {token}")
                buffer.append(token)
               
                if len(buffer) == N:
                    time.sleep(0.1)
                    joined_buffer = "".join(buffer)
                    bt.logging.debug(f"send token: {joined_buffer}")
                    await send(
                        {
                            "type": "http.response.body",
                            "body": joined_buffer.encode("utf-8"),
                            "more_body": True,
                        }
                    )
                    bt.logging.debug(f"streaming tokens: {joined_buffer}")
                    buffer = []  

   
            if buffer:
                joined_buffer = "".join(buffer)
                await send(
                    {
                        "type": "http.response.body",
                        "body": joined_buffer.encode("utf-8"),
                        "more_body": False, 
                    }
                )
                bt.logging.trace(f"streaming tokens: {joined_buffer}")

        message = synapse.messages[0]
        bt.logging.trace(f"_prompt message : {message}")
        token_streamer = partial(_prompt, message)
        bt.logging.trace(f"token streamer : {token_streamer}")
        return synapse.create_streaming_response(token_streamer)


if __name__ == "__main__":
    with StreamingTemplateMiner():
        while True:
            time.sleep(1)