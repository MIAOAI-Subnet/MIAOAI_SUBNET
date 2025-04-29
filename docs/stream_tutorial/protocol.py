import pydantic
import bittensor as bt

from abc import ABC, abstractmethod
from typing import List, Union, Callable, Awaitable
from starlette.responses import StreamingResponse


class StreamPrompting(bt.StreamingSynapse):
 
    roles: List[str] = pydantic.Field(
        ...,
        title="role",
        description="A list of roles in the StreamPrompting scene. Immutable.",
        allow_mutation=False,
    )

    messages: List[str] = pydantic.Field(
        ...,
        title="messages",
        description="A list of messages in a StreamPrompting scenario. Immutable.",
        allow_mutation=False,
    )

    required_hash_fields: List[str] = pydantic.Field(
        ["messages"],
        title="Required Hash Fields",
        description="A list of fields required for the hash.",
        allow_mutation=False,
    )

    completion: str = pydantic.Field(
        "",
        title="finished",
        description="The completion status of the current StreamPrompting object. This property is mutable and can be updated.",
    )

    async def process_streaming_response(self, response: StreamingResponse):
      
        if self.completion is None:
            self.completion = ""
        bt.logging.debug(
            "Handles streaming responses (StreamingSynapse base class)."
        )
        async for chunk in response.content.iter_any():
            bt.logging.debug(f"process block: {chunk}")
            tokens = chunk.decode("utf-8").split("\n")
            for token in tokens:
                bt.logging.debug(f"--Processing of markers: {token}")
                if token:
                    self.completion += token
            bt.logging.debug(f"pass markers {tokens}")
            yield tokens

    def deserialize(self) -> str:
     
        return self.completion

    def extract_response_json(self, response: StreamingResponse) -> dict:
    
        headers = {
            k.decode("utf-8"): v.decode("utf-8")
            for k, v in response.__dict__["_raw_headers"]
        }

        def extract_info(prefix):
            return {
                key.split("_")[-1]: value
                for key, value in headers.items()
                if key.startswith(prefix)
            }

        return {
            "name": headers.get("name", ""),
            "timeout": float(headers.get("timeout", 0)),
            "total_size": int(headers.get("total_size", 0)),
            "header_size": int(headers.get("header_size", 0)),
            "dendrite": extract_info("bt_header_dendrite"),
            "axon": extract_info("bt_header_axon"),
            "roles": self.roles,
            "messages": self.messages,
            "completion": self.completion,
        }