from fastapi import FastAPI
import bittensor as bt
import uvicorn
from typing import Optional
from pydantic import BaseModel  

app = FastAPI()


class AudioRequest(BaseModel):  
    audio_data: str


class DogSoundProtocol(bt.Synapse):
    audio_data: str
    is_dog_sound: Optional[bool] = None
    probability: Optional[float] = None
    response_time: Optional[float] = None

    def deserialize(self) -> dict:
        return {
            'is_dog_sound': self.is_dog_sound,
            'probability': self.probability,
            'response_time': self.response_time
        }


wallet = bt.wallet(
    name='fish',
    hotkey='fish',
    path='/ubuntu/.bittensor/wallets'
)
subnet = bt.metagraph(netuid=248)
dendrite = bt.dendrite(wallet=wallet)

@app.post("/recognize")
async def recognize_dog_sound(request: AudioRequest):  
    
    responses = await dendrite(
        axons=subnet.axons,
        synapse=DogSoundProtocol(audio_data=request.audio_data),
        timeout=12
    )

    return responses[0].deserialize() if responses else {"error": "No response"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)