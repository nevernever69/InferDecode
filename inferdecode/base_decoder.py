from abc import ABC, abstractmethod

class BaseDecoder(ABC):
    @abstractmethod
    async def generate_full_trace(self, prompt: str, max_steps: int, temperature: float, top_p: float, top_k: int, decoding_strategy: str):
        """Generate a trace of decoding steps."""
        pass

