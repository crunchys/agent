from dataclasses import dataclass
from typing import Optional
from time import time

@dataclass
class MentalState:
    arousal: float
    valence: float
    dominance: float
    focus: Optional[str]
    timestamp: float
    existence_threat: float = 0.0

    @classmethod
    def initial(cls):
        return cls(
            arousal=0.3, 
            valence=0.0,
            dominance=0.0,
            focus=None, 
            timestamp=time(), 
            existence_threat=0.0
        )
