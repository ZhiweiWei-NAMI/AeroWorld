"""Selector backbone configuration stored in selector checkpoints."""
from dataclasses import dataclass

@dataclass
class ScoringConfig:
    chunk_tokens: int = 8192       # E union D, excluding task prefix and selected directory
    lora_rank: int = 16
    head_layers: int = 2
    seed: int = 0
    revisit_tokens: int = 2048
    directory_tokens: int = 1024
    temperature: float = 1.0
    records_per_head: int = 1
    revisit_visual_tokens: int = 0
    gate_coupled: bool = True     # reference: candidate utilities also influence KEEP/EDIT

    def __post_init__(self):
        if self.temperature <= 0 or not 0 < self.revisit_tokens < self.chunk_tokens or self.directory_tokens < 32:
            raise ValueError("require 0 < revisit_tokens < chunk_tokens, directory_tokens >= 32 and positive temperature")
