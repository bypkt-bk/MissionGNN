from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch

@dataclass
class Config:
    # Data/paths
    data_root: Path = Path("./embeddings")
    subgraph_dir: Path = Path("./subgraphs")
    checkpoint_dir: Path = Path("./checkpoints")

    # Classes
    classes: List[str] = field(default_factory=lambda: [
        "Abuse","Arrest","Arson","Assault","Burglary","Explosion",
        "Fighting","RoadAccidents","Robbery","Shooting","Shoplifting",
        "Stealing","Vandalism",
    ])

    # Model sizes
    embed_dim: int = 1024
    gnn_hidden: int = 64
    transformer_d_model: int = 13 * 64  # num_classes * gnn_hidden
    transformer_ff: int = 256
    transformer_heads: int = 8

    # Optimisation
    lr: float = 1e-4
    weight_decay: float = 0.1
    lambda_1: float = 1e-3               # regular anomaly encouragement
    # -------- decaying threshold --------
    threshold_start: float = 1.0         # τ₀ in reference code
    threshold_decay: float = 0.9999      # γ   in reference code  (τ ← τ·γ per batch)

    epochs: int = 10
    batch_train: int = 32
    batch_val: int = 32

    # Others
    seed: int = 42
    device: str = "mps" if torch.backends.mps.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu")

cfg = Config()
