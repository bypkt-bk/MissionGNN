import torch
from pathlib import Path

def load_cached_class_graph(class_name: str, cache_dir: str = "cached_graphs") -> tuple[torch.Tensor, torch.Tensor]:
    graph_path = Path(cache_dir) / f"{class_name}_graph.pt"
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph not found: {graph_path}")
    data = torch.load(graph_path, map_location="cpu", weights_only=False)
    return data["vertex_embeds"], data["edge_index"]

