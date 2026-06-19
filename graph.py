import sys
import torchvision.transforms.functional as TF

sys.modules["torchvision.transforms.functional_tensor"] = TF
from pathlib import Path
from config import cfg
from graph.kg_loader import load_class_graph

import torch
import torch.nn.functional as F

def save_all_graphs():
    output_dir = Path("cached_graphs")
    output_dir.mkdir(parents=True, exist_ok=True)

    for class_name in cfg.classes:
        print(f"processing: {class_name}...")
        
        vertex_embeds, edge_index = load_class_graph(class_name)
        
        save_path = output_dir / f"{class_name}_graph.pt"
        torch.save({
            "vertex_embeds": vertex_embeds,
            "edge_index": edge_index
        }, save_path)
        
    print(f"saved all graphs successfully to folder {output_dir}")

if __name__ == "__main__":
    save_all_graphs()