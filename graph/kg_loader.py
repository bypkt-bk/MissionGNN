"""Load vertices + edges for each anomaly class and build tensors ready for PyG."""
from pathlib import Path
from typing import List, Tuple
import torch
from imagebind.models.imagebind_model import ModalityType
import imagebind

from config import cfg

model = imagebind.models.imagebind_model.imagebind_huge(pretrained=True)
model.eval().to(cfg.device)


def _embed_words(words: List[str]) -> torch.Tensor:
    """Return ImageBind text embeddings stacked into [len(words), 1024]."""
    inputs = {
        ModalityType.TEXT: imagebind.data.load_and_transform_text(
            [f"this CCTV footage is related to {w}" for w in words], cfg.device
        ),
    }
    with torch.no_grad():
        return model(inputs)[ModalityType.TEXT].cpu()


def _fuzzy_match(keyword: str, vertices: list) -> str:
    """Find the vertex that best matches the given keyword using fuzzy string matching.
    Returns the matched vertex name or None if no match is found."""
    kw_lower = keyword.lower()
    # exact match ก่อน
    for v in vertices:
        if v.lower() == kw_lower:
            return v
    for v in vertices:
        if kw_lower in v.lower():
            return v
    for v in vertices:
        if v.lower() in kw_lower and len(v) > 3:  # min length 4 ป้องกัน false positive
            return v
    return None


def load_class_graph(class_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (vertex_embeddings, edge_index) for a given anomaly class."""
    # -- read edge list
    with open(cfg.subgraph_dir / f"subgraph_{class_name}.txt", "r") as f:
        edges = [tuple(line.split("->")) for line in f.read().splitlines()]
    vertices = sorted({v for e in edges for v in e})

    # build edge index (source, target)
    edge_index = torch.tensor(
        [[vertices.index(s), vertices.index(t)] for s, t in edges], dtype=torch.long
    )

    # attach sensor + encoding nodes
    sensor_idx, encoding_idx = len(vertices), len(vertices) + 1
    # sensor connects TO each keyword node (fuzzy match)
    with open(cfg.subgraph_dir / f"keywords_{class_name}.txt", "r") as f:
        keywords = [w.strip().lower() for w in f.read().splitlines()]
    matched_vertices = set()
    unmatched = []
    for kw in keywords:
        matched_v = _fuzzy_match(kw, vertices)
        if matched_v is None:
            unmatched.append(kw)
            continue
        matched_vertices.add(matched_v)
        edge_index = torch.vstack([
            edge_index,
            torch.tensor([[sensor_idx, vertices.index(matched_v)]])
        ])

    if unmatched:
        print(f"[WARN] {class_name}: {len(unmatched)}/{len(keywords)} keywords unmatched "
              f"(matched {len(matched_vertices)} vertices via fuzzy)")
    else:
        print(f"[OK]   {class_name}: all {len(keywords)} keywords matched")

    if not matched_vertices:
        print(f"[FALLBACK] {class_name}: sensor connecting to ALL {len(vertices)} vertices")
        for i in range(len(vertices)):
            edge_index = torch.vstack([
                edge_index,
                torch.tensor([[sensor_idx, i]])
            ])

    # every non-keyword leaf CONNECTS TO encoding node
    for v in vertices:
        if v not in matched_vertices:
            edge_index = torch.vstack([
                edge_index,
                torch.tensor([[vertices.index(v), encoding_idx]])
            ])

    vertices.extend(["<SENSOR>", "<ENCODING>"])
    
    # embed vertices
    vertex_embeds = _embed_words(vertices[:-2])

    return vertex_embeds, edge_index.T.contiguous()