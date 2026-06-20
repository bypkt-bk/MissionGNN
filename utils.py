import sys
import torchvision.transforms.functional as TF
sys.modules["torchvision.transforms.functional_tensor"] = TF
import torch
from torchmetrics import AUROC, AveragePrecision
from sklearn.metrics import f1_score
from functools import partial
from models.missiongnn import MissionGNN
from config import cfg
from pathlib import Path
class MetricCollection:
    def __init__(self, num_classes: int):
        self.aurocs = [AUROC(task="binary") for _ in range(num_classes + 1)]
        self.f1 = partial(f1_score, average='micro')
        self.ap = [AveragePrecision(task="binary") for _ in range(num_classes + 1)]

    def __call__(self, scores: torch.Tensor, targets: torch.Tensor):
        with torch.no_grad():
            f1 = self.f1(targets, torch.max(scores, 1).indices)
            aucs = [m(scores[:, i], targets == i) for i, m in enumerate(self.aurocs)]
            aps = [m(scores[:, i], targets == i) for i, m in enumerate(self.ap)]
        return dict(f1=f1, mauc=sum(aucs[1:]).item() / len(aucs[1:]), map=sum(aps[1:]).item() / len(aps[1:]))


def configure_from_checkpoint(checkpoint: Path) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    gnn_hidden = state["branches.0.conv1.lin.weight"].shape[0]
    transformer_d_model = state["temporal.cls_head.weight"].shape[1]
    transformer_ff = state["temporal.transformer.layers.0.linear1.weight"].shape[0]

    cfg.gnn_hidden = gnn_hidden
    cfg.transformer_d_model = transformer_d_model
    cfg.transformer_ff = transformer_ff


def load_missiongnn(checkpoint: Path, device: str) -> MissionGNN:
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    configure_from_checkpoint(checkpoint)
    model = MissionGNN().to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model_state = model.state_dict()
    mismatched = []
    for key in state:
        if key not in model_state:
            mismatched.append((key, "MISSING_IN_MODEL", state[key].shape, None))
        elif model_state[key].shape != state[key].shape:
            mismatched.append((key, "SHAPE_MISMATCH", state[key].shape, model_state[key].shape))

    missing_in_ckpt = [k for k in model_state if k not in state]

    print(f"=== {len(mismatched)} mismatched keys ===")
    for key, reason, ckpt_shape, model_shape in mismatched:
        print(f"  {reason:16s} {key:50s} ckpt={ckpt_shape} model={model_shape}")

    print(f"\n=== {len(missing_in_ckpt)} keys in model but missing in checkpoint ===")
    for key in missing_in_ckpt:
        print(f"  {key}")

    compatible = {
        key: value
        for key, value in state.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    skipped = len(state) - len(compatible)
    if skipped:
        print(f"Warning: skipped {skipped} checkpoint keys with shape mismatches.")
    model.load_state_dict(compatible, strict=False)
    model.eval()
    return model