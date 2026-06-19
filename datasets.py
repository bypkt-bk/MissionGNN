from pathlib import Path
from typing import List, Optional
import random, glob
import tqdm
import torch
from torch.utils.data import Dataset
import numpy as np
from config import cfg


class SensorSequenceDataset(Dataset):
    def __init__(
        self,
        classes: List[str],
        positives: List[Path],
        negatives: Optional[List[Path]] = None,
        normals: Optional[List[Path]] = None,
    ):
        self.classes = classes
        files_list, targets_list = [], []
        rng = random.Random(cfg.seed)

        # --- Positive ---
        for positive in positives:
            for label, cls in enumerate(classes, start=1):
                found = sorted(positive.rglob(f"{cls}*.pt"))
                rng.shuffle(found)
                files_list += found
                targets_list += [label] * len(found)

        # --- Normals ---
        def _add_glob(paths: List[Path], label_val: int):
            for p in paths:
                found = sorted(p.rglob("*.pt"))
                rng.shuffle(found)
                files_list.extend(found)
                targets_list.extend([label_val] * len(found))

        if negatives:
            _add_glob(negatives, 0)
        if normals:
            _add_glob(normals, 0)

        sort_idx = np.argsort([str(f) for f in files_list])
        files_sorted   = [files_list[i]   for i in sort_idx]
        targets_sorted = [targets_list[i] for i in sort_idx]

        print(f"Preloading {len(files_sorted)} files...")
        all_feats, all_targets, first_idx_map = [], [], []
        last_vid = ""
        current_start = 0
        current_idx = 0

        for fp, tgt in tqdm.tqdm(zip(files_sorted, targets_sorted), total=len(files_sorted)):
            try:
                data = torch.load(fp, map_location="cpu")
                feats = data['features'] if isinstance(data, dict) else data
                if feats.dim() == 1:
                    feats = feats.unsqueeze(0)
            except Exception as e:
                print(f"Error: {fp}: {e}")
                continue

            vid_name = fp.name.split('_part')[0]
            if vid_name != last_vid:
                current_start = current_idx
                last_vid = vid_name

            n = feats.size(0)
            all_feats.append(feats)
            all_targets.extend([tgt] * n)
            first_idx_map.extend([current_start] * n)
            current_idx += n

        self.features  = torch.cat(all_feats, dim=0)
        self.targets   = torch.tensor(all_targets, dtype=torch.long)
        self.first_idx = torch.tensor(first_idx_map, dtype=torch.long)
        print(f"Loaded: {len(self.features)} frames | {self.features.element_size() * self.features.nelement() / 1e9:.2f} GB")

        unique, counts = torch.unique(self.targets, return_counts=True)
        print("\n=== Dataset Distribution ===")
        for c, n in zip(unique.tolist(), counts.tolist()):
            cls_name = "Normal" if c == 0 else classes[c-1]
            print(f"  class {c:2d} ({cls_name:20s}): {n:6d} frames ({100*n/len(self.targets):.1f}%)")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        start = max(self.first_idx[idx].item(), idx - 29)
        sensors_raw = self.features[start:idx+1]
        seq_len = sensors_raw.size(0)

        sensors = torch.zeros(30, sensors_raw.size(1))
        sensors[-seq_len:] = sensors_raw

        mask = torch.zeros(30)
        mask[-seq_len:] = 1

        return sensors, mask, self.targets[idx]
