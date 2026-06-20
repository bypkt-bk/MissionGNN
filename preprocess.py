from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import tqdm

# ---- FIX pytorchvideo bug (must run before pytorchvideo imports) ----
sys.modules["torchvision.transforms.functional_tensor"] = F

from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from pytorchvideo.models.hub import slowfast_r50
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

# ---- Feature dims ----
SLOWFAST_DIM = 2304
IMAGEBIND_DIM = 1024
FUSED_DIM = SLOWFAST_DIM + IMAGEBIND_DIM  # 3328


@dataclass(frozen=True)
class Config:
    video_root: Path = Path("./UCFCrimeDataset/Anomaly-Videos")
    embedding_root: Path = Path("./UCFCrimeDataset/Embeddings")
    crime_classes: tuple[str, ...] = ("Abuse", "Arrest", "Arson", "Assault", "RoadAccidents", "Burglary", "Explosion", 
              "Fighting", "Robbery", "Shooting", "Stealing", "Shoplifting", "Vandalism",)
    video_extensions: tuple[str, ...] = (".mp4", ".avi", ".mkv")
    clip_duration: float = 2.0
    stride: float = 1.0
    min_fps: float = 16.0
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_slowfast_transform(num_frames: int = 32) -> ApplyTransformToKey:
    class PackPath(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.alpha = 4

        def forward(self, frames: torch.Tensor) -> list[torch.Tensor]:
            fast = frames
            slow = torch.index_select(
                frames,
                1,
                torch.linspace(
                    0, frames.shape[1] - 1, frames.shape[1] // self.alpha
                ).long(),
            )
            return [slow, fast]

    return ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo([0.45] * 3, [0.225] * 3),
                ShortSideScale(256),
                CenterCropVideo(256),
                PackPath(),
            ]
        ),
    )


def get_imagebind_transform() -> Compose:
    return Compose(
        [
            UniformTemporalSubsample(3),
            Lambda(lambda x: x / 255.0),
            NormalizeVideo(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
            ShortSideScale(224),
            CenterCropVideo(224),
        ]
    )


class FusionModel(nn.Module):
    """SlowFast + ImageBind dual-backbone fusion with learnable per-modality scaling."""

    def __init__(self, device: str = "cuda") -> None:
        super().__init__()
        self.device = torch.device(device)

        self.slowfast = slowfast_r50(pretrained=True)
        self.slowfast.blocks[-1] = nn.Identity()
        self.slowfast.to(self.device).eval()

        self.imagebind = imagebind_model.imagebind_huge(pretrained=True)
        self.imagebind.to(self.device).eval()

        # Learnable sigmoid-bounded scale per modality, range (0, 2)
        self.sf_scale = nn.Parameter(torch.tensor(0.0))
        self.ib_scale = nn.Parameter(torch.tensor(0.0))

        self.to(self.device)

    def forward(
        self, sf_input: list[torch.Tensor], ib_input: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            sf_feat = self.slowfast(sf_input)
            if sf_feat.dim() == 5:
                sf_feat = sf_feat.mean(dim=[2, 3, 4])
            sf_feat = nn.functional.normalize(sf_feat, dim=1)

            ib_feat = self.imagebind({ModalityType.VISION: ib_input})[
                ModalityType.VISION
            ]
            ib_feat = nn.functional.normalize(ib_feat, dim=1)

        sf_scale = torch.sigmoid(self.sf_scale) * 2
        ib_scale = torch.sigmoid(self.ib_scale) * 2

        sf_feat = sf_scale * sf_feat
        ib_feat = ib_scale * ib_feat

        fused = torch.cat([sf_feat, ib_feat], dim=1)
        fused = nn.functional.normalize(fused, dim=1)
        return fused


def validate_video(
    video: EncodedVideo, clip_duration: float, min_fps: float
) -> tuple[bool, float]:
    try:
        clip = video.get_clip(0, min(clip_duration, float(video.duration)))
        frames = clip.get("video")
        if frames is None:
            return False, 0.0
        fps = frames.shape[1] / clip_duration
        return fps >= min_fps, fps
    except Exception:
        return False, 0.0


def build_output_path(embedding_class_dir: Path, video_path: Path) -> Path:
    return embedding_class_dir / f"{video_path.stem}.pt"


def embed_video(
    model: FusionModel,
    sf_tf: ApplyTransformToKey,
    ib_tf: Compose,
    video_path: Path,
    output_path: Path,
    config: Config,
) -> None:
    video = EncodedVideo.from_path(str(video_path), decode_audio=False)

    ok, fps = validate_video(video, config.clip_duration, config.min_fps)
    if not ok:
        print(f"Skipping (low fps or unreadable): {video_path}")
        return

    current = 0.0
    feats: list[torch.Tensor] = []
    times: list[float] = []

    while current < video.duration - config.clip_duration:
        clip = video.get_clip(current, current + config.clip_duration)

        if clip.get("video") is None:
            current += config.stride
            continue

        sf = sf_tf(clip.copy())["video"]
        sf = [x.unsqueeze(0).to(config.device) for x in sf]

        ib = clip["video"].float()
        ib = ib_tf(ib).unsqueeze(0).to(config.device)

        with torch.no_grad():
            feat = model(sf, ib)

        feats.append(feat.cpu())
        times.append(current + config.clip_duration / 2)

        current += config.stride

    if not feats:
        print(f"No valid clips extracted: {video_path}")
        return

    tensor = torch.vstack(feats)
    tensor = nn.functional.normalize(tensor, dim=1)  # per-video normalization

    torch.save(
        {
            "features": tensor,
            "timestamps": times,
            "fps": fps,
            "clip_duration": config.clip_duration,
            "stride": config.stride,
        },
        output_path,
    )


def list_class_videos(
    class_video_dir: Path, video_extensions: tuple[str, ...]
) -> list[Path]:
    if not class_video_dir.exists():
        return []
    return sorted(
        p
        for p in class_video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in video_extensions
    )


def generate_embeddings(config: Config) -> None:
    ensure_dir(config.embedding_root)

    model = FusionModel(device=config.device).eval()
    sf_tf = get_slowfast_transform()
    ib_tf = get_imagebind_transform()

    for crime_class in tqdm.tqdm(config.crime_classes, desc="Classes"):
        class_video_dir = config.video_root / crime_class
        video_files = list_class_videos(class_video_dir, config.video_extensions)

        if not video_files:
            print(f"Skipping missing or empty directory: {class_video_dir}")
            continue

        embedding_class_dir = config.embedding_root / crime_class
        ensure_dir(embedding_class_dir)

        for video_path in tqdm.tqdm(video_files, desc=crime_class, leave=False):
            output_path = build_output_path(embedding_class_dir, video_path)
            if output_path.exists():
                continue

            try:
                embed_video(model, sf_tf, ib_tf, video_path, output_path, config)
            except Exception as e:
                print(f"Failed: {video_path}: {e}")


def main() -> None:
    config = Config()
    generate_embeddings(config)


if __name__ == "__main__":
    main()
