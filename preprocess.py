from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import cv2
import imagebind
import numpy as np
import torch
import tqdm
from sklearn.model_selection import train_test_split
from torchvision.io import read_video

from imagebind.models.imagebind_model import ModalityType


@dataclass(frozen=True)
class Config:
    video_root: Path = Path("./UCFCrimeDataset/Anomaly-Videos")
    image_root: Path = Path("./Anomaly-Images")
    embedding_root: Path = Path("./UCFCrimeDataset/Embeddings")
    crime_classes: tuple[str, ...] = ("Abuse", "Arrest", "Arson", "Assault", "RoadAccidents", "Burglary", "Explosion", 
              "Fighting", "Robbery", "Shooting", "Stealing", "Shoplifting", "Vandalism")
    batch_size: int = 10
    train_split: float = 0.8
    random_state: int = 42
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_frames_for_video(video_path: Path, output_dir: Path) -> None:
    ensure_dir(output_dir)

    frames, _, _ = read_video(str(video_path), output_format="TCHW")

    for frame_idx in range(frames.shape[0]):
        frame_path = output_dir / f"{frame_idx:09d}.png"

        # Uncomment to skip existing frames
        # if frame_path.exists():
        #     continue

        frame = frames[frame_idx].permute(1, 2, 0).numpy()
        cv2.imwrite(str(frame_path), frame[:, :, ::-1])


def extract_all_frames(config: Config) -> None:
    for crime_class in tqdm.tqdm(config.crime_classes, desc="Classes"):
        class_video_dir = config.video_root / crime_class
        class_image_dir = config.image_root / crime_class

        if not class_video_dir.exists():
            print(f"Skipping missing directory: {class_video_dir}")
            continue

        video_files = sorted(
            [p for p in class_video_dir.iterdir() if p.is_file()]
        )

        for video_path in tqdm.tqdm(video_files, desc=crime_class, leave=False):
            scene_name = video_path.stem
            output_dir = class_image_dir / scene_name
            print(f"Extracting: {video_path}")
            extract_frames_for_video(video_path, output_dir)


def load_imagebind_model(device: str) -> torch.nn.Module:
    model = imagebind.models.imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)
    return model


def list_scene_dirs(image_root: Path) -> list[Path]:
    return sorted(
        Path(p) for p in glob.glob(str(image_root / "*" / "*")) if Path(p).is_dir()
    )


def scene_label(scene_path: Path) -> str:
    return scene_path.parent.name


def summarize_split(train_scenes: list[Path], valid_scenes: list[Path]) -> None:
    classes = sorted({scene_label(path) for path in train_scenes + valid_scenes})
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    counts = np.zeros((2, len(classes)), dtype=int)

    for split_idx, split_scenes in enumerate((train_scenes, valid_scenes)):
        for path in split_scenes:
            counts[split_idx, class_to_idx[scene_label(path)]] += 1

    percentages = counts / counts.sum(axis=1, keepdims=True) * 100

    print("Classes:", classes)
    print("Counts:\n", counts)
    print("Percentages:\n", percentages)
    print(f"Train scenes: {len(train_scenes)}")
    print(f"Valid scenes: {len(valid_scenes)}")


def build_embedding_output_path(
    embedding_split_dir: Path,
    scene_name: str,
    frame_stem: str,
) -> Path:
    return embedding_split_dir / f"{scene_name}_{frame_stem}.pt"


def get_missing_batch_items(
    image_paths: list[Path],
    embedding_split_dir: Path,
) -> tuple[bool, list[str], list[str]]:
    scene_names = [img_path.parent.name for img_path in image_paths]
    frame_stems = [img_path.stem for img_path in image_paths]

    missing = any(
        not build_embedding_output_path(embedding_split_dir, scene_name, frame_stem).exists()
        for scene_name, frame_stem in zip(scene_names, frame_stems)
    )

    return missing, scene_names, frame_stems


def save_embeddings(
    embeddings: torch.Tensor,
    scene_names: list[str],
    frame_stems: list[str],
    embedding_split_dir: Path,
) -> None:
    ensure_dir(embedding_split_dir)

    for scene_name, frame_stem, embedding in zip(scene_names, frame_stems, embeddings):
        output_path = build_embedding_output_path(
            embedding_split_dir=embedding_split_dir,
            scene_name=scene_name,
            frame_stem=frame_stem,
        )
        torch.save(embedding, output_path)


def embed_scene_frames(
    model: torch.nn.Module,
    scene_path: Path,
    device: str,
    batch_size: int,
    embedding_split_dir: Path,
) -> None:
    image_paths = sorted(scene_path.glob("*.png"))
    if not image_paths:
        return

    for start_idx in tqdm.tqdm(
        range(0, len(image_paths), batch_size),
        desc=f"Embedding {scene_path.name}",
        leave=False,
    ):
        batch_paths = image_paths[start_idx : start_idx + batch_size]

        missing, scene_names, frame_stems = get_missing_batch_items(
            image_paths=batch_paths,
            embedding_split_dir=embedding_split_dir,
        )
        if not missing:
            continue

        inputs = {
            ModalityType.VISION: imagebind.data.load_and_transform_vision_data(
                [str(path) for path in batch_paths],
                device,
            )
        }

        with torch.no_grad():
            embeddings = model(inputs)[ModalityType.VISION].cpu()

        save_embeddings(
            embeddings=embeddings,
            scene_names=scene_names,
            frame_stems=frame_stems,
            embedding_split_dir=embedding_split_dir,
        )


def generate_train_embeddings(config: Config) -> None:
    scenes = list_scene_dirs(config.image_root)
    if not scenes:
        print(f"No scene directories found under: {config.image_root}")
        return

    train_scenes, valid_scenes = train_test_split(
        scenes,
        test_size=1.0 - config.train_split,
        random_state=config.random_state,
    )

    summarize_split(train_scenes, valid_scenes)

    model = load_imagebind_model(config.device)
    train_embedding_dir = config.embedding_root / "Train"
    ensure_dir(train_embedding_dir)

    for scene_path in tqdm.tqdm(train_scenes, desc="Train scenes"):
        print(f"Processing scene: {scene_path}")
        embed_scene_frames(
            model=model,
            scene_path=scene_path,
            device=config.device,
            batch_size=config.batch_size,
            embedding_split_dir=train_embedding_dir,
        )


def main() -> None:
    config = Config()

    # Step 1: Extract frames from videos
    extract_all_frames(config)

    # Step 2: Generate ImageBind embeddings for training scenes
    generate_train_embeddings(config)


if __name__ == "__main__":
    main()
