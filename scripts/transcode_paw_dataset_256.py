"""Transcode a raw-resolution PAW LeRobot dataset to the processed 256x256 convention.

Raw PAW recordings store cameras at native resolution (720x1280 static, 480x640
wrists) plus extra state features (e.g. desired_tcp_pose_*). The processed
datasets we train and deploy on use a horizontal-center square crop (full
height), resized to 256x256, and a fixed feature set. This script rewrites a
raw dataset to match a processed reference dataset's schema so the two can be
merged with ``lerobot.datasets.aggregate.aggregate_datasets``.

Usage:
    uv run scripts/transcode_paw_dataset_256.py \
        --src pokeandwiggle/interventions_..._20_int \
        --ref pokeandwiggle/stack_duplo_..._07-08T19-48 \
        --dst pokeandwiggle/interventions_..._20_int_256
"""

import numpy as np
from PIL import Image
import torch
import tqdm
import tyro

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

# Keys lerobot adds on its own; they must not be passed to create().
_AUTO_KEYS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def _crop_resize_256(img: torch.Tensor) -> np.ndarray:
    """CHW float32 [0,1] -> horizontal-center square crop -> 256x256 HWC uint8."""
    arr = (img.numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
    h, w = arr.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    arr = arr[y0 : y0 + s, x0 : x0 + s]
    if s == 256:
        return arr
    return np.asarray(Image.fromarray(arr).resize((256, 256), Image.BILINEAR))


def main(src: str, ref: str, dst: str) -> None:
    ref_features = {
        key: {k: v for k, v in feat.items() if k in ("dtype", "shape", "names")}
        for key, feat in LeRobotDatasetMetadata(ref).features.items()
        if key not in _AUTO_KEYS
    }
    video_keys = [k for k, f in ref_features.items() if f["dtype"] == "video"]
    data_keys = [k for k, f in ref_features.items() if f["dtype"] != "video"]

    src_ds = LeRobotDataset(src)
    dst_ds = LeRobotDataset.create(
        dst, fps=src_ds.meta.fps, features=ref_features, robot_type=src_ds.meta.robot_type
    )

    prev_ep = None
    for i in tqdm.tqdm(range(len(src_ds)), desc="Transcoding"):
        item = src_ds[i]
        ep = int(item["episode_index"])
        if prev_ep is not None and ep != prev_ep:
            dst_ds.save_episode()
        prev_ep = ep

        frame = {key: _crop_resize_256(item[key]) for key in video_keys}
        for key in data_keys:
            # getitem squeezes scalars to 0-dim; restore the declared shape.
            frame[key] = np.asarray(item[key], dtype=np.float32).reshape(ref_features[key]["shape"])
        frame["task"] = item["task"]
        dst_ds.add_frame(frame)
    dst_ds.save_episode()
    print(f"Done: {dst} ({dst_ds.meta.total_episodes} episodes, {dst_ds.meta.total_frames} frames)")


if __name__ == "__main__":
    tyro.cli(main)
