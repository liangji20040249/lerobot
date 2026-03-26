#!/usr/bin/env python3
"""
用 LeRobot 官方 API 将 HDF5 原始数据转换为 LeRobot 标准数据集。
需要在安装了 lerobot 的环境中运行（服务器上）。

用法：
  python3 convert_to_lerobot.py

输入：  ./raw_data/episode_XXXX.h5
输出：  ./lerobot_dataset/
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"]     = "/root/autodl-tmp/hf_cache"

import numpy as np
import h5py
from pathlib import Path

RAW_DIR = Path("/root/autodl-tmp/raw_data")
OUT_DIR = Path("/root/autodl-tmp/lerobot_dataset_v2")
FPS     = 10
CAMERAS = ["cam1", "cam2", "dogcam", "wristcam"]
IMG_KEY = lambda cam: f"observation.images.{cam}"

# ── State / Action 字段名 ─────────────────────────────────────────────────────
LEGS   = ["fr", "fl", "rr", "rl"]
JOINTS = ["abad", "hip", "knee", "foot"]
STATE_NAMES = {
    "motors": (
        [f"jp_{l}_{j}" for l in LEGS for j in JOINTS] +
        [f"jv_{l}_{j}" for l in LEGS for j in JOINTS] +
        [f"jt_{l}_{j}" for l in LEGS for j in JOINTS] +
        ["quat_w","quat_x","quat_y","quat_z"] +
        ["gyro_x","gyro_y","gyro_z"] +
        ["acc_x","acc_y","acc_z"] +
        ["vel_x","vel_y","vel_z"] +
        [f"contact_{l}" for l in LEGS]
    )
}
ACTION_NAMES = {"motors": [f"target_{l}_{j}" for l in LEGS for j in JOINTS]}

# ── LeRobot features 定义 ─────────────────────────────────────────────────────
features = {
    "observation.state": {
        "dtype": "float32",
        "shape": (65,),
        "names": STATE_NAMES,
    },
    "action": {
        "dtype": "float32",
        "shape": (16,),
        "names": ACTION_NAMES,
    },
    **{
        IMG_KEY(cam): {
            "dtype": "video",
            "shape": (360, 640, 3),
            "names": ["height", "width", "channel"],
        }
        for cam in CAMERAS
    },
}

# ── 创建数据集 ────────────────────────────────────────────────────────────────
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 清理旧目录
import shutil
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)

dataset = LeRobotDataset.create(
    repo_id="robot_dog/zsl1w",
    fps=FPS,
    features=features,
    root=str(OUT_DIR),
    robot_type="zsl1w",
    use_videos=True,
    vcodec="h264",          # h264 编码速度快，兼容性好
)
print(f"数据集创建于: {OUT_DIR}\n")

# ── 逐 episode 写入 ───────────────────────────────────────────────────────────
h5_files = sorted(RAW_DIR.glob("episode_*.h5"))
assert h5_files, f"未找到 HDF5 文件: {RAW_DIR}"
print(f"找到 {len(h5_files)} 个 episode\n")

for ep_idx, h5_path in enumerate(h5_files):
    with h5py.File(h5_path, "r") as f:
        task_str = str(f.attrs["task"])
        T        = f["states"].shape[0]
        states   = f["states"][:]       # (T, 65)
        actions  = f["actions"][:]      # (T, 16)

        # 预读所有相机图像
        cam_imgs = {cam: f[cam][:] for cam in CAMERAS}  # (T, H, W, 3)

    print(f"[ep {ep_idx:02d}] T={T}  task='{task_str}'")

    for t in range(T):
        frame = {
            "observation.state": states[t],          # (65,) float32
            "action":            actions[t],          # (16,) float32
            "task":              task_str,
        }
        for cam in CAMERAS:
            frame[IMG_KEY(cam)] = cam_imgs[cam][t]   # (H, W, 3) uint8

        dataset.add_frame(frame)

    dataset.save_episode()
    print(f"  → episode {ep_idx} 保存完成")

# 必须调用 finalize，否则 episodes parquet 不会落盘
dataset.finalize()

print(f"\n全部完成！数据集目录: {OUT_DIR}")
print(f"总帧数: {dataset.meta.total_frames}")
print(f"总 episodes: {dataset.meta.total_episodes}")
