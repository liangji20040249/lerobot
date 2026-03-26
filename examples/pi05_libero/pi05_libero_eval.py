#!/usr/bin/env python
"""
PI0.5 LIBERO 在线评测脚本
- 在 LIBERO 仿真环境中运行 PI0.5 模型推理
- 输出：每任务成功率、汇总 JSON、每任务录制视频

用法：
    MUJOCO_GL=egl python pi05_libero_eval.py [--tasks 0,1,2] [--n_episodes 3]
"""

import os, sys, json, time, argparse
from pathlib import Path

# ── 环境变量（必须在 import torch/mujoco 之前设置） ──────────────────────────
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ── lerobot 源码路径 ──────────────────────────────────────────────────────────
LEROBOT_SRC = "/root/workspace/test_lerobot/lerobot/src"
if LEROBOT_SRC not in sys.path:
    sys.path.insert(0, LEROBOT_SRC)

import numpy as np
import torch
import gymnasium as gym

# ── Patch 1: 禁用 torch.compile（避免 800s autotune） ────────────────────────
torch.compile = lambda f, **kwargs: f


# ── Patch 2: tokenizer 用 slow 版本（避免 GemmaTokenizerFast tiktoken 依赖） ──
def _patch_tokenizer():
    from lerobot.processor import TokenizerProcessorStep
    from transformers import AutoTokenizer

    def patched_post_init(self):
        if self.tokenizer is not None:
            self.input_tokenizer = self.tokenizer
        elif self.tokenizer_name is not None:
            self.input_tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                cache_dir=os.environ["HF_HOME"],
                local_files_only=False,
                use_fast=False,
            )
        else:
            raise ValueError("Either tokenizer_name or tokenizer must be provided.")
        self.input_tokenizer.padding_side = self.padding_side

    TokenizerProcessorStep.__post_init__ = patched_post_init


# ── 常量 ─────────────────────────────────────────────────────────────────────
MODEL_HF_PATH   = "lerobot/pi05_libero_finetuned_quantiles_v044"
MODEL_PATH      = (
    "/root/autodl-tmp/hf_cache"
    "/models--lerobot--pi05_libero_finetuned_quantiles_v044"
    "/snapshots/e710c1b684ccc2cb7269d8020d04113497cb1f4e"
)
SUITE_NAME      = "libero_10"
DEVICE          = "cuda"
OUTPUT_DIR      = Path("/root/autodl-tmp/pi05_libero_results")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model_and_processors():
    _patch_tokenizer()
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors

    log("Loading PI0.5 model...")
    policy = PI05Policy.from_pretrained(MODEL_PATH)
    policy = policy.to(DEVICE)
    policy.eval()
    log(f"Model loaded. Device={DEVICE}")

    # 加载 normalizer / tokenizer 等 processor（从模型目录读取 json）
    log("Loading pre/post processors...")
    preprocessor_overrides = {
        "device_processor": {"device": DEVICE},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=MODEL_PATH,
        preprocessor_overrides=preprocessor_overrides,
    )
    log("Processors loaded.")
    return policy, preprocessor, postprocessor


def make_libero_env(suite, suite_name: str, task_id: int, n_envs: int = 1):
    """创建单任务的向量化 LIBERO 环境"""
    from lerobot.envs.libero import LiberoEnv
    from functools import partial

    def _make(episode_index):
        return LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=suite_name,
            camera_name="agentview_image,robot0_eye_in_hand_image",
            obs_type="pixels_agent_pos",
            episode_index=episode_index,
            init_states=True,
        )

    fns = [partial(_make, i) for i in range(n_envs)]
    vec_env = gym.vector.SyncVectorEnv(fns)
    return vec_env


def run_rollout(env, policy, preprocessor, postprocessor, env_preprocessor,
                max_steps: int, record_video: bool = True):
    """运行单次 rollout，返回 (is_success, frames)"""
    from lerobot.envs.utils import preprocess_observation, add_envs_task

    policy.reset()
    obs, info = env.reset()
    frames = []

    done = np.zeros(env.num_envs, dtype=bool)
    success = False

    for step in range(max_steps):
        # 记录帧（仅第 0 个环境）
        if record_video:
            frame = env.envs[0].render()  # (H, W, 3)
            frames.append(frame)

        # 预处理观测：gym 格式 → lerobot tensor 格式
        obs_t = preprocess_observation(obs)

        # 添加任务描述
        obs_t = add_envs_task(env, obs_t)

        # LiberoProcessorStep：翻转图像 + 构建 state 向量
        obs_t = env_preprocessor(obs_t)

        # policy 专属预处理（归一化 + tokenize）
        obs_t = preprocessor(obs_t)

        # 推理
        with torch.no_grad():
            action = policy.select_action(obs_t)

        # 后处理（反归一化）
        action = postprocessor(action)

        # action → numpy
        action_np = action.to("cpu").numpy()  # (B, action_dim)

        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action_np)

        # 检查成功（lerobot 把 is_success 放在 final_info）
        if "final_info" in info:
            final_info = info["final_info"]
            if isinstance(final_info, dict) and "is_success" in final_info:
                success = bool(final_info["is_success"].any())
            elif isinstance(final_info, list):
                success = any(fi.get("is_success", False) for fi in final_info if fi)

        done = terminated | truncated | done
        if done.all():
            break

    return success, frames


def save_video(frames, path: Path, fps: int = 10):
    """用 imageio 保存视频（如果安装了的话，否则 fallback 到 opencv）"""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio
        imageio.mimwrite(str(path), frames, fps=fps, quality=7)
        return True
    except ImportError:
        pass
    try:
        import cv2
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        return True
    except Exception as e:
        log(f"  [warn] video save failed: {e}")
        return False


def evaluate(task_ids=None, n_episodes=3, n_video_per_task=1):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    videos_dir = OUTPUT_DIR / "videos"

    # 加载模型
    policy, preprocessor, postprocessor = load_model_and_processors()

    # 构建 env_preprocessor（LiberoProcessorStep）
    from lerobot.processor.env_processor import LiberoProcessorStep
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    env_preprocessor = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])

    # 加载 LIBERO suite
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[SUITE_NAME]()
    total_tasks = len(suite.tasks)
    if task_ids is None:
        task_ids = list(range(total_tasks))

    log(f"Suite={SUITE_NAME}, tasks={task_ids}, n_episodes={n_episodes}")
    log(f"Output dir: {OUTPUT_DIR}")

    all_results = []
    overall_success = []

    for task_id in task_ids:
        task_lang = suite.tasks[task_id].language
        log(f"\n{'='*60}")
        log(f"Task {task_id}: {task_lang}")
        log(f"{'='*60}")

        # 每个 episode 单独创建环境（确保 init_state 对应正确）
        task_successes = []
        task_videos = []

        for ep_idx in range(n_episodes):
            log(f"  Episode {ep_idx+1}/{n_episodes} ...")
            t0 = time.time()

            # 创建 1 个环境，episode_index = ep_idx
            from lerobot.envs.libero import LiberoEnv
            env = gym.vector.SyncVectorEnv([
                lambda tid=task_id, ei=ep_idx: LiberoEnv(
                    task_suite=suite,
                    task_id=tid,
                    task_suite_name=SUITE_NAME,
                    camera_name="agentview_image,robot0_eye_in_hand_image",
                    obs_type="pixels_agent_pos",
                    episode_index=ei,
                    init_states=True,
                )
            ])

            max_steps = env.call("_max_episode_steps")[0]
            record = (ep_idx < n_video_per_task)

            try:
                success, frames = run_rollout(
                    env, policy, preprocessor, postprocessor,
                    env_preprocessor, max_steps=max_steps, record_video=record
                )
            except Exception as e:
                log(f"  [error] episode failed: {e}")
                import traceback; traceback.print_exc()
                success = False
                frames = []
            finally:
                env.close()

            elapsed = time.time() - t0
            task_successes.append(success)
            log(f"  Episode {ep_idx+1}: success={success}, time={elapsed:.1f}s")

            # 保存视频
            if record and frames:
                vid_path = videos_dir / f"task{task_id:02d}_ep{ep_idx:02d}.mp4"
                saved = save_video(frames, vid_path, fps=10)
                if saved:
                    task_videos.append(str(vid_path))
                    log(f"  Video saved: {vid_path}")

        task_sr = np.mean(task_successes)
        overall_success.extend(task_successes)
        log(f"Task {task_id} success_rate: {task_sr*100:.1f}% ({sum(task_successes)}/{n_episodes})")

        all_results.append({
            "task_id": task_id,
            "task": task_lang,
            "success_rate": float(task_sr),
            "successes": task_successes,
            "video_paths": task_videos,
        })

    # 汇总
    overall_sr = float(np.mean(overall_success))
    summary = {
        "suite": SUITE_NAME,
        "task_ids": task_ids,
        "n_episodes_per_task": n_episodes,
        "overall_success_rate": overall_sr,
        "per_task": all_results,
    }

    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    log(f"\n{'='*60}")
    log(f"OVERALL SUCCESS RATE: {overall_sr*100:.1f}%")
    log(f"Results saved: {out_path}")
    log(f"Videos dir: {videos_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks", type=str, default=None,
        help="逗号分隔的 task_id，如 '0,1,2,3'。默认全部 10 个任务"
    )
    parser.add_argument(
        "--n_episodes", type=int, default=3,
        help="每个任务的 episode 数量（默认 3）"
    )
    parser.add_argument(
        "--n_video", type=int, default=1,
        help="每个任务录制视频的 episode 数量（默认 1）"
    )
    args = parser.parse_args()

    task_ids = None
    if args.tasks:
        task_ids = [int(x.strip()) for x in args.tasks.split(",")]

    evaluate(task_ids=task_ids, n_episodes=args.n_episodes, n_video_per_task=args.n_video)


if __name__ == "__main__":
    main()
