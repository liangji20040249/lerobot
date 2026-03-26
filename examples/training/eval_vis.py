#!/usr/bin/env python3
import os
import torch
import numpy as np
import imageio
import torchvision.transforms as T
from pathlib import Path
import sys

# --- 1. 导入策略 ---
try:
    from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
except ImportError:
    try:
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    except:
        print("❌ 无法导入 DiffusionPolicy")
        exit(1)

# --- 2. 导入环境 ---
print("🔍 导入 PushTEnv...")
PushTEnv = None
try:
    from gym_pusht.envs.pusht import PushTEnv
except ImportError:
    try:
        from gym_pusht.envs.pusht_env import PushTEnv
    except ImportError:
        pass

if PushTEnv is None:
    print("❌ 找不到 PushTEnv，请确认 gym-pusht 已安装")
    exit(1)

def find_latest_model():
    base_dir = Path("outputs")
    if not base_dir.exists():
        print("❌ 找不到 outputs 目录")
        exit(1)
    models = list(base_dir.rglob("model.safetensors"))
    if not models:
        print("❌ 未找到模型文件")
        exit(1)
    return max(models, key=os.path.getmtime).parent

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型
    model_path = find_latest_model()
    print(f"✅ Loading Model from: {model_path}")
    policy = DiffusionPolicy.from_pretrained(str(model_path))
    policy.to(device)
    policy.eval()

    # 2. 实例化环境
    print("🌍 Instantiating PushTEnv...")
    try:
        env = PushTEnv(render_mode="rgb_array")
    except:
        env = PushTEnv() # 旧版兼容

    print("🎬 Starting Rollout...")
    
    # Reset
    reset_res = env.reset()
    if isinstance(reset_res, tuple):
        observation = reset_res[0]
    else:
        observation = reset_res
    
    frames = []
    max_steps = 300
    resize_transform = T.Resize((96, 96), antialias=True)
    
    for step in range(max_steps):
        # --- A. 渲染 (获取图像) ---
        try:
            frame = env.render()
        except:
            frame = env.render(mode='rgb_array')

        if frame is None:
            # 如果 render 失败，尝试从 observation 里找（虽然不太可能）
            print("Warning: Render returned None")
            continue
            
        frames.append(frame)

        # --- B. 数据适配 (关键修复) ---
        
        # 1. 图像处理: 直接使用渲染出来的 frame
        # frame 是 (H, W, C), uint8
        img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        
        # 强制缩放至 96x96 (模型输入要求)
        if img_tensor.shape[1] != 96:
            img_tensor = resize_transform(img_tensor)

        # 2. 状态处理: 适配 observation 类型
        if isinstance(observation, dict):
            # 如果碰巧是字典
            state = observation['agent_pos']
        else:
            # 【核心修复】如果 observation 是数组
            # PushT 原生 state: [agent_x, agent_y, block_x, block_y, ...]
            # 我们只需要前两维 (agent_pos)
            state = observation[:2]

        state_tensor = torch.from_numpy(state).float()
        
        # 3. 组装 Batch
        batch = {
            'observation.image': img_tensor.unsqueeze(0).to(device),
            'observation.state': state_tensor.unsqueeze(0).to(device)
        }

        # --- C. 推理 ---
        with torch.no_grad():
            action = policy.select_action(batch)

        # --- D. 执行 ---
        action_np = action.squeeze(0).cpu().numpy()
        step_res = env.step(action_np)
        
        # 解包
        if len(step_res) == 5:
            observation, reward, term, trunc, info = step_res
            done = term or trunc
        else:
            observation, reward, done, info = step_res

        if done:
            print(f"🏁 Done at step {step}")
            break
            
        if step % 50 == 0:
            print(f"Step {step}/{max_steps}...")

    env.close()

    # 3. 保存
    save_path = "eval_pusht_final.mp4"
    if len(frames) > 0:
        print(f"💾 Saving {len(frames)} frames to {save_path}...")
        imageio.mimsave(save_path, frames, fps=20)
        print(f"✅ 视频已生成: {os.path.abspath(save_path)}")
    else:
        print("❌ 未捕获到画面。")

if __name__ == "__main__":
    main()