#!/usr/bin/env python
"""
PI0.5 LIBERO 离线推理与评测脚本 v3
修复:
  - embed_tokens weight tying (checkpoint 省略了与 lm_head 共享的 embed_tokens.weight)
  - compile_model=False (避免 torch.compile 编译超时)
  - tokenizer 用 slow 版本
"""

import os, sys, json, time
os.environ["HF_ENDPOINT"]     = "https://hf-mirror.com"
os.environ["HF_HOME"]         = "/root/autodl-tmp/hf_cache"

import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

MODEL_ID   = "lerobot/pi05_libero_finetuned_quantiles_v044"
DATASET_ID = "lerobot/libero_10_image"
OUTPUT_DIR = Path("/root/autodl-tmp/pi05_inference_results")
N_EPISODES = 3
DEVICE     = "cuda"
ACTION_DIM = 7
ACTION_NAMES = ["eef_x", "eef_y", "eef_z", "eef_rx", "eef_ry", "eef_rz", "gripper"]

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Patch TokenizerProcessorStep ───────────────────────────────────────────────
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
    log("Tokenizer patched (use_fast=False)")


# ── 加载模型（修复版：monkey-patch torch.compile 避免 draccus 问题）────────────
def load_model_and_processors():
    from huggingface_hub import snapshot_download
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors

    # ★ 禁用 torch.compile，避免 800s+ autotune 开销
    _orig_compile = torch.compile
    torch.compile = lambda f, **kwargs: f
    log("torch.compile patched to noop (skip autotune)")

    log(f"Locating model: {MODEL_ID}")
    model_path = snapshot_download(MODEL_ID, cache_dir=os.environ["HF_HOME"], ignore_patterns=["*.git*"])
    log(f"Model path: {model_path}")

    log("Loading PI05Policy.from_pretrained ...")
    policy = PI05Policy.from_pretrained(model_path)
    policy.eval().to(DEVICE)
    log(f"Policy on {DEVICE}, dtype={policy.config.dtype}, compile=noop")

    # 恢复 torch.compile
    torch.compile = _orig_compile

    _patch_tokenizer()

    log("Loading preprocessor/postprocessor ...")
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=model_path)
    log("Preprocessors ready.")
    return policy, preprocessor, postprocessor

# ── 加载数据集 ──────────────────────────────────────────────────────────────────
def load_dataset(n_episodes):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    log(f"Loading dataset: {DATASET_ID}")
    meta = LeRobotDatasetMetadata(DATASET_ID)
    task_map = {int(row["task_index"]): task for task, row in meta.tasks.iterrows()}
    log(f"  episodes={meta.total_episodes}, frames={meta.total_frames}, fps={meta.fps}")

    log(f"Downloading {n_episodes} episodes ...")
    dataset = LeRobotDataset(DATASET_ID, episodes=list(range(n_episodes)))
    log(f"  {len(dataset)} frames loaded.")
    return dataset, meta, task_map


# ── 单 episode 推理 ────────────────────────────────────────────────────────────
def infer_episode(policy, preprocessor, postprocessor, frames, task_str):
    policy.reset()
    pred_actions, gt_actions = [], []

    for i, frame in enumerate(frames):
        batch = dict(frame)
        if "observation.images.wrist_image" in batch:
            batch["observation.images.image2"] = batch.pop("observation.images.wrist_image")
        batch["task"] = task_str

        preprocessed = preprocessor(batch)

        with torch.no_grad():
            raw_action = policy.select_action(preprocessed)   # (1, 7)

        pred = postprocessor(raw_action).squeeze(0).cpu().float().numpy()
        pred_actions.append(pred)
        gt_actions.append(frame["action"].cpu().float().numpy())

        if (i + 1) % 50 == 0:
            log(f"    frame {i+1}/{len(frames)} ...")

    return np.array(pred_actions), np.array(gt_actions)


# ── 指标 ────────────────────────────────────────────────────────────────────────
def compute_metrics(pred, gt):
    err  = pred - gt
    mae  = np.abs(err).mean(axis=0)
    mse  = (err ** 2).mean(axis=0)
    rmse = np.sqrt(mse)
    dot  = (pred * gt).sum(axis=-1)
    norm = np.linalg.norm(pred, axis=-1) * np.linalg.norm(gt, axis=-1) + 1e-8
    cos  = (dot / norm).mean()
    return {"mae_per_joint": mae.tolist(), "mse_per_joint": mse.tolist(),
            "rmse_per_joint": rmse.tolist(), "mean_mae": float(mae.mean()),
            "mean_rmse": float(rmse.mean()), "cosine_sim": float(cos)}


# ── HTML 报告 ──────────────────────────────────────────────────────────────────
def generate_html_report(all_ep_results, output_path):
    import base64, io, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    def fig_to_b64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    episode_blocks = []
    for ep in all_ep_results:
        pred, gt = np.array(ep["pred_actions"]), np.array(ep["gt_actions"])
        m, T = ep["metrics"], pred.shape[0]

        # Trajectory plot
        fig = plt.figure(figsize=(18, 10))
        fig.suptitle(f"Episode {ep['episode_idx']}  —  \"{ep['task'][:90]}\"", fontsize=10, y=1.01)
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35)
        for j in range(ACTION_DIM):
            ax = fig.add_subplot(gs[j // 3, j % 3])
            ax.plot(np.arange(T), gt[:, j],   "b-",  lw=1.2, alpha=0.8, label="Ground Truth")
            ax.plot(np.arange(T), pred[:, j], "r--", lw=1.2, alpha=0.8, label="Predicted")
            ax.set_title(f"{ACTION_NAMES[j]}  MAE={m['mae_per_joint'][j]:.4f}", fontsize=8)
            ax.set_xlabel("Frame", fontsize=7); ax.tick_params(labelsize=6); ax.grid(alpha=0.3)
            if j == 0: ax.legend(fontsize=7)
        traj_b64 = fig_to_b64(fig)

        # Error heatmap
        fig2, ax2 = plt.subplots(figsize=(12, 3))
        im = ax2.imshow(np.abs(pred - gt).T, aspect="auto", cmap="hot")
        ax2.set_yticks(range(7)); ax2.set_yticklabels(ACTION_NAMES, fontsize=8)
        ax2.set_xlabel("Frame", fontsize=9); ax2.set_title("Absolute Error Heatmap")
        plt.colorbar(im, ax=ax2, label="|pred - gt|")
        heat_b64 = fig_to_b64(fig2)

        joint_rows = "".join(
            f"<tr><td>{n}</td><td>{m['mae_per_joint'][j]:.5f}</td>"
            f"<td>{m['rmse_per_joint'][j]:.5f}</td><td>{m['mse_per_joint'][j]:.6f}</td></tr>"
            for j, n in enumerate(ACTION_NAMES))

        episode_blocks.append(f"""
        <div class="ep-card">
          <h2>Episode {ep['episode_idx']}
            <span class="badge">CosSim {m['cosine_sim']:.4f}</span>
            <span class="badge green">Mean MAE {m['mean_mae']:.4f}</span>
            <span class="badge purple">Mean RMSE {m['mean_rmse']:.4f}</span>
          </h2>
          <p class="task-desc">Task: {ep['task']}</p>
          <p class="meta">Frames: {T} &nbsp;|&nbsp; fps=10 &nbsp;|&nbsp; Duration ≈ {T/10:.1f}s</p>
          <h4 style="color:#8b949e;font-size:13px;margin:12px 0 6px">Predicted vs Ground Truth</h4>
          <img src="data:image/png;base64,{traj_b64}" style="width:100%;border-radius:6px">
          <h4 style="color:#8b949e;font-size:13px;margin:16px 0 6px">Absolute Error Heatmap</h4>
          <img src="data:image/png;base64,{heat_b64}" style="width:100%;border-radius:6px">
          <table><thead><tr><th>Joint</th><th>MAE</th><th>RMSE</th><th>MSE</th></tr></thead>
          <tbody>{joint_rows}</tbody></table>
        </div>""")

    all_maes  = [ep["metrics"]["mean_mae"]  for ep in all_ep_results]
    all_rmses = [ep["metrics"]["mean_rmse"] for ep in all_ep_results]
    all_cos   = [ep["metrics"]["cosine_sim"] for ep in all_ep_results]
    gj_mae    = np.mean([ep["metrics"]["mae_per_joint"] for ep in all_ep_results], axis=0)
    gj_rows   = "".join(f"<tr><td>{n}</td><td>{gj_mae[j]:.5f}</td></tr>" for j,n in enumerate(ACTION_NAMES))

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<title>PI0.5 LIBERO 推理评测报告</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,"Segoe UI",sans-serif;background:#0d1117;color:#e6edf3;line-height:1.6}}
  .header{{background:#161b22;border-bottom:1px solid #30363d;padding:24px 48px}}
  .header h1{{font-size:22px;color:#58a6ff;margin-bottom:4px}}
  .header p{{color:#8b949e;font-size:13px}}
  .container{{max-width:1100px;margin:0 auto;padding:32px 48px}}
  .summary{{display:flex;gap:24px;flex-wrap:wrap;background:#161b22;border:1px solid #30363d;border-radius:10px;padding:24px;margin-bottom:32px}}
  .stat{{text-align:center;flex:1;min-width:110px}}
  .stat .val{{font-size:24px;font-weight:700;color:#f0f6fc}}
  .stat .lbl{{font-size:11px;color:#8b949e;margin-top:4px}}
  .ep-card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:24px;margin-bottom:28px}}
  .ep-card h2{{font-size:16px;color:#f0f6fc;margin-bottom:8px;display:flex;align-items:center;gap:8px;flex-wrap:wrap}}
  .badge{{padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600;background:rgba(88,166,255,.15);color:#58a6ff;border:1px solid rgba(88,166,255,.3)}}
  .badge.green{{background:rgba(63,185,80,.15);color:#3fb950;border-color:rgba(63,185,80,.3)}}
  .badge.purple{{background:rgba(188,140,255,.15);color:#bc8cff;border-color:rgba(188,140,255,.3)}}
  .task-desc{{color:#d2a8ff;font-size:13px;margin:4px 0}}
  .meta{{color:#8b949e;font-size:12px;margin:4px 0 12px}}
  table{{width:100%;border-collapse:collapse;font-size:13px;margin-top:14px}}
  thead th{{background:#21262d;color:#8b949e;padding:8px 12px;text-align:left;border:1px solid #30363d}}
  tbody td{{padding:7px 12px;border:1px solid #21262d}}
  tbody tr:hover td{{background:rgba(255,255,255,.02)}}
  hr{{border-color:#21262d;margin:28px 0}}
  h3{{color:#cdd9e5;margin-bottom:14px}}
  .global-table{{max-width:420px}}
</style></head><body>
<div class="header">
  <h1>π₀.₅ × LIBERO — 离线推理评测报告</h1>
  <p>Model: {MODEL_ID} &nbsp;|&nbsp; Dataset: {DATASET_ID} &nbsp;|&nbsp; Episodes: {len(all_ep_results)}</p>
</div>
<div class="container">
  <div class="summary">
    <div class="stat"><div class="val">{len(all_ep_results)}</div><div class="lbl">Episodes</div></div>
    <div class="stat"><div class="val">{sum(ep["n_frames"] for ep in all_ep_results)}</div><div class="lbl">Total Frames</div></div>
    <div class="stat"><div class="val">{np.mean(all_maes):.4f}</div><div class="lbl">Avg Mean MAE</div></div>
    <div class="stat"><div class="val">{np.mean(all_rmses):.4f}</div><div class="lbl">Avg Mean RMSE</div></div>
    <div class="stat"><div class="val">{np.mean(all_cos):.4f}</div><div class="lbl">Avg Cosine Sim</div></div>
  </div>
  <h3>各关节全局平均 MAE ({len(all_ep_results)} episodes)</h3>
  <table class="global-table">
    <thead><tr><th>Joint</th><th>Global Mean MAE</th></tr></thead>
    <tbody>{gj_rows}</tbody>
  </table>
  <hr><h3>Episode 详情</h3>
  {"".join(episode_blocks)}
</div></body></html>"""
    with open(output_path, "w") as f:
        f.write(html)
    log(f"HTML report → {output_path}")


# ── 主流程 ─────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    policy, preprocessor, postprocessor = load_model_and_processors()
    dataset, meta, task_map = load_dataset(N_EPISODES)

    log("Grouping frames by episode ...")
    ep_frames = defaultdict(list)
    for idx in range(len(dataset)):
        s = dataset[idx]
        ep_frames[int(s["episode_index"].item())].append(s)
    for ep_idx in ep_frames:
        ep_frames[ep_idx].sort(key=lambda s: s["frame_index"].item())

    all_ep_results = []
    for ep_idx in sorted(ep_frames.keys())[:N_EPISODES]:
        frames   = ep_frames[ep_idx]
        task_str = task_map.get(int(frames[0]["task_index"].item()), "unknown")

        log(f"\n--- Episode {ep_idx} ({len(frames)} frames) ---")
        log(f"    Task: {task_str[:80]}")
        t0 = time.time()
        try:
            pred, gt = infer_episode(policy, preprocessor, postprocessor, frames, task_str)
            elapsed  = time.time() - t0
            log(f"    Done: {elapsed:.1f}s  ({len(frames)/elapsed:.1f} fps)")
            metrics  = compute_metrics(pred, gt)
            log(f"    MAE={metrics['mean_mae']:.4f}  RMSE={metrics['mean_rmse']:.4f}  CosSim={metrics['cosine_sim']:.4f}")
            all_ep_results.append({"episode_idx": ep_idx, "task": task_str, "n_frames": len(frames),
                                   "pred_actions": pred.tolist(), "gt_actions": gt.tolist(), "metrics": metrics})
        except Exception as e:
            import traceback; log(f"    ERROR: {e}"); traceback.print_exc()

    if not all_ep_results:
        log("No results."); sys.exit(1)

    # JSON
    json_path = OUTPUT_DIR / "results.json"
    with open(json_path, "w") as f:
        json.dump([{k: v for k, v in ep.items() if k not in ("pred_actions","gt_actions")}
                   for ep in all_ep_results], f, indent=2)
    log(f"JSON → {json_path}")

    # CSV
    import csv
    csv_path = OUTPUT_DIR / "predictions.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode_idx","frame_idx","task"] +
                   [f"pred_{n}" for n in ACTION_NAMES] + [f"gt_{n}" for n in ACTION_NAMES])
        for ep in all_ep_results:
            p, g = np.array(ep["pred_actions"]), np.array(ep["gt_actions"])
            for t in range(len(p)):
                w.writerow([ep["episode_idx"], t, ep["task"]] + p[t].tolist() + g[t].tolist())
    log(f"CSV → {csv_path}")

    generate_html_report(all_ep_results, OUTPUT_DIR / "report.html")

    log("\n" + "="*60)
    log("Output files:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        log(f"  {f.name:30s}  {f.stat().st_size/1024:8.1f} KB")
    log("="*60)
    log(f"Download: scp -P 38960 -r root@connect.nmb1.seetacloud.com:{OUTPUT_DIR} ./pi05_results")

if __name__ == "__main__":
    main()
