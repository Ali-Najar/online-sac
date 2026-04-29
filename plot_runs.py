import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def smooth(s, window):
    if window <= 1:
        return s
    return s.rolling(window, min_periods=1).mean()


def find_runs(base_dir):
    base = Path(base_dir)
    if (base / "online_summary.csv").exists():
        return [base]
    return sorted([p for p in base.iterdir() if p.is_dir() and (p / "online_summary.csv").exists()])


def label(run_dir):
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return run_dir.name
    with open(cfg_path) as f:
        cfg = json.load(f)
    return f"{run_dir.name} reset={cfg.get('ttt_reset_interval')}/{cfg.get('mdps_per_update')} seed={cfg.get('seed')}"


def plot_metric(runs, metric, ylabel, out_path, smooth_window):
    plt.figure()
    for r in runs:
        df = pd.read_csv(r / "online_summary.csv")
        x = df["env_steps"] if "env_steps" in df.columns else df["update"]
        if metric in df.columns:
            plt.plot(x, smooth(df[metric], smooth_window), label=label(r))
    plt.xlabel("Env steps")
    plt.ylabel(ylabel)
    plt.title(metric)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", default="runs/ttt_sac_hc_vel")
    p.add_argument("--smooth-window", type=int, default=10)
    p.add_argument("--out-dir", default="")
    args = p.parse_args()

    runs = find_runs(args.base_dir)
    if not runs:
        raise FileNotFoundError(f"No runs found under {args.base_dir}")

    out_dir = args.out_dir or os.path.join(args.base_dir, "comparison_plots")
    os.makedirs(out_dir, exist_ok=True)

    for metric, ylabel in [
        ("return_mean", "Return"),
        ("velocity_error_mean", "Velocity error"),
        ("critic_loss", "Critic loss"),
        ("actor_loss", "Actor loss"),
        ("forecast_loss", "Forecast loss"),
        ("forecast_obs_loss", "Forecast obs loss"),
        ("forecast_reward_loss", "Forecast reward loss"),
        ("rollout_time_sec", "Seconds"),
        ("train_time_sec", "Seconds"),
    ]:
        plot_metric(runs, metric, ylabel, os.path.join(out_dir, f"{metric}.png"), args.smooth_window)

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
