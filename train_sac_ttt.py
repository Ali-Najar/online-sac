from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from checkpoint import load_checkpoint, save_checkpoint
from envs import HalfCheetahVelEnv, make_vector_env
from model import build_modules
from normalization import RunningMeanStd
from replay import SegmentReplayBuffer
from rollout import collect_online_window, make_token, sample_velocity_schedule
from sac import sac_update


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--num-updates", type=int, default=1000)

    # Online stream.
    p.add_argument("--mdps-per-update", type=int, default=1)
    p.add_argument("--ttt-reset-interval", type=int, default=1)
    p.add_argument("--rollout-steps", type=int, default=50)
    p.add_argument(
        "--no-reset-on-mdp-change",
        action="store_true",
        help=(
            "Do not reset the environment when the target velocity changes. "
            "The MuJoCo state continues; only target_velocity changes at MDP boundaries."
        ),
    )

    # Velocity schedule. Default matches LILAC Half-Cheetah Vel.
    p.add_argument("--velocity-schedule", type=str, default="lilac_sine", choices=["lilac_sine", "uniform"])
    p.add_argument("--lilac-velocity-base", type=float, default=1.5)
    p.add_argument("--lilac-velocity-amplitude", type=float, default=1.5)
    p.add_argument("--lilac-velocity-frequency", type=float, default=0.5)
    p.add_argument("--train-vel-min", type=float, default=0.0)
    p.add_argument("--train-vel-max", type=float, default=3.0)
    p.add_argument("--eval-velocities", type=float, nargs="*", default=[0.25, 0.75, 1.25, 1.75, 2.25, 2.75])
    p.add_argument("--ctrl-cost-weight", type=float, default=0.05)
    p.add_argument(
        "--oracle",
        action="store_true",
        help="Append the current target velocity to the observation, like the LILAC oracle baseline.",
    )

    # TTT.
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-hidden-layers", type=int, default=2)
    p.add_argument("--num-attention-heads", type=int, default=4)
    p.add_argument("--ttt-layer-type", type=str, default="mlp", choices=["linear", "mlp"])
    p.add_argument("--mini-batch-size", type=int, default=4)
    p.add_argument(
        "--zero-ttt-output",
        action="store_true",
        help="Ablation: replace TTT context z with zeros in both rollout and SAC updates.",
    )

    # SAC.
    p.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 256])
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--reward-scale", type=float, default=1.0)

    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--context-lr", type=float, default=3e-4)

    p.add_argument("--replay-size", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--random-steps", type=int, default=5000)
    p.add_argument("--updates-per-online-window", type=int, default=50)

    # TTT forecasting auxiliary objective.
    p.add_argument("--forecast-horizon", type=int, default=5)
    p.add_argument("--forecast-obs-coef", type=float, default=1.0)
    p.add_argument("--forecast-reward-coef", type=float, default=1.0)

    # Logging.
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--save-interval", type=int, default=50)
    p.add_argument(
        "--video-interval",
        type=int,
        default=0,
        help="If > 0, save one actual training trajectory video every N updates.",
    )
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--video-dir", type=str, default="videos")
    p.add_argument("--out-dir", type=str, default="runs/ttt_sac_hc_vel")
    p.add_argument("--run-index", type=int, default=None)
    p.add_argument("--no-run-index", action="store_true")
    p.add_argument("--checkpoint", type=str, default="")

    return p.parse_args()


def validate_args(args):
    if args.ttt_reset_interval <= 0:
        raise ValueError("--ttt-reset-interval must be positive")
    if args.mdps_per_update <= 0:
        raise ValueError("--mdps-per-update must be positive")
    if args.ttt_reset_interval > args.mdps_per_update:
        raise ValueError("--ttt-reset-interval should be <= --mdps-per-update")
    if args.rollout_steps <= args.forecast_horizon:
        raise ValueError("--rollout-steps must be larger than --forecast-horizon")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_rows(path: str, rows: list[dict], fieldnames: list[str]):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def make_indexed_run_dir(base_dir: str, run_index: int | None):
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    if run_index is None:
        existing = []
        for child in base.iterdir():
            if child.is_dir() and child.name.startswith("run_"):
                suffix = child.name.replace("run_", "", 1)
                if suffix.isdigit():
                    existing.append(int(suffix))
        run_index = max(existing) + 1 if existing else 0

    run_dir = base / f"run_{run_index:04d}"
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    return str(run_dir)


def save_config(args):
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)


def video_target_velocity(args, update: int) -> float:
    """Use a deterministic representative velocity for the saved video."""
    if args.velocity_schedule == "lilac_sine":
        episode_index = (update - 1) * args.mdps_per_update
        return float(
            args.lilac_velocity_base
            + args.lilac_velocity_amplitude * np.sin(args.lilac_velocity_frequency * episode_index)
        )

    # For the uniform schedule, use the center of the training range so videos
    # are comparable across updates.
    return float(0.5 * (args.train_vel_min + args.train_vel_max))


@torch.no_grad()
def save_single_trajectory_video(modules, args, obs_rms, device, update: int) -> None:
    """Save one deterministic rollout video with the current policy.

    This does not update obs_rms or replay. It is just for visualization.
    """
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise ImportError(
            "Video saving needs imageio. Install with: pip install imageio imageio-ffmpeg"
        ) from exc

    video_dir = os.path.join(args.out_dir, args.video_dir)
    os.makedirs(video_dir, exist_ok=True)

    env = HalfCheetahVelEnv(
        seed=args.seed + 1_000_000 + update,
        train_vel_range=(args.train_vel_min, args.train_vel_max),
        eval_velocities=args.eval_velocities,
        max_episode_steps=args.rollout_steps,
        ctrl_cost_weight=args.ctrl_cost_weight,
        oracle=args.oracle,
        render_mode="rgb_array",
    )

    target_velocity = video_target_velocity(args, update)
    env.set_target_velocity(target_velocity)

    raw_obs, _info = env.reset()
    frames = []
    frame = env.render()
    if frame is not None:
        frames.append(frame)

    obs = obs_rms.normalize(raw_obs[None])[0]
    action_dim = env.action_space.shape[0]
    prev_action = np.zeros((1, action_dim), dtype=np.float32)
    prev_reward = np.zeros((1, 1), dtype=np.float32)
    prev_done = np.ones((1, 1), dtype=np.float32)
    cache_params = None

    episode_return = 0.0
    velocity_errors = []

    modules.context.eval()
    modules.actor.eval()

    video_steps = args.rollout_steps
    if args.no_reset_on_mdp_change:
        video_steps = args.rollout_steps * args.mdps_per_update

    for _step in range(video_steps):
        if args.no_reset_on_mdp_change and _step % args.rollout_steps == 0:
            mdp_offset = _step // args.rollout_steps
            if args.velocity_schedule == "lilac_sine":
                episode_index = (update - 1) * args.mdps_per_update + mdp_offset
                target_velocity = float(
                    args.lilac_velocity_base
                    + args.lilac_velocity_amplitude
                    * np.sin(args.lilac_velocity_frequency * episode_index)
                )
            else:
                target_velocity = float(0.5 * (args.train_vel_min + args.train_vel_max))
            env.set_target_velocity(target_velocity)
            if args.oracle:
                raw_obs[-1] = target_velocity
                obs = obs_rms.normalize(raw_obs[None])[0]

        token = make_token(obs[None], prev_action, prev_reward, prev_done)
        token_t = torch.tensor(token, dtype=torch.float32, device=device)
        obs_t = torch.tensor(obs[None], dtype=torch.float32, device=device)

        z_t, cache_params = modules.context.act_step(token_t, cache_params)
        action = modules.actor.act(obs_t, z_t, deterministic=True)
        action_np = action.cpu().numpy()[0].astype(np.float32)

        next_raw_obs, reward, terminated, truncated, info = env.step(action_np)

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        done = bool(terminated or truncated)
        episode_return += float(reward)
        velocity_errors.append(float(info.get("velocity_error", np.nan)))

        obs = obs_rms.normalize(next_raw_obs[None])[0]
        prev_action = action_np[None]
        prev_reward = np.array([[float(reward)]], dtype=np.float32)
        prev_done = np.array([[float(done)]], dtype=np.float32)

        if done:
            break

    env.close()

    modules.context.train()
    modules.actor.train()

    video_path = os.path.join(video_dir, f"trajectory_update_{update:06d}.mp4")
    imageio.mimsave(video_path, frames, fps=args.video_fps)

    mean_error = float(np.nanmean(velocity_errors)) if velocity_errors else float("nan")
    print(
        f"Saved video to {video_path} | "
        f"target_velocity={target_velocity:.3f} "
        f"return={episode_return:.2f} "
        f"mean_velerr={mean_error:.3f}"
    )


def main():
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    if args.checkpoint or args.no_run_index:
        os.makedirs(args.out_dir, exist_ok=True)
    else:
        args.out_dir = make_indexed_run_dir(args.out_dir, args.run_index)
    save_config(args)
    print(f"Writing run to: {args.out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    envs = make_vector_env(args, args.num_envs)

    obs_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]
    token_dim = obs_dim + action_dim + 2

    print(
        f"obs_dim={obs_dim}, action_dim={action_dim}, token_dim={token_dim}, "
        f"oracle={args.oracle}, zero_ttt_output={args.zero_ttt_output}, "
        f"no_reset_on_mdp_change={args.no_reset_on_mdp_change}"
    )


    obs_rms = RunningMeanStd(shape=(obs_dim,))
    modules = build_modules(args, obs_dim=obs_dim, action_dim=action_dim, token_dim=token_dim, device=device)

    optimizers = {
        "actor": optim.Adam(modules.actor.parameters(), lr=args.actor_lr),
        "critic": optim.Adam(list(modules.q1.parameters()) + list(modules.q2.parameters()), lr=args.critic_lr),
        "context": optim.Adam(list(modules.context.parameters()) + list(modules.forecaster.parameters()), lr=args.context_lr),
    }

    replay = SegmentReplayBuffer(capacity_segments=args.replay_size)

    start_update = 0
    total_env_steps = 0
    if args.checkpoint:
        start_update, total_env_steps = load_checkpoint(args.checkpoint, modules, obs_rms, device)
        print(f"Loaded checkpoint {args.checkpoint} at update {start_update}, env_steps={total_env_steps}")

    summary_path = os.path.join(args.out_dir, "online_summary.csv")
    mdp_path = os.path.join(args.out_dir, "online_mdp_rows.csv")
    step_path = os.path.join(args.out_dir, "online_step_rows.csv")

    summary_fields = [
        "update",
        "env_steps",
        "return_mean",
        "return_final_mdp_mean",
        "velocity_error_mean",
        "velocity_error_final_mdp_mean",
        "critic_loss",
        "actor_loss",
        "forecast_loss",
        "forecast_obs_loss",
        "forecast_reward_loss",
        "rollout_time_sec",
        "train_time_sec",
        "replay_segments",
    ]
    mdp_fields = [
        "update",
        "env",
        "mdp_index",
        "target_velocity",
        "ttt_reset_interval",
        "ttt_restarted",
        "return",
        "velocity_error",
    ]
    step_fields = [
        "update",
        "env",
        "mdp_index",
        "step",
        "global_mdp_step",
        "global_env_step",
        "target_velocity",
        "ttt_reset_interval",
        "ttt_restarted",
        "raw_reward",
        "velocity_error",
        "done",
    ]

    last_stats = {
        "critic_loss": np.nan,
        "actor_loss": np.nan,
        "forecast_loss": np.nan,
        "forecast_obs_loss": np.nan,
        "forecast_reward_loss": np.nan,
    }

    for update in range(start_update + 1, args.num_updates + 1):
        velocity_schedule = sample_velocity_schedule(
            args=args,
            num_envs=args.num_envs,
            num_mdps=args.mdps_per_update,
            global_episode_start=(update - 1) * args.mdps_per_update,
        )

        rollout_t0 = time.perf_counter()
        window = collect_online_window(
            modules=modules,
            envs=envs,
            obs_rms=obs_rms,
            args=args,
            device=device,
            update_index=update,
            velocity_schedule=velocity_schedule,
            total_env_steps=total_env_steps,
        )
        rollout_time_sec = time.perf_counter() - rollout_t0

        # Add one segment per vector-env member.
        for env_i in range(args.num_envs):
            replay.add_segment(
                obs=window.obs[env_i],
                tokens=window.tokens[env_i],
                actions=window.actions[env_i],
                rewards=window.rewards[env_i],
                dones=window.dones[env_i],
            )

        total_env_steps += args.num_envs * args.mdps_per_update * args.rollout_steps

        train_t0 = time.perf_counter()
        stats_accum = []
        if len(replay) >= args.batch_size:
            for _ in range(args.updates_per_online_window):
                batch = replay.sample(args.batch_size, device=device)
                stats_accum.append(sac_update(modules, optimizers, batch, args))

        if stats_accum:
            last_stats = {
                key: float(np.mean([s[key] for s in stats_accum]))
                for key in stats_accum[0].keys()
            }
        train_time_sec = time.perf_counter() - train_t0

        return_mean = float(window.mdp_returns.mean())
        return_final = float(window.mdp_returns[:, -1].mean())
        error_mean = float(window.mdp_velocity_error.mean())
        error_final = float(window.mdp_velocity_error[:, -1].mean())

        summary_row = {
            "update": update,
            "env_steps": total_env_steps,
            "return_mean": return_mean,
            "return_final_mdp_mean": return_final,
            "velocity_error_mean": error_mean,
            "velocity_error_final_mdp_mean": error_final,
            **last_stats,
            "rollout_time_sec": rollout_time_sec,
            "train_time_sec": train_time_sec,
            "replay_segments": len(replay),
        }

        append_rows(summary_path, [summary_row], summary_fields)
        append_rows(mdp_path, window.mdp_rows, mdp_fields)
        append_rows(step_path, window.step_rows, step_fields)

        if update % args.log_interval == 0:
            print(
                f"update={update:06d} steps={total_env_steps:09d} | "
                f"return={return_mean:8.2f} final={return_final:8.2f} "
                f"velerr={error_mean:.3f} | "
                f"critic={last_stats['critic_loss']:.3f} "
                f"actor={last_stats['actor_loss']:.3f} "
                f"forecast={last_stats['forecast_loss']:.3f} | "
                f"rollout={rollout_time_sec:.2f}s train={train_time_sec:.2f}s "
                f"replay={len(replay)}"
            )

        if args.save_interval > 0 and update % args.save_interval == 0:
            ckpt_path = os.path.join(args.out_dir, f"checkpoint_{update:06d}.pt")
            save_checkpoint(ckpt_path, modules, obs_rms, args, update, total_env_steps)
            print(f"Saved checkpoint to: {ckpt_path}")

        # Training trajectory videos are saved inside collect_online_window(...)
        # so they capture the actual env-0 training trajectory from this update.

    final_path = os.path.join(args.out_dir, "final.pt")
    save_checkpoint(final_path, modules, obs_rms, args, args.num_updates, total_env_steps)

    print(f"Saved final checkpoint to: {final_path}")
    print(f"Saved online summary to: {summary_path}")
    print(f"Saved per-MDP online rows to: {mdp_path}")
    print(f"Saved per-step online rows to: {step_path}")

    envs.close()


if __name__ == "__main__":
    main()
