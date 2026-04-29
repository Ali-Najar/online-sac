from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import os
import gymnasium as gym
import numpy as np
import torch

from envs import set_vector_target_velocities
from normalization import RunningMeanStd


@dataclass
class OnlineWindow:
    obs: np.ndarray          # (N, M, T+1, obs_dim), normalized
    tokens: np.ndarray       # (N, M, T, token_dim)
    actions: np.ndarray      # (N, M, T, action_dim)
    rewards: np.ndarray      # (N, M, T)
    dones: np.ndarray        # (N, M, T)
    mdp_returns: np.ndarray  # (N, M)
    mdp_velocity_error: np.ndarray
    mdp_rows: List[dict]
    step_rows: List[dict]


def sample_velocity_schedule(args, num_envs: int, num_mdps: int, global_episode_start: int = 0) -> np.ndarray:
    if args.velocity_schedule == "uniform":
        return np.random.uniform(
            low=args.train_vel_min,
            high=args.train_vel_max,
            size=(num_envs, num_mdps),
        ).astype(np.float32)

    if args.velocity_schedule == "lilac_sine":
        episode_indices = global_episode_start + np.arange(num_mdps, dtype=np.float32)
        velocities = (
            args.lilac_velocity_base
            + args.lilac_velocity_amplitude * np.sin(args.lilac_velocity_frequency * episode_indices)
        ).astype(np.float32)
        return np.tile(velocities[None, :], (num_envs, 1)).astype(np.float32)

    raise ValueError(f"Unknown velocity_schedule: {args.velocity_schedule}")


def make_token(obs: np.ndarray, prev_action: np.ndarray, prev_reward: np.ndarray, prev_done: np.ndarray) -> np.ndarray:
    return np.concatenate([obs, prev_action, prev_reward, prev_done], axis=-1).astype(np.float32)


def _patch_oracle_target_in_obs(obs: np.ndarray, velocities: np.ndarray, oracle: bool) -> np.ndarray:
    """If oracle obs is enabled, replace the appended target velocity in-place.

    Needed when the MDP target changes without env.reset(), because the last
    observation came from the old target.
    """
    if oracle:
        obs = obs.copy()
        obs[:, -1] = velocities.astype(np.float32)
    return obs


def _get_or_init_continuous_state(envs: gym.vector.SyncVectorEnv, obs_rms: RunningMeanStd, args):
    """State carried across collect_online_window calls for non-reset mode."""
    if not hasattr(envs, "_continuous_raw_obs"):
        raw_obs, _info = envs.reset()
        obs_rms.update(raw_obs)
        action_dim = envs.single_action_space.shape[0]
        envs._continuous_raw_obs = raw_obs.astype(np.float32)
        envs._continuous_prev_action = np.zeros((envs.num_envs, action_dim), dtype=np.float32)
        envs._continuous_prev_reward = np.zeros((envs.num_envs, 1), dtype=np.float32)
        envs._continuous_prev_done = np.ones((envs.num_envs, 1), dtype=np.float32)
    return (
        envs._continuous_raw_obs,
        envs._continuous_prev_action,
        envs._continuous_prev_reward,
        envs._continuous_prev_done,
    )


def _save_continuous_state(
    envs: gym.vector.SyncVectorEnv,
    raw_obs: np.ndarray,
    prev_action: np.ndarray,
    prev_reward: np.ndarray,
    prev_done: np.ndarray,
) -> None:
    envs._continuous_raw_obs = raw_obs.astype(np.float32)
    envs._continuous_prev_action = prev_action.astype(np.float32)
    envs._continuous_prev_reward = prev_reward.astype(np.float32)
    envs._continuous_prev_done = prev_done.astype(np.float32)


def _maybe_append_training_frame(envs: gym.vector.SyncVectorEnv, frames: list, capture_video: bool) -> None:
    """Capture env 0 from the actual training vector env."""
    if not capture_video:
        return

    frame = envs.envs[0].render()
    if frame is not None:
        frames.append(frame)


def _save_training_video(frames: list, args, update_index: int) -> None:
    if not frames:
        return

    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise ImportError(
            "Video saving needs imageio. Install with: pip install imageio imageio-ffmpeg"
        ) from exc

    video_dir = os.path.join(args.out_dir, args.video_dir)
    os.makedirs(video_dir, exist_ok=True)

    video_path = os.path.join(video_dir, f"training_update_{update_index:06d}.mp4")
    imageio.mimsave(video_path, frames, fps=args.video_fps)
    print(f"Saved training trajectory video to: {video_path}")


@torch.no_grad()
def collect_online_window(
    modules,
    envs: gym.vector.SyncVectorEnv,
    obs_rms: RunningMeanStd,
    args,
    device: torch.device,
    update_index: int,
    velocity_schedule: np.ndarray,
    total_env_steps: int,
) -> OnlineWindow:
    num_envs = envs.num_envs
    num_mdps = args.mdps_per_update
    ep_len = args.rollout_steps
    obs_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]
    token_dim = obs_dim + action_dim + 2

    obs_buf = np.zeros((num_envs, num_mdps, ep_len + 1, obs_dim), dtype=np.float32)
    token_buf = np.zeros((num_envs, num_mdps, ep_len, token_dim), dtype=np.float32)
    action_buf = np.zeros((num_envs, num_mdps, ep_len, action_dim), dtype=np.float32)
    reward_buf = np.zeros((num_envs, num_mdps, ep_len), dtype=np.float32)
    done_buf = np.zeros((num_envs, num_mdps, ep_len), dtype=np.float32)

    mdp_returns = np.zeros((num_envs, num_mdps), dtype=np.float32)
    mdp_velocity_error = np.zeros((num_envs, num_mdps), dtype=np.float32)

    mdp_rows: List[dict] = []
    step_rows: List[dict] = []

    cache_params = None

    capture_video = args.video_interval > 0 and update_index % args.video_interval == 0
    video_frames: list = []

    for mdp_idx in range(num_mdps):
        ttt_restarted = mdp_idx % args.ttt_reset_interval == 0
        if ttt_restarted:
            cache_params = None

        current_velocities = velocity_schedule[:, mdp_idx]
        set_vector_target_velocities(envs, current_velocities)

        if args.no_reset_on_mdp_change:
            # Continuous mode: do not reset the MuJoCo state at MDP boundaries.
            # Only the reward target changes. Keep previous action/reward/done.
            raw_obs, prev_action, prev_reward, prev_done = _get_or_init_continuous_state(
                envs, obs_rms, args
            )
            raw_obs = _patch_oracle_target_in_obs(raw_obs, current_velocities, args.oracle)
            # Include the current state under the new target in obs normalization.
            obs_rms.update(raw_obs)
        else:
            # Episodic mode: LILAC-style episode boundary reset.
            raw_obs, _info = envs.reset()
            obs_rms.update(raw_obs)
            prev_action = np.zeros((num_envs, action_dim), dtype=np.float32)
            prev_reward = np.zeros((num_envs, 1), dtype=np.float32)
            prev_done = np.ones((num_envs, 1), dtype=np.float32)

        obs = obs_rms.normalize(raw_obs)
        obs_buf[:, mdp_idx, 0] = obs

        # This frame belongs to the actual training trajectory, not a separate eval env.
        # In episodic mode, this captures the post-reset initial state for each MDP.
        # In no-reset mode, this captures the continuing state after changing target velocity.
        _maybe_append_training_frame(envs, video_frames, capture_video)

        velocity_error_sum = np.zeros(num_envs, dtype=np.float32)

        for step in range(ep_len):
            token = make_token(obs, prev_action, prev_reward, prev_done)
            token_t = torch.tensor(token, dtype=torch.float32, device=device)
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

            z_t, cache_params = modules.context.act_step(token_t, cache_params)

            if total_env_steps < args.random_steps:
                action_np = np.stack([envs.single_action_space.sample() for _ in range(num_envs)], axis=0).astype(np.float32)
            else:
                action = modules.actor.act(obs_t, z_t, deterministic=False)
                action_np = action.cpu().numpy().astype(np.float32)

            next_raw_obs, reward, terminated, truncated, info = envs.step(action_np)
            _maybe_append_training_frame(envs, video_frames, capture_video)
            done = np.logical_or(terminated, truncated)
            velocity_error = np.asarray(info.get("velocity_error", np.zeros(num_envs)), dtype=np.float32)

            obs_rms.update(next_raw_obs)
            next_obs = obs_rms.normalize(next_raw_obs)

            token_buf[:, mdp_idx, step] = token
            action_buf[:, mdp_idx, step] = action_np
            reward_buf[:, mdp_idx, step] = reward.astype(np.float32)
            done_buf[:, mdp_idx, step] = done.astype(np.float32)
            obs_buf[:, mdp_idx, step + 1] = next_obs

            mdp_returns[:, mdp_idx] += reward.astype(np.float32)
            velocity_error_sum += velocity_error

            for env_i in range(num_envs):
                global_env_step = total_env_steps + step + 1
                step_rows.append(
                    {
                        "update": update_index,
                        "env": env_i,
                        "mdp_index": mdp_idx + 1,
                        "step": step + 1,
                        "global_mdp_step": (update_index - 1) * num_mdps + mdp_idx + 1,
                        "global_env_step": global_env_step,
                        "target_velocity": float(current_velocities[env_i]),
                        "ttt_reset_interval": int(args.ttt_reset_interval),
                        "ttt_restarted": int(ttt_restarted),
                        "raw_reward": float(reward[env_i]),
                        "velocity_error": float(velocity_error[env_i]),
                        "done": int(done[env_i]),
                    }
                )

            prev_action = action_np
            prev_reward = reward[:, None].astype(np.float32)
            prev_done = done[:, None].astype(np.float32)
            raw_obs = next_raw_obs.astype(np.float32)
            obs = next_obs

            if args.no_reset_on_mdp_change:
                _save_continuous_state(envs, raw_obs, prev_action, prev_reward, prev_done)

        mdp_velocity_error[:, mdp_idx] = velocity_error_sum / float(ep_len)

        for env_i in range(num_envs):
            mdp_rows.append(
                {
                    "update": update_index,
                    "env": env_i,
                    "mdp_index": mdp_idx + 1,
                    "target_velocity": float(current_velocities[env_i]),
                    "ttt_reset_interval": int(args.ttt_reset_interval),
                    "ttt_restarted": int(ttt_restarted),
                    "return": float(mdp_returns[env_i, mdp_idx]),
                    "velocity_error": float(mdp_velocity_error[env_i, mdp_idx]),
                }
            )

        total_env_steps += num_envs * ep_len

    _save_training_video(video_frames, args, update_index)

    return OnlineWindow(
        obs=obs_buf,
        tokens=token_buf,
        actions=action_buf,
        rewards=reward_buf,
        dones=done_buf,
        mdp_returns=mdp_returns,
        mdp_velocity_error=mdp_velocity_error,
        mdp_rows=mdp_rows,
        step_rows=step_rows,
    )
