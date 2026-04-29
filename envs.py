from __future__ import annotations

from typing import Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np


class HalfCheetahVelEnv(gym.Wrapper):
    """HalfCheetah target-velocity MDP.

    MDP identity is target_velocity. The trainer explicitly sets the target
    velocity before every episode rollout.
    """

    def __init__(
        self,
        seed: int = 0,
        train_vel_range: Tuple[float, float] = (0.0, 3.0),
        eval_velocities: Sequence[float] = (0.25, 0.75, 1.25, 1.75, 2.25, 2.75),
        max_episode_steps: int = 50,
        ctrl_cost_weight: float = 0.05,
        oracle: bool = False,
        render_mode: Optional[str] = None,
    ):
        env = self._make_halfcheetah(max_episode_steps=max_episode_steps, render_mode=render_mode)
        super().__init__(env)

        self.rng = np.random.default_rng(seed)
        self.train_vel_range = tuple(float(x) for x in train_vel_range)
        self.eval_velocities = tuple(float(v) for v in eval_velocities)
        self.ctrl_cost_weight = float(ctrl_cost_weight)
        self.oracle = bool(oracle)

        if self.oracle:
            low = np.asarray(self.observation_space.low, dtype=np.float32)
            high = np.asarray(self.observation_space.high, dtype=np.float32)
            self.observation_space = gym.spaces.Box(
                low=np.concatenate([low, np.array([-np.inf], dtype=np.float32)]),
                high=np.concatenate([high, np.array([np.inf], dtype=np.float32)]),
                dtype=np.float32,
            )

        self.fixed_target_velocity: Optional[float] = None
        self.target_velocity = 0.0

    @staticmethod
    def _make_halfcheetah(max_episode_steps: int, render_mode: Optional[str] = None) -> gym.Env:
        last_error = None
        for env_id in ["HalfCheetah-v5", "HalfCheetah-v4", "HalfCheetah-v3"]:
            try:
                return gym.make(env_id, max_episode_steps=max_episode_steps, render_mode=render_mode)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise RuntimeError(
            "Could not create HalfCheetah. Install gymnasium[mujoco]. "
            f"Last error: {last_error}"
        )

    def set_target_velocity(self, velocity: Optional[float]) -> None:
        self.fixed_target_velocity = None if velocity is None else float(velocity)
        # Important for continuous/non-reset mode: reward uses self.target_velocity
        # immediately, without waiting for reset().
        if velocity is not None:
            self.target_velocity = float(velocity)

    def sample_target_velocity(self) -> float:
        if self.fixed_target_velocity is not None:
            return float(self.fixed_target_velocity)

        lo, hi = self.train_vel_range
        return float(self.rng.uniform(lo, hi))

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.astype(np.float32)
        if not self.oracle:
            return obs
        return np.concatenate([obs, np.array([self.target_velocity], dtype=np.float32)], axis=0)

    def reset(self, **kwargs):
        self.target_velocity = self.sample_target_velocity()
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        info["target_velocity"] = self.target_velocity
        return self._augment_obs(obs), info

    def step(self, action):
        obs, _base_reward, terminated, truncated, info = self.env.step(action)

        if "x_velocity" not in info:
            raise RuntimeError(
                "HalfCheetah info does not contain x_velocity. "
                "Use a Gymnasium MuJoCo HalfCheetah version that reports x_velocity."
            )

        x_velocity = float(info["x_velocity"])
        # ctrl_cost = self.ctrl_cost_weight * float(np.square(action).sum())
        # velocity_error = abs(x_velocity - self.target_velocity)

        # # LILAC writes -||v_s-v_g||^2 - 0.05||a||^2. Use squared error by default.
        # reward = -(velocity_error ** 2) - ctrl_cost

        velocity_error = abs(x_velocity - self.target_velocity)
        ctrl_cost = self.ctrl_cost_weight * float(np.linalg.norm(action, ord=2))

        # LILAC Half-Cheetah Vel:
        # r(s, a) = -||v_s - v_g||_2 - 0.05 * ||a||_2
        # Since v_s and v_g are scalars, ||v_s - v_g||_2 = abs(v_s - v_g).
        reward = -velocity_error - ctrl_cost

        info = dict(info)
        info["target_velocity"] = self.target_velocity
        info["velocity_error"] = velocity_error
        info["hcvel_ctrl_cost"] = ctrl_cost
        return self._augment_obs(obs), float(reward), terminated, truncated, info


def make_env_fn(args, index: int):
    def thunk():
        return HalfCheetahVelEnv(
            seed=args.seed + index,
            train_vel_range=(args.train_vel_min, args.train_vel_max),
            eval_velocities=args.eval_velocities,
            max_episode_steps=(
                args.rollout_steps
                if not args.no_reset_on_mdp_change
                else args.rollout_steps * args.mdps_per_update * args.num_updates + 1
            ),
            ctrl_cost_weight=args.ctrl_cost_weight,
            oracle=args.oracle,
            render_mode=("rgb_array" if args.video_interval > 0 else None),
        )

    return thunk


def make_vector_env(args, num_envs: int) -> gym.vector.SyncVectorEnv:
    return gym.vector.SyncVectorEnv([make_env_fn(args, i) for i in range(num_envs)])


def set_vector_target_velocities(envs: gym.vector.SyncVectorEnv, velocities) -> np.ndarray:
    if np.isscalar(velocities):
        velocity_vector = np.full(envs.num_envs, float(velocities), dtype=np.float32)
    else:
        velocity_vector = np.asarray(velocities, dtype=np.float32)
        if velocity_vector.shape != (envs.num_envs,):
            raise ValueError(f"Expected velocity shape {(envs.num_envs,)}, got {velocity_vector.shape}")

    for env, velocity in zip(envs.envs, velocity_vector):
        env.set_target_velocity(float(velocity))

    return velocity_vector
