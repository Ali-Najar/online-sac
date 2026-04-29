from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from model import soft_update
from replay import SegmentBatch


def _shift_next_z(z: torch.Tensor) -> torch.Tensor:
    """z: (B,M,T,H). Return z for next transition token where available.

    At episode ends the target is masked by done, so the exact last value is not
    important; we simply repeat the last z.
    """
    b, m, t, h = z.shape
    flat = z.reshape(b, m * t, h)
    shifted = torch.cat([flat[:, 1:], flat[:, -1:, :]], dim=1)
    return shifted.reshape(b, m, t, h)


def forecaster_loss(modules, batch: SegmentBatch, z: torch.Tensor, horizon: int, obs_coef: float, reward_coef: float):
    """K-step open-loop prediction loss within each episode."""
    obs = batch.obs
    actions = batch.actions
    rewards = batch.rewards

    b, m, t, obs_dim = obs.shape[0], obs.shape[1], actions.shape[2], obs.shape[-1]
    k = int(horizon)
    if k <= 0 or t < k:
        return z.new_tensor(0.0), {"forecast_obs_loss": 0.0, "forecast_reward_loss": 0.0}

    max_start = t - k + 1

    z0 = z[:, :, :max_start].reshape(-1, z.shape[-1])
    start_obs = obs[:, :, :max_start].reshape(-1, obs_dim)

    action_windows = []
    true_next_obs = []
    true_rewards = []
    for start in range(max_start):
        action_windows.append(actions[:, :, start : start + k])
        true_next_obs.append(obs[:, :, start + 1 : start + k + 1])
        true_rewards.append(rewards[:, :, start : start + k])

    action_seq = torch.stack(action_windows, dim=2).reshape(-1, k, actions.shape[-1])
    target_obs = torch.stack(true_next_obs, dim=2).reshape(-1, k, obs_dim)
    target_rewards = torch.stack(true_rewards, dim=2).reshape(-1, k)

    pred_obs, pred_rewards = modules.forecaster(z0, start_obs, action_seq)

    obs_loss = F.mse_loss(pred_obs, target_obs)
    reward_loss = F.mse_loss(pred_rewards, target_rewards)
    loss = obs_coef * obs_loss + reward_coef * reward_loss
    return loss, {
        "forecast_obs_loss": float(obs_loss.detach().item()),
        "forecast_reward_loss": float(reward_loss.detach().item()),
    }


def sac_update(modules, optimizers: Dict[str, torch.optim.Optimizer], batch: SegmentBatch, args) -> Dict[str, float]:
    """One SAC+TTT update.

    TTT is trained by K-step forecasting.
    SAC actor/critics receive z_t as an additional task/context vector.
    For stability in this simple version, z_t is detached for SAC losses.
    """
    z = modules.context(batch.tokens)

    # Train context encoder + forecaster.
    forecast_loss, forecast_stats = forecaster_loss(
        modules=modules,
        batch=batch,
        z=z,
        horizon=args.forecast_horizon,
        obs_coef=args.forecast_obs_coef,
        reward_coef=args.forecast_reward_coef,
    )
    optimizers["context"].zero_grad(set_to_none=True)
    forecast_loss.backward()
    optimizers["context"].step()

    # Recompute z after context update, then detach for SAC.
    with torch.no_grad():
        z = modules.context(batch.tokens)
    z_next = _shift_next_z(z)

    obs = batch.obs[:, :, :-1]
    next_obs = batch.obs[:, :, 1:]
    actions = batch.actions
    rewards = batch.rewards.unsqueeze(-1)
    dones = batch.dones.unsqueeze(-1)

    flat_obs = obs.reshape(-1, obs.shape[-1])
    flat_next_obs = next_obs.reshape(-1, next_obs.shape[-1])
    flat_actions = actions.reshape(-1, actions.shape[-1])
    flat_rewards = rewards.reshape(-1, 1)
    flat_dones = dones.reshape(-1, 1)
    flat_z = z.reshape(-1, z.shape[-1])
    flat_z_next = z_next.reshape(-1, z_next.shape[-1])

    # Critic update.
    with torch.no_grad():
        next_action, next_logp = modules.actor.sample(flat_next_obs, flat_z_next)
        target_q1 = modules.q1_target(flat_next_obs, next_action, flat_z_next)
        target_q2 = modules.q2_target(flat_next_obs, next_action, flat_z_next)
        target_q = torch.min(target_q1, target_q2) - args.alpha * next_logp
        y = args.reward_scale * flat_rewards + args.gamma * (1.0 - flat_dones) * target_q

    q1 = modules.q1(flat_obs, flat_actions, flat_z)
    q2 = modules.q2(flat_obs, flat_actions, flat_z)
    critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

    optimizers["critic"].zero_grad(set_to_none=True)
    critic_loss.backward()
    optimizers["critic"].step()

    # Actor update.
    new_action, logp = modules.actor.sample(flat_obs, flat_z)
    q1_pi = modules.q1(flat_obs, new_action, flat_z)
    q2_pi = modules.q2(flat_obs, new_action, flat_z)
    q_pi = torch.min(q1_pi, q2_pi)
    actor_loss = (args.alpha * logp - q_pi).mean()

    optimizers["actor"].zero_grad(set_to_none=True)
    actor_loss.backward()
    optimizers["actor"].step()

    soft_update(modules.q1, modules.q1_target, args.tau)
    soft_update(modules.q2, modules.q2_target, args.tau)

    return {
        "critic_loss": float(critic_loss.detach().item()),
        "actor_loss": float(actor_loss.detach().item()),
        "forecast_loss": float(forecast_loss.detach().item()),
        **forecast_stats,
    }
