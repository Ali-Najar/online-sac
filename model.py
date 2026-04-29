from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ttt import TTTCache, TTTConfig, TTTModel


LOG_STD_MIN = -20
LOG_STD_MAX = 2


def mlp(input_dim: int, output_dim: int, hidden_sizes=(256, 256), act=nn.ReLU) -> nn.Sequential:
    layers = []
    d = input_dim
    for h in hidden_sizes:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers += [nn.Linear(d, output_dim)]
    return nn.Sequential(*layers)


class TTTContextEncoder(nn.Module):
    def __init__(self, config: TTTConfig, token_dim: int, zero_ttt_output: bool = False):
        super().__init__()
        self.config = config
        self.hidden_size = int(config.hidden_size)
        self.input_encoder = nn.Linear(token_dim, self.hidden_size)
        self.model = TTTModel(config)
        self.zero_ttt_output = bool(zero_ttt_output)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, M, T, token_dim) -> z: (B, M, T, H)."""
        b, m, t, d = tokens.shape
        x = self.input_encoder(tokens.reshape(b, m * t, d))
        out = self.model(inputs_embeds=x, use_cache=False, return_dict=True)
        z = out.last_hidden_state.reshape(b, m, t, self.hidden_size)
        if self.zero_ttt_output:
            z = torch.zeros_like(z)
        return z

    def act_step(self, token: torch.Tensor, cache_params: Optional[TTTCache]) -> Tuple[torch.Tensor, Optional[TTTCache]]:
        """token: (B, token_dim) -> z_t: (B, H)."""
        x = self.input_encoder(token[:, None, :])
        out = self.model(
            inputs_embeds=x,
            cache_params=cache_params,
            use_cache=True,
            return_dict=True,
        )
        z = out.last_hidden_state[:, -1, :]
        if self.zero_ttt_output:
            z = torch.zeros_like(z)
        return z, out.cache_params


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, z_dim: int, action_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = mlp(obs_dim + z_dim, 2 * action_dim, hidden_sizes)
        self.action_dim = action_dim

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, z], dim=-1)
        mean, log_std = self.net(x).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs, z)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)

        # Tanh correction.
        log_prob = normal.log_prob(x_t) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    @torch.no_grad()
    def act(self, obs: torch.Tensor, z: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean, log_std = self.forward(obs, z)
        if deterministic:
            return torch.tanh(mean)
        std = log_std.exp()
        x_t = torch.distributions.Normal(mean, std).sample()
        return torch.tanh(x_t)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, z_dim: int, action_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        self.q = mlp(obs_dim + z_dim + action_dim, 1, hidden_sizes)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action, z], dim=-1)
        return self.q(x)


class KStepForecaster(nn.Module):
    """Open-loop dynamics/reward decoder.

    Given z_t, s_t, and a sequence of actions a_t:t+K-1, predict
    s_{t+1:t+K} and r_{t:t+K-1}. The z_t is fixed across the forecast horizon.
    """

    def __init__(self, obs_dim: int, z_dim: int, action_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = mlp(obs_dim + z_dim + action_dim, obs_dim + 1, hidden_sizes)
        self.obs_dim = obs_dim

    def forward(self, z: torch.Tensor, start_obs: torch.Tensor, action_seq: torch.Tensor):
        """z: (N,H), start_obs: (N,O), action_seq: (N,K,A)."""
        current = start_obs
        pred_next_obs = []
        pred_rewards = []

        horizon = action_seq.shape[1]
        for k in range(horizon):
            inp = torch.cat([z, current, action_seq[:, k]], dim=-1)
            out = self.net(inp)
            delta_obs = out[:, : self.obs_dim]
            reward = out[:, self.obs_dim : self.obs_dim + 1]

            current = current + delta_obs
            pred_next_obs.append(current)
            pred_rewards.append(reward)

        return torch.stack(pred_next_obs, dim=1), torch.cat(pred_rewards, dim=1)


@dataclass
class TttSacModules:
    context: TTTContextEncoder
    actor: SquashedGaussianActor
    q1: Critic
    q2: Critic
    q1_target: Critic
    q2_target: Critic
    forecaster: KStepForecaster


def build_ttt_config(args) -> TTTConfig:
    return TTTConfig(
        vocab_size=1,
        hidden_size=args.hidden_size,
        intermediate_size=args.hidden_size * 3,
        num_attention_heads=args.num_attention_heads,
        max_position_embeddings=max(2048, args.mdps_per_update * args.rollout_steps),
        num_hidden_layers=args.num_hidden_layers,
        ttt_layer_type=args.ttt_layer_type,
        rms_norm_eps=1e-5,
        use_cache=True,
        mini_batch_size=args.mini_batch_size,
        scan_checkpoint_group_size=0,
        tie_word_embeddings=False,
    )


def build_modules(args, obs_dim: int, action_dim: int, token_dim: int, device: torch.device) -> TttSacModules:
    config = build_ttt_config(args)
    context = TTTContextEncoder(config, token_dim, zero_ttt_output=args.zero_ttt_output).to(device)
    actor = SquashedGaussianActor(obs_dim, args.hidden_size, action_dim, args.hidden_sizes).to(device)
    q1 = Critic(obs_dim, args.hidden_size, action_dim, args.hidden_sizes).to(device)
    q2 = Critic(obs_dim, args.hidden_size, action_dim, args.hidden_sizes).to(device)
    q1_target = Critic(obs_dim, args.hidden_size, action_dim, args.hidden_sizes).to(device)
    q2_target = Critic(obs_dim, args.hidden_size, action_dim, args.hidden_sizes).to(device)
    forecaster = KStepForecaster(obs_dim, args.hidden_size, action_dim, args.hidden_sizes).to(device)

    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    return TttSacModules(
        context=context,
        actor=actor,
        q1=q1,
        q2=q2,
        q1_target=q1_target,
        q2_target=q2_target,
        forecaster=forecaster,
    )


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for src, tgt in zip(source.parameters(), target.parameters()):
            tgt.data.mul_(1.0 - tau).add_(tau * src.data)
