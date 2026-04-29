from __future__ import annotations

import torch


def save_checkpoint(path: str, modules, obs_rms, args, update: int, total_env_steps: int) -> None:
    torch.save(
        {
            "context": modules.context.state_dict(),
            "actor": modules.actor.state_dict(),
            "q1": modules.q1.state_dict(),
            "q2": modules.q2.state_dict(),
            "q1_target": modules.q1_target.state_dict(),
            "q2_target": modules.q2_target.state_dict(),
            "forecaster": modules.forecaster.state_dict(),
            "obs_rms": obs_rms.state_dict(),
            "args": vars(args),
            "update": int(update),
            "total_env_steps": int(total_env_steps),
        },
        path,
    )


def load_checkpoint(path: str, modules, obs_rms, device):
    checkpoint = torch.load(path, map_location=device)
    modules.context.load_state_dict(checkpoint["context"])
    modules.actor.load_state_dict(checkpoint["actor"])
    modules.q1.load_state_dict(checkpoint["q1"])
    modules.q2.load_state_dict(checkpoint["q2"])
    modules.q1_target.load_state_dict(checkpoint["q1_target"])
    modules.q2_target.load_state_dict(checkpoint["q2_target"])
    modules.forecaster.load_state_dict(checkpoint["forecaster"])

    if "obs_rms" in checkpoint:
        obs_rms.load_state_dict(checkpoint["obs_rms"])

    return int(checkpoint.get("update", 0)), int(checkpoint.get("total_env_steps", 0))
