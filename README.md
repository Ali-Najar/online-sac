

## Oracle observations and video logging

Oracle baseline:

```bash
python train_sac_ttt.py --oracle
```

This appends the current target velocity to the observation vector, matching the LILAC oracle idea.

Video logging:

```bash
python train_sac_ttt.py --video-interval 50 --video-fps 30
```

Videos are saved under:

```text
<out-dir>/videos/trajectory_update_XXXXXX.mp4
```

The video rollout is deterministic and does not update replay or observation statistics.


## Continuous target-change mode

Use this flag to avoid resetting the MuJoCo state when the target velocity changes:

```bash
python train_sac_ttt.py --no-reset-on-mdp-change
```

Default behavior:

```text
target velocity changes -> env.reset()
```

With `--no-reset-on-mdp-change`:

```text
target velocity changes -> no env.reset(); only env.target_velocity changes
```

This is useful if you want the cheetah to keep moving continuously while the reward target changes every `--rollout-steps` environment steps.

Changed files for this feature:

```text
envs.py
rollout.py
train_sac_ttt.py
README.md
```


## Training-trajectory videos

`--video-interval` now saves an actual training trajectory from env 0 during
rollout collection, instead of creating a separate visualization/evaluation
environment.

```bash
python train_sac_ttt.py --video-interval 50
```

The saved file is:

```text
<out-dir>/videos/training_update_XXXXXX.mp4
```

Changed files for this behavior:

```text
envs.py
rollout.py
train_sac_ttt.py
README.md
```
