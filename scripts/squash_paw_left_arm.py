"""Squash the frozen left arm in the Poke & Wiggle norm stats.

The dual-FR3 robot only drives the right arm; the left arm is static, so its
state/action columns have ~zero spread (std ~2e-4). pi0.5 uses quantile
normalization: ``(x - q01) / (q99 - q01 + 1e-6) * 2 - 1``. With q99 ~ q01 the
tiny left-arm jitter would be amplified ~thousands-fold. This script rewrites the
left-arm dims so they normalize to ~0 (and unnormalize back to their mean):

    std  -> 1.0
    q01  -> mean - 0.5
    q99  -> mean + 0.5   (mean kept)

Run after ``compute_norm_stats.py``:
    uv run scripts/squash_paw_left_arm.py --repo-id pokeandwiggle/<dataset>
"""

import dataclasses

import numpy as np
import tyro

import openpi.policies.paw_policy as paw_policy
import openpi.shared.normalize as normalize
import openpi.training.config as _config


def main(config_name: str = "pi05_paw", repo_id: str | None = None) -> None:
    config = _config.get_config(config_name)
    if repo_id is not None:
        config = dataclasses.replace(config, data=dataclasses.replace(config.data, repo_id=repo_id))
    data_config = config.data.create(config.assets_dirs, config.model)
    stats_dir = config.assets_dirs / data_config.asset_id

    stats = normalize.load(stats_dir)
    dims = list(paw_policy.LEFT_ARM_DIMS)

    squashed = {}
    for key, ns in stats.items():
        std, q01, q99 = np.array(ns.std), np.array(ns.q01), np.array(ns.q99)
        std[dims] = 1.0
        q01[dims] = ns.mean[dims] - 0.5
        q99[dims] = ns.mean[dims] + 0.5
        squashed[key] = dataclasses.replace(ns, std=std, q01=q01, q99=q99)
        print(f"{key}: squashed left-arm dims {dims}")

    normalize.save(stats_dir, squashed)
    print(f"Wrote squashed stats to: {stats_dir / 'norm_stats.json'}")


if __name__ == "__main__":
    tyro.cli(main)
