import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore
    # Training-time real-time chunking (arXiv 2512.05964): simulate an inference delay of
    # ``rtc_delay`` steps by conditioning on that many ground-truth actions as a clean prefix and
    # computing the loss on the remaining postfix only. None disables it. Unlike the paper, which
    # samples the delay per example, this uses one constant delay — inference configs always use
    # this form, with the delay the deployment actually serves.
    rtc_delay: int | None = None
    # Per-example delay sampling (the paper's scheme): entry ``i`` is the probability of an
    # ``i``-step delay, drawn independently for every training example. The last entry must be
    # positive (trailing zeros are refused), so ``len(rtc_delay_probs) - 1`` is the largest delay
    # the checkpoint learns to condition on. Training-only and mutually exclusive with
    # ``rtc_delay``: at inference the served delay is one concrete number, so a checkpoint trained
    # with this is served with ``rtc_delay`` set instead; a serve-time call without a prefix then
    # samples the unconditioned delay-0 mode.
    rtc_delay_probs: tuple[float, ...] | None = None

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.rtc_delay is not None:
            if not self.pi05:
                raise ValueError("rtc_delay is only implemented for pi05=True")
            if not 0 <= self.rtc_delay < self.action_horizon:
                raise ValueError(f"rtc_delay must be in [0, {self.action_horizon}), got {self.rtc_delay}")
        if self.rtc_delay_probs is not None:
            # A caller building the config from parsed JSON hands a list; normalize so the config
            # stays hashable.
            object.__setattr__(self, "rtc_delay_probs", tuple(float(p) for p in self.rtc_delay_probs))
            if not self.pi05:
                raise ValueError("rtc_delay_probs is only implemented for pi05=True")
            if self.rtc_delay is not None:
                raise ValueError(
                    "rtc_delay and rtc_delay_probs are mutually exclusive: a constant delay is "
                    "rtc_delay_probs with all mass on one entry"
                )
            if not 1 <= len(self.rtc_delay_probs) - 1 < self.action_horizon:
                raise ValueError(
                    f"rtc_delay_probs covers delays 0..len-1, so it needs 2..{self.action_horizon} "
                    f"entries, got {len(self.rtc_delay_probs)}"
                )
            if min(self.rtc_delay_probs) < 0:
                raise ValueError(f"rtc_delay_probs must be non-negative, got {self.rtc_delay_probs}")
            if self.rtc_delay_probs[-1] == 0:
                raise ValueError(
                    "the last entry of rtc_delay_probs must be positive: trailing zeros would hide "
                    "the largest delay the checkpoint is trained for"
                )
            if abs(sum(self.rtc_delay_probs) - 1.0) > 1e-6:
                raise ValueError(f"rtc_delay_probs must sum to 1, got {sum(self.rtc_delay_probs)}")
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
