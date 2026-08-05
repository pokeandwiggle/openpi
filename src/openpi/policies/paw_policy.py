"""Data transforms for the Poke & Wiggle dual-FR3 pedestal robot.

The LeRobot dataset stores state and action split across per-arm columns rather
than as a single flat vector. ``PawInputs`` concatenates them into the 16-dim
contract the pi0.5 model expects, in this fixed order (left block = dims 0-7):

    [ tcp_pose_left(7), gripper_width_left(1),
      tcp_pose_right(7), gripper_width_right(1) ]

The robot only drives the right arm; the left arm is frozen. We keep it in the
vector to match the deploy-time observation layout and squash it in the norm
stats (left dims 0-7 get std=1 so they normalize to ~0).

The three cameras map onto pi0.5's three image slots:
    rgb_static -> base_0_rgb, rgb_left -> left_wrist_0_rgb, rgb_right -> right_wrist_0_rgb.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# Sub-feature keys concatenated into the flat state / action vectors, in order.
STATE_KEYS = (
    "observation.state.tcp_pose_left",
    "observation.state.gripper_width_left",
    "observation.state.tcp_pose_right",
    "observation.state.gripper_width_right",
)
ACTION_KEYS = (
    "action.tcp_pose_left",
    "action.gripper_width_left",
    "action.tcp_pose_right",
    "action.gripper_width_right",
)
# Flat dimension of the assembled state / action vector.
PAW_DIM = 16
# Indices of the frozen left arm within the flat vector (squashed in norm stats).
LEFT_ARM_DIMS = (0, 1, 2, 3, 4, 5, 6, 7)


def _concat_columns(data: dict, keys) -> np.ndarray:
    """Concatenate dataset columns along the feature axis.

    Pose columns arrive as ``(..., 7)`` while scalar-width columns (gripper_width)
    arrive squeezed to ``(...,)``. Align every column to the group's max rank by
    appending a trailing feature axis, then concatenate. Works for both single-frame
    state (rank 1) and action chunks carrying a leading horizon axis (rank 2).
    """
    arrays = [np.atleast_1d(np.asarray(data[k])) for k in keys]
    rank = max(a.ndim for a in arrays)
    arrays = [a.reshape(*a.shape, *([1] * (rank - a.ndim))) for a in arrays]
    return np.concatenate(arrays, axis=-1)


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class PawInputs(transforms.DataTransformFn):
    """Assembles dataset columns into the pi0.5 model input contract.

    Used for both training and inference. During training the dataset provides
    the dotted ``observation.state.*`` / ``action.*`` columns directly (actions
    carry a leading action-horizon axis). During inference the policy server
    provides the same keys for the current frame (no action keys).
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        state = _concat_columns(data, STATE_KEYS)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": _parse_image(data["observation.images.rgb_static"]),
                "left_wrist_0_rgb": _parse_image(data["observation.images.rgb_left"]),
                "right_wrist_0_rgb": _parse_image(data["observation.images.rgb_right"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Actions are only present during training; each key carries an action-horizon axis.
        if all(k in data for k in ACTION_KEYS):
            inputs["actions"] = _concat_columns(data, ACTION_KEYS)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PawOutputs(transforms.DataTransformFn):
    """Slices the padded model output back to the 16-dim action vector (inference only)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :PAW_DIM])}
