# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert LeRobot-style episode frames into RLinf replay-buffer data."""

import argparse
import io
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer

_ACTION_KEYS = ("actions", "action")
_DONE_KEYS = ("done", "dones")
_REWARD_KEYS = ("rewards", "reward")
_SUCCESS_KEYS = ("is_success", "success", "success_once", "success_at_end")
_TERMINATION_KEYS = ("terminated", "termination", "terminations")
_TRUNCATION_KEYS = ("truncated", "truncation", "truncations")
_STATE_KEYS = ("states", "state", "observation.state", "observation/state")
_NEXT_STATE_KEYS = (
    "next_states",
    "next_state",
    "next_observation.state",
    "next_observation/state",
)
_MAIN_IMAGE_KEYS = (
    "main_images",
    "image",
    "full_image",
    "observation.image",
    "observation/image",
    "observation.images.front",
    # CRIT-R4-6: route slash-form to the same target so it isn't silently
    # dropped after passing the unknown-key guard.
    "observation/images/front",
)
_NEXT_MAIN_IMAGE_KEYS = (
    "next_main_images",
    "next_image",
    "next_full_image",
    "next_observation.image",
    "next_observation/image",
    "next_observation.images.front",
    "next_observation/images/front",
)
_WRIST_IMAGE_KEYS = (
    "wrist_images",
    "wrist_image",
    "observation.wrist_image",
    "observation/wrist_image",
    "observation.images.wrist",
    "observation/images/wrist",
)
_NEXT_WRIST_IMAGE_KEYS = (
    "next_wrist_images",
    "next_wrist_image",
    "next_observation.wrist_image",
    "next_observation/wrist_image",
    "next_observation.images.wrist",
    "next_observation/images/wrist",
)
_EXTRA_VIEW_IMAGE_KEYS = (
    "extra_view_images",
    "extra_view_image",
    "observation.extra_view_image",
    "observation/extra_view_image",
)
_NEXT_EXTRA_VIEW_IMAGE_KEYS = (
    "next_extra_view_images",
    "next_extra_view_image",
    "next_observation.extra_view_image",
    "next_observation/extra_view_image",
)
_REQUIRED_PARQUET_KEYS = (
    ("episode_index",),
    ("frame_index", "index"),
    _STATE_KEYS,
    _ACTION_KEYS,
)
_STATE_ONLY_PARQUET_KEYS = (
    "episode_index",
    "frame_index",
    "index",
    *_STATE_KEYS,
    *_NEXT_STATE_KEYS,
    *_ACTION_KEYS,
    *_DONE_KEYS,
    *_REWARD_KEYS,
    *_SUCCESS_KEYS,
    *_TERMINATION_KEYS,
    *_TRUNCATION_KEYS,
    "intervene_flag",
    "task",
)
_INDEXED_KEY_PATTERN = re.compile(r"^(?P<prefix>.+?)(?P<sep>[-/.])(?P<index>\d+)$")
_IMAGE_OBS_KEYS = ("main_images", "wrist_images", "extra_view_images")


def _as_tensor(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
    elif hasattr(value, "flags") and not value.flags.writeable:
        tensor = torch.as_tensor(value.copy())
    else:
        tensor = torch.as_tensor(value)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.contiguous()


def _is_pil_image(value: Any) -> bool:
    try:
        from PIL import Image
    except ImportError:
        return False
    return isinstance(value, Image.Image)


def _decode_image_bytes(value: Any) -> torch.Tensor:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ValueError("Pillow is required to decode image bytes.") from exc

    try:
        with Image.open(io.BytesIO(bytes(value))) as image:
            array = np.asarray(image.convert("RGB"))
    except Exception as exc:
        raise ValueError("Failed to decode image bytes.") from exc
    return _as_tensor(array)


def _decode_image_path(value: Any, frame: dict[str, Any]) -> torch.Tensor:
    image_path = Path(str(value))
    dataset_root = frame.get("_lerobot_dataset_root")
    if dataset_root is not None:
        resolved_root = Path(str(dataset_root)).resolve()
        resolved_path = (
            image_path.resolve()
            if image_path.is_absolute()
            else (resolved_root / image_path).resolve()
        )
        if not resolved_path.is_relative_to(resolved_root):
            raise ValueError(
                f"Image path '{image_path}' escapes dataset root '{resolved_root}'."
            )
        if not resolved_path.is_file():
            raise ValueError(f"Image file not found: {resolved_path}")
        return _decode_image_bytes(resolved_path.read_bytes())
    if image_path.is_absolute():
        if not image_path.is_file():
            raise ValueError(f"Image file not found: {image_path}")
        return _decode_image_bytes(image_path.read_bytes())
    raise ValueError(
        f"Relative image path '{image_path}' cannot be resolved without "
        "_lerobot_dataset_root."
    )


def _validate_image_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim not in (3, 4):
        raise ValueError(
            "Image tensor shape must be [H, W, C] or [V, H, W, C]; "
            f"got {tuple(tensor.shape)}."
        )
    if tensor.shape[-1] not in (1, 3, 4):
        raise ValueError(
            "Image tensor shape must use channel-last layout with 1, 3, or 4 "
            f"channels; got {tuple(tensor.shape)}."
        )
    return tensor.contiguous()


def _image_tensor(value: Any, frame: dict[str, Any]) -> torch.Tensor:
    """Convert a frame image cell to a validated ``[H, W, C]`` tensor.

    Channel-order contract: every code path here produces / expects RGB
    channel-last layout. Bytes / path / PIL inputs are converted via
    ``PIL.Image.convert("RGB")``. RAW numpy / torch tensor inputs MUST
    already be RGB — the converter cannot detect BGR from a raw uint8
    array and would silently mislabel ``CollectEpisode`` images recorded
    from BGR cameras (e.g. OpenCV / RealWorld) as RGB.

    Producers writing raw arrays MUST convert before handing data to
    ``CollectEpisode`` / the LeRobot replay-buffer converter. The
    documented contract is restated in
    ``docs/source-en/rst_source/tutorials/components/replay_buffer.rst``.
    """
    if isinstance(value, dict):
        for key in ("array", "data", "image"):
            if key in value and value[key] is not None:
                return _image_tensor(value[key], frame)
        if value.get("bytes") is not None:
            return _validate_image_tensor(_decode_image_bytes(value["bytes"]))
        if value.get("path") is not None:
            return _validate_image_tensor(_decode_image_path(value["path"], frame))
        raise ValueError("Image metadata does not contain bytes, path, or array data.")

    if isinstance(value, (bytes, bytearray, memoryview)):
        return _validate_image_tensor(_decode_image_bytes(value))
    if isinstance(value, (str, Path)):
        return _validate_image_tensor(_decode_image_path(value, frame))
    if _is_pil_image(value):
        return _validate_image_tensor(_as_tensor(np.asarray(value.convert("RGB"))))

    tensor = _as_tensor(value)
    return _validate_image_tensor(tensor)


def _scalar(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    tensor = _as_tensor(value)
    if tensor.numel() == 0:
        return default
    return tensor.reshape(-1)[0].item()


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, dict):
        return all(_is_missing_value(item) for item in value.values())
    if isinstance(value, (str, bytes, bytearray, memoryview, Path)):
        return False
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return True
        if value.numel() == 1 and value.dtype.is_floating_point:
            # Treat scalar / length-1 NaN as missing; finite values are present.
            return bool(torch.isnan(value.reshape(-1)[0]).item())
        return False
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return True
        if value.size == 1 and np.issubdtype(value.dtype, np.floating):
            return bool(np.isnan(value.reshape(-1)[0]))
        if value.ndim == 0:
            value = value.item()
        else:
            return False
    try:
        import pandas as pd
    except ImportError:
        pd = None
    if pd is not None:
        try:
            missing = pd.isna(value)
            if isinstance(missing, (bool, np.bool_)):
                return bool(missing)
        except (TypeError, ValueError):
            pass
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def _strict_scalar_bool(value: Any, *, field_name: str) -> bool:
    """Convert a scalar tensor / numpy / Python value to a strict bool.

    Accepts:
      * Python ``bool`` (or numpy ``bool_``)
      * Finite numeric values that compare exactly equal to ``0`` or ``1``
      * Tensors / arrays containing one of the above
    Rejects:
      * ``NaN`` / ``+/-Inf``
      * Non-binary numeric values (``0.5``, ``2``, ``-1``, …)

    This guards against ``bool(0.5)``-style truthy coercion that previously
    accepted arbitrary numeric data as a true bool.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    # Unwrap 0-d tensor / array to a Python scalar.
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"LeRobot {field_name} must be scalar; got shape {tuple(value.shape)}."
            )
        if value.dtype == torch.bool:
            return bool(value.reshape(-1)[0].item())
        value = value.reshape(-1)[0].item()
    elif isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(
                f"LeRobot {field_name} must be scalar; got shape {tuple(value.shape)}."
            )
        if value.dtype == np.bool_:
            return bool(value.reshape(-1)[0])
        value = value.reshape(-1)[0].item()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        if value not in (0, 1):
            raise ValueError(
                f"LeRobot {field_name} must be a binary 0/1 value; got {value}."
            )
        return bool(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError(
                f"LeRobot {field_name} must be a finite 0/1 value; got {value}."
            )
        if value != 0.0 and value != 1.0:
            raise ValueError(
                f"LeRobot {field_name} must be exactly 0 or 1; got {value}."
            )
        return bool(value)
    raise ValueError(
        f"LeRobot {field_name} must be a bool or 0/1 numeric value; got "
        f"type={type(value).__name__}, value={value!r}."
    )


def _find_value(frame: dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in frame and not _is_missing_value(frame[key]):
            return frame[key]
    return None


def _find_key_value(
    frame: dict[str, Any], keys: Sequence[str]
) -> tuple[str, Any] | None:
    for key in keys:
        if key in frame and not _is_missing_value(frame[key]):
            return key, frame[key]
    return None


def _indexed_key_prefixes(keys: Sequence[str]) -> tuple[str, ...]:
    return tuple(key[:-1] if key.endswith("s") else key for key in keys)


def _find_indexed_key_values(
    frame: dict[str, Any],
    keys: Sequence[str],
) -> list[tuple[int, str, Any]]:
    prefixes = set(_indexed_key_prefixes(keys))
    indexed_values = []
    for key, value in frame.items():
        match = _INDEXED_KEY_PATTERN.match(key)
        if match is None:
            continue
        if match.group("prefix") in prefixes and not _is_missing_value(value):
            indexed_values.append((int(match.group("index")), key, value))
    indexed_values.sort(key=lambda item: item[0])
    return indexed_values


def _find_obs_key_values(
    frame: dict[str, Any],
    keys: Sequence[str],
) -> list[tuple[int | None, str, Any]] | None:
    key_value = _find_key_value(frame, keys)
    if key_value is not None:
        key, value = key_value
        return [(None, key, value)]

    indexed_key_values = _find_indexed_key_values(frame, keys)
    if not indexed_key_values:
        return None
    return indexed_key_values


def _required_value(frame: dict[str, Any], keys: Sequence[str]) -> Any:
    value = _find_value(frame, keys)
    if value is None:
        key_names = ", ".join(keys)
        raise ValueError(f"LeRobot frame is missing one of: {key_names}")
    return value


def _has_value(frame: dict[str, Any], keys: Sequence[str]) -> bool:
    return _find_value(frame, keys) is not None


def _has_action(frame: dict[str, Any]) -> bool:
    return _has_value(frame, _ACTION_KEYS)


def _has_next_observation(frame: dict[str, Any]) -> bool:
    return _has_value(frame, _NEXT_STATE_KEYS)


def _check_finite_nonempty(tensor: torch.Tensor, *, field_name: str) -> torch.Tensor:
    if tensor.numel() == 0:
        raise ValueError(f"LeRobot {field_name} is empty (no elements).")
    if not torch.isfinite(tensor).all():
        raise ValueError(
            f"LeRobot {field_name} contains non-finite values (NaN/Inf); "
            "every element must be finite."
        )
    return tensor


def _state_tensor(value: Any) -> torch.Tensor:
    tensor = _as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    elif tensor.ndim == 2 and tensor.shape[0] == 1:
        tensor = tensor.reshape(-1)
    elif tensor.ndim != 1:
        raise ValueError(
            "LeRobot state must be scalar, 1-D, or [1, state_dim]; "
            f"got shape {tuple(tensor.shape)}."
        )
    return _check_finite_nonempty(tensor, field_name="state")


def _stack_states(
    frames: Sequence[dict[str, Any]], keys: Sequence[str]
) -> torch.Tensor:
    values = [_state_tensor(_required_value(frame, keys)) for frame in frames]
    return torch.stack(values, dim=0).unsqueeze(1).contiguous()


def _action_tensor(value: Any) -> torch.Tensor:
    tensor = _as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 1:
        pass
    elif tensor.ndim == 2 and tensor.shape[0] == 1:
        tensor = tensor.reshape(-1)
    else:
        raise ValueError(
            "LeRobot action must be scalar, 1-D, or [1, action_dim]; "
            f"got shape {tuple(tensor.shape)}."
        )
    return _check_finite_nonempty(tensor, field_name="action")


def _stack_actions(frames: Sequence[dict[str, Any]]) -> torch.Tensor:
    values = [_action_tensor(_required_value(frame, _ACTION_KEYS)) for frame in frames]
    return torch.stack(values, dim=0).unsqueeze(1).contiguous()


def _validate_indexed_obs_keys(
    *,
    key_values: Sequence[tuple[int | None, str, Any]],
    output_key: str,
    frame_index: int,
    expected_indices: tuple[int, ...] | None,
) -> tuple[int, ...] | None:
    indices = tuple(index for index, _, _ in key_values if index is not None)
    if not indices:
        if expected_indices is not None:
            raise ValueError(
                f"LeRobot indexed observation '{output_key}' changes from indexed "
                f"keys to non-indexed keys at frame {frame_index}."
            )
        return expected_indices

    if len(indices) != len(set(indices)):
        raise ValueError(
            f"LeRobot indexed observation '{output_key}' has duplicate indices "
            f"at frame {frame_index}: {indices}."
        )
    contiguous_indices = tuple(range(len(indices)))
    if indices != contiguous_indices:
        raise ValueError(
            f"LeRobot indexed observation '{output_key}' must use zero-based "
            f"contiguous indices; got {indices} at frame {frame_index}."
        )
    if expected_indices is not None and indices != expected_indices:
        raise ValueError(
            f"LeRobot indexed observation '{output_key}' index set changed at "
            f"frame {frame_index}: expected {expected_indices}, got {indices}."
        )
    return indices


def _stack_optional_obs(
    frames: Sequence[dict[str, Any]],
    keys: Sequence[str],
    *,
    output_key: str,
    state_only: bool,
    dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    if state_only:
        return None

    values = []
    saw_key = False
    missing_indices: list[int] = []
    expected_mode: str | None = None
    expected_indices: tuple[int, ...] | None = None
    for index, frame in enumerate(frames):
        key_values = _find_obs_key_values(frame, keys)
        if key_values is None:
            if saw_key:
                key_names = ", ".join(keys)
                raise ValueError(
                    f"LeRobot observation '{output_key}' is missing from frame {index}; "
                    f"expected one of: {key_names}. Use --state-only to ignore image fields."
                )
            missing_indices.append(index)
            values.append(None)
            continue

        saw_key = True
        mode = (
            "indexed"
            if any(key_index is not None for key_index, _, _ in key_values)
            else "direct"
        )
        if expected_mode is not None and mode != expected_mode:
            raise ValueError(
                f"LeRobot observation '{output_key}' changes schema mode from "
                f"{expected_mode} to {mode} at frame {index}."
            )
        expected_mode = mode
        expected_indices = _validate_indexed_obs_keys(
            key_values=key_values,
            output_key=output_key,
            frame_index=index,
            expected_indices=expected_indices,
        )
        try:
            tensors = [_image_tensor(value, frame) for _, _, value in key_values]
            tensor = (
                torch.stack(tensors, dim=0).contiguous()
                if mode == "indexed"
                else tensors[0]
            )
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
        except ValueError as exc:
            key_names = ", ".join(key for _, key, _ in key_values)
            raise ValueError(
                f"LeRobot observation '{key_names}' cannot be converted to a tensor. "
                "This usually means the parquet stores image metadata or paths, "
                "not decoded image arrays. Use --state-only to build a state-only "
                f"replay buffer, or decode images before conversion. {exc}"
            ) from exc
        values.append(tensor)

    if not saw_key:
        return None
    if any(value is None for value in values):
        key_names = ", ".join(keys)
        missing_text = ", ".join(str(index) for index in missing_indices)
        raise ValueError(
            f"LeRobot observation '{output_key}' is present only on some frames; "
            f"missing frame indices: {missing_text}. Expected one of {key_names} "
            "on every frame. Use --state-only to ignore image fields."
        )
    return torch.stack(values, dim=0).unsqueeze(1).contiguous()


# Keys that are recognised by the buffer (any observation.images.<name>
# entry NOT covered below is treated as unknown and rejected, otherwise
# silently dropped data could go unnoticed).
_KNOWN_OBSERVATION_IMAGE_KEYS = frozenset(
    (
        # Current-step keys.
        "observation.image",
        "observation/image",
        "observation.images.front",
        "observation.wrist_image",
        "observation/wrist_image",
        "observation.images.wrist",
        "observation.extra_view_image",
        "observation/extra_view_image",
        # Next-step keys.
        "next_observation.image",
        "next_observation/image",
        "next_observation.images.front",
        "next_observation.wrist_image",
        "next_observation/wrist_image",
        "next_observation.images.wrist",
        "next_observation.extra_view_image",
        "next_observation/extra_view_image",
    )
)
_OBSERVATION_IMAGES_PREFIXES = (
    "observation.images.",
    "observation/images/",
    "next_observation.images.",
    "next_observation/images/",
)

# Camera names that ARE routed by ``_MAIN_IMAGE_KEYS`` / ``_WRIST_IMAGE_KEYS``
# (and the ``next_*`` variants).  The unknown-key guard accepts a key only
# when its trailing camera name is in this set; otherwise it would silently
# drop e.g. ``cam_left_wrist`` (substring "wrist") or accept slash-form
# ``observation/images/front`` even though only the dot-form is actually
# routed downstream (CRIT-R4-6).
_RECOGNISED_CAMERA_NAMES = frozenset(("front", "wrist"))


def _detect_unknown_camera_keys(
    frame: dict[str, Any], frame_position: int
) -> None:
    """Raise if *frame* exposes ``observation.images.<name>`` keys we don't recognise.

    Currently the converter only routes ``front`` to ``main_images`` and
    ``wrist`` to ``wrist_images``.  Other camera names (e.g. ``top``,
    ``cam_left_wrist``) would otherwise be silently dropped, corrupting the
    replay buffer for multi-camera setups.
    """
    unknown_keys = []
    for key in frame:
        if not isinstance(key, str):
            continue
        if key in _KNOWN_OBSERVATION_IMAGE_KEYS:
            continue
        matched_prefix = next(
            (prefix for prefix in _OBSERVATION_IMAGES_PREFIXES if key.startswith(prefix)),
            None,
        )
        if matched_prefix is None:
            continue
        # Only an EXACT trailing ``front``/``wrist`` is a known route — substring
        # matching would silently accept `cam_left_wrist`, `front_view`, etc.
        camera_name = key[len(matched_prefix):]
        if camera_name in ("front", "wrist"):
            continue
        unknown_keys.append(key)
    if unknown_keys:
        raise ValueError(
            "LeRobot frame at position "
            f"{frame_position} contains camera keys the converter does not "
            f"know how to route: {sorted(unknown_keys)}. The converter only "
            "recognises 'observation.images.front' (→ main_images) and "
            "'observation.images.wrist' (→ wrist_images); copy unknown camera "
            "tensors into 'extra_view_image' (or 'extra_view_images') before "
            "conversion, or extend the converter to map the additional "
            "cameras explicitly."
        )


def _build_obs(
    frames: Sequence[dict[str, Any]], *, state_only: bool, use_next: bool = False
) -> dict[str, torch.Tensor]:
    state_keys = _NEXT_STATE_KEYS if use_next else _STATE_KEYS
    main_image_keys = _NEXT_MAIN_IMAGE_KEYS if use_next else _MAIN_IMAGE_KEYS
    wrist_image_keys = _NEXT_WRIST_IMAGE_KEYS if use_next else _WRIST_IMAGE_KEYS
    extra_view_image_keys = (
        _NEXT_EXTRA_VIEW_IMAGE_KEYS if use_next else _EXTRA_VIEW_IMAGE_KEYS
    )
    # Reject unknown camera keys before stacking — fail loudly so unrelated
    # cameras don't get silently dropped (MAJ-10).
    if not state_only:
        for position, frame in enumerate(frames):
            _detect_unknown_camera_keys(frame, position)
    obs: dict[str, torch.Tensor] = {"states": _stack_states(frames, state_keys)}
    optional_obs = {
        "main_images": _stack_optional_obs(
            frames,
            main_image_keys,
            output_key="main_images",
            state_only=state_only,
        ),
        "wrist_images": _stack_optional_obs(
            frames,
            wrist_image_keys,
            output_key="wrist_images",
            state_only=state_only,
        ),
        "extra_view_images": _stack_optional_obs(
            frames,
            extra_view_image_keys,
            output_key="extra_view_images",
            state_only=state_only,
        ),
    }
    obs.update({key: value for key, value in optional_obs.items() if value is not None})
    return obs


def _validate_obs_symmetry(
    curr_obs: dict[str, torch.Tensor], next_obs: dict[str, torch.Tensor]
) -> None:
    for key in ("states", *_IMAGE_OBS_KEYS):
        has_curr = key in curr_obs and curr_obs[key] is not None
        has_next = key in next_obs and next_obs[key] is not None
        if has_curr != has_next:
            missing_side = "next_obs" if has_curr else "curr_obs"
            raise ValueError(
                f"LeRobot trajectory has asymmetric observations: "
                f"{missing_side} is missing {key}."
            )
        if has_curr and curr_obs[key].shape != next_obs[key].shape:
            raise ValueError(
                f"LeRobot trajectory has mismatched {key} shapes between "
                f"curr_obs {tuple(curr_obs[key].shape)} and "
                f"next_obs {tuple(next_obs[key].shape)}."
            )


def _strict_bool_from_frame(
    frame: dict[str, Any], keys: Sequence[str], default: bool, *, field_name: str
) -> bool:
    value = _find_value(frame, keys)
    if value is None:
        return default
    return _strict_scalar_bool(value, field_name=field_name)


def _optional_bool(
    frame: dict[str, Any], keys: Sequence[str], *, field_name: str
) -> bool | None:
    value = _find_value(frame, keys)
    if value is None:
        return None
    return _strict_scalar_bool(value, field_name=f"{field_name} terminal flag")


def _has_terminal_metadata(frame: dict[str, Any] | None) -> bool:
    if frame is None:
        return False
    return any(
        _find_value(frame, keys) is not None
        for keys in (_DONE_KEYS, _TERMINATION_KEYS, _TRUNCATION_KEYS)
    )


def _target_owns_terminal_metadata(
    source_frame: dict[str, Any],
    target_frame: dict[str, Any] | None,
) -> bool:
    if target_frame is None or not _has_terminal_metadata(target_frame):
        return False
    # Reject explicit disagreements between source and target frames so we
    # don't silently pick one side's view of the world (MAJ-11). Mismatches
    # almost always indicate the source pipeline is buggy.
    _check_source_target_terminal_agreement(source_frame, target_frame)
    if any(
        _find_value(target_frame, keys) is not None
        for keys in (_TERMINATION_KEYS, _TRUNCATION_KEYS)
    ):
        return True

    target_done = _optional_bool(target_frame, _DONE_KEYS, field_name="done")
    if target_done is True:
        return True

    source_done = _optional_bool(source_frame, _DONE_KEYS, field_name="done")
    source_has_split = any(
        _find_value(source_frame, keys) is not None
        for keys in (_TERMINATION_KEYS, _TRUNCATION_KEYS)
    )
    return source_done is None and not source_has_split


def _check_source_target_terminal_agreement(
    source_frame: dict[str, Any],
    target_frame: dict[str, Any],
) -> None:
    """Reject contradictory positive terminal claims between source and target.

    The legacy LeRobot layout puts terminal flags on the observation-only
    target frame, so it is common (and valid) for the source frame to leave
    ``done=False`` as a default while the target frame asserts ``done=True``;
    the resolver promotes the target value in that case.  What is **not**
    valid is the opposite: a source frame that asserts a *positive* terminal
    flag (``done=True``, ``terminated=True``, ``truncated=True``) while the
    target frame contradicts it with ``False``.  Previously the resolver
    would silently pick the source frame's view; we now raise to surface the
    real producer bug.
    """
    for keys, field_name in (
        (_DONE_KEYS, "done"),
        (_TERMINATION_KEYS, "terminated"),
        (_TRUNCATION_KEYS, "truncated"),
    ):
        source_value = _optional_bool(source_frame, keys, field_name=field_name)
        target_value = _optional_bool(target_frame, keys, field_name=field_name)
        if source_value is True and target_value is False:
            raise ValueError(
                f"LeRobot source and target frames disagree on '{field_name}': "
                f"source={source_value}, target={target_value}. The source "
                "frame claims a positive terminal flag that the target frame "
                "contradicts; reconcile the two frames (either remove the "
                "duplicate metadata or make both sides agree) before "
                "conversion."
            )


def _validate_terminal_frame_flags(frame: dict[str, Any]) -> None:
    done_value = _optional_bool(frame, _DONE_KEYS, field_name="done")
    termination_value = _optional_bool(
        frame, _TERMINATION_KEYS, field_name="terminated"
    )
    truncation_value = _optional_bool(frame, _TRUNCATION_KEYS, field_name="truncated")
    if done_value is None:
        return
    if termination_value is None and truncation_value is None:
        return
    split_done = _known_split_done(termination_value, truncation_value)
    if split_done is None:
        return
    if done_value != split_done:
        raise ValueError(
            "LeRobot frame has inconsistent done, terminated, and truncated flags."
        )


def _known_split_done(
    termination_value: bool | None, truncation_value: bool | None
) -> bool | None:
    if termination_value is True or truncation_value is True:
        return True
    if termination_value is False and truncation_value is False:
        return False
    return None


def _terminal_flags_from_transition(
    source_frame: dict[str, Any],
    *,
    target_frame: dict[str, Any] | None,
    default_done: bool,
) -> tuple[bool, bool, bool]:
    _validate_terminal_frame_flags(source_frame)
    if target_frame is not None:
        _validate_terminal_frame_flags(target_frame)

    metadata_frame = (
        target_frame
        if _target_owns_terminal_metadata(source_frame, target_frame)
        else source_frame
    )
    done_value = _optional_bool(metadata_frame, _DONE_KEYS, field_name="done")
    termination_value = _optional_bool(
        metadata_frame, _TERMINATION_KEYS, field_name="terminated"
    )
    truncation_value = _optional_bool(
        metadata_frame, _TRUNCATION_KEYS, field_name="truncated"
    )
    if termination_value is True and truncation_value is True:
        raise ValueError("LeRobot frame has both terminated and truncated set to true.")
    split_done = _known_split_done(termination_value, truncation_value)
    has_terminal_metadata = _has_terminal_metadata(metadata_frame)
    if (
        default_done
        and done_value is None
        and split_done is None
        and has_terminal_metadata
    ):
        raise ValueError(
            "LeRobot final action frame has incomplete terminal metadata. "
            "Set done, or provide terminated/truncated flags that determine done."
        )
    if done_value is not None and split_done is not None and done_value != split_done:
        raise ValueError(
            "LeRobot frame has inconsistent done, terminated, and truncated flags."
        )

    done = done_value if done_value is not None else split_done
    if done is None:
        done = default_done
    truncated = (
        truncation_value
        if truncation_value is not None
        else bool(done and termination_value is False)
    )
    terminated = (
        termination_value if termination_value is not None else done and not truncated
    )
    return bool(done), bool(terminated), bool(truncated)


def _reward_from_transition(
    source_frame: dict[str, Any],
    *,
    target_frame: dict[str, Any] | None,
    done: bool,
    terminal_reward: float,
) -> float:
    reward_value = None
    if target_frame is not None:
        reward_value = _find_value(target_frame, _REWARD_KEYS)
    if reward_value is None:
        reward_value = _find_value(source_frame, _REWARD_KEYS)
    if reward_value is not None:
        reward_tensor = _as_tensor(reward_value)
        if reward_tensor.numel() == 0:
            raise ValueError(
                "LeRobot reward is empty (no elements); rewards must be "
                "scalar finite floats per transition."
            )
        if reward_tensor.numel() != 1:
            raise ValueError(
                "LeRobot reward must be scalar per transition; "
                f"got shape {tuple(reward_tensor.shape)}."
            )
        reward_scalar = float(reward_tensor.reshape(-1)[0].item())
        if not np.isfinite(reward_scalar):
            raise ValueError(
                f"LeRobot reward must be finite; got {reward_scalar!r}."
            )
        return reward_scalar
    success = any(
        _strict_bool_from_frame(source_frame, (key,), False, field_name=key)
        for key in _SUCCESS_KEYS
    )
    if target_frame is not None:
        success = success or any(
            _strict_bool_from_frame(target_frame, (key,), False, field_name=key)
            for key in _SUCCESS_KEYS
        )
    return float(terminal_reward if done and success else 0.0)


def _frame_order(frame: dict[str, Any]) -> tuple[int, int]:
    episode_index = int(_scalar(frame.get("episode_index"), 0))
    frame_index = frame.get("frame_index", frame.get("index", 0))
    return episode_index, int(_scalar(frame_index, 0))


def _validate_contiguous_frame_indices(
    ordered_frames: Sequence[dict[str, Any]],
) -> None:
    if not ordered_frames:
        return
    has_frame_index = [
        ("frame_index" in frame or "index" in frame) for frame in ordered_frames
    ]
    if not any(has_frame_index):
        # No frame declares an index — list order is the source of truth and
        # the stable sort above preserves it.  Nothing to validate.
        return
    if not all(has_frame_index):
        # Mixed presence is the dangerous case (MAJ-12): frames without an
        # index default to (episode=0, frame=0) and get silently reordered
        # against frames that do declare one.  Raise instead of silently
        # skipping validation.
        missing_positions = [
            index for index, present in enumerate(has_frame_index) if not present
        ]
        raise ValueError(
            "LeRobot episode frames have inconsistent frame_index presence: "
            "some frames declare 'frame_index' (or 'index') while others do "
            "not. Mixed presence reorders the episode silently. Missing key "
            f"at positions: {missing_positions}."
        )

    episode_index, expected_frame_index = _frame_order(ordered_frames[0])
    for frame in ordered_frames:
        current_episode_index, frame_index = _frame_order(frame)
        if current_episode_index != episode_index:
            raise ValueError(
                "LeRobot episode frames must share one episode_index; "
                f"got {current_episode_index} after {episode_index}."
            )
        if frame_index != expected_frame_index:
            raise ValueError(
                "LeRobot episode frame_index values must be contiguous; "
                f"expected {expected_frame_index}, got {frame_index}."
            )
        expected_frame_index += 1


def _resolve_transition_frames(
    ordered_frames: Sequence[dict[str, Any]],
) -> tuple[Sequence[dict[str, Any]], Sequence[dict[str, Any]] | None]:
    action_frames = [frame for frame in ordered_frames if _has_action(frame)]
    if not action_frames:
        raise ValueError("LeRobot episode does not contain action frames.")

    has_explicit_next = any(_has_next_observation(frame) for frame in action_frames)
    if has_explicit_next:
        missing_next_indices = [
            str(index)
            for index, frame in enumerate(action_frames)
            if not _has_next_observation(frame)
        ]
        if missing_next_indices:
            raise ValueError(
                "LeRobot action frames are missing explicit next observation fields "
                f"at source indices: {', '.join(missing_next_indices)}."
            )
        return action_frames, None

    if len(ordered_frames) < 2:
        raise ValueError("At least two LeRobot frames are required for one transition.")
    if _has_action(ordered_frames[-1]):
        raise ValueError(
            "The final LeRobot frame has an action but no explicit next observation. "
            "Export next_state/next_* fields or append an observation-only final frame "
            "so the terminal action can form a complete transition."
        )
    source_frames = ordered_frames[:-1]
    target_frames = ordered_frames[1:]
    for index, frame in enumerate(source_frames):
        if not _has_action(frame):
            raise ValueError(f"LeRobot source frame {index} is missing an action.")
    return source_frames, target_frames


def _validate_terminal_reward(terminal_reward: float) -> float:
    if not isinstance(terminal_reward, (int, float, np.integer, np.floating)):
        raise ValueError(
            f"terminal_reward must be a finite number; got "
            f"type={type(terminal_reward).__name__}."
        )
    terminal_reward_float = float(terminal_reward)
    if not np.isfinite(terminal_reward_float):
        raise ValueError(
            f"terminal_reward must be finite (not NaN/Inf); got {terminal_reward!r}."
        )
    return terminal_reward_float


def lerobot_episode_to_trajectory(
    frames: Sequence[dict[str, Any]],
    *,
    default_intervene: bool = True,
    model_weights_id: str = "lerobot",
    terminal_reward: float = 1.0,
    state_only: bool = False,
    num_action_chunks: int = 1,
) -> Trajectory:
    """Convert one LeRobot-style episode into a replay-buffer trajectory.

    Transition metadata such as rewards, dones, truncations, and intervention
    flags stays aligned with the source frame action. The preferred layout stores
    explicit ``next_state`` / ``next_*`` observations on each action frame. Legacy
    layouts may instead append an observation-only final frame.

    Args:
        num_action_chunks: Trailing chunk dimension to emit for
            ``loss_mask``. LeRobot frames carry one action per step, so we
            mark every chunk position as valid. This must match the training
            rollout's ``actor.model.num_action_chunks``; otherwise demo +
            rollout ``concat_batch`` raises on the trailing-dim mismatch.

    Raises:
        ValueError: If ``state``/``action``/``reward`` contain non-finite values
            or are empty, ``terminal_reward`` is non-finite, or
            ``num_action_chunks < 1``.
    """
    if num_action_chunks < 1:
        raise ValueError(
            f"num_action_chunks must be >= 1; got {num_action_chunks}."
        )
    terminal_reward = _validate_terminal_reward(terminal_reward)
    ordered_frames = sorted(frames, key=_frame_order)
    _validate_contiguous_frame_indices(ordered_frames)
    source_frames, target_frames = _resolve_transition_frames(ordered_frames)
    actions = _stack_actions(source_frames)

    terminal_metadata_targets = (
        [None for _ in source_frames]
        if target_frames is None
        else [
            None if _has_action(target_frame) else target_frame
            for target_frame in target_frames
        ]
    )
    terminal_flags = [
        _terminal_flags_from_transition(
            source_frame,
            target_frame=target_frame,
            default_done=index == len(source_frames) - 1,
        )
        for index, (source_frame, target_frame) in enumerate(
            zip(source_frames, terminal_metadata_targets, strict=True)
        )
    ]
    done_values = [done for done, _, _ in terminal_flags]
    middle_terminal_indices = [
        index for index, done in enumerate(done_values[:-1]) if done
    ]
    if middle_terminal_indices:
        raise ValueError(
            "LeRobot transition is terminal before final action frame at source "
            f"indices: {middle_terminal_indices}. Split episodes before conversion."
        )
    if (
        done_values
        and not done_values[-1]
        and (
            _has_terminal_metadata(source_frames[-1])
            or _has_terminal_metadata(terminal_metadata_targets[-1])
        )
    ):
        raise ValueError(
            "LeRobot final action frame is explicitly marked non-terminal. "
            "Set done/terminated/truncated consistently for the final transition."
        )
    termination_values = [terminated for _, terminated, _ in terminal_flags]
    truncation_values = [truncated for _, _, truncated in terminal_flags]
    done_tensor = torch.as_tensor(done_values, dtype=torch.bool).reshape(-1, 1, 1)
    truncation_tensor = torch.as_tensor(truncation_values, dtype=torch.bool).reshape(
        -1, 1, 1
    )
    termination_tensor = torch.as_tensor(termination_values, dtype=torch.bool).reshape(
        -1, 1, 1
    )

    rewards = [
        _reward_from_transition(
            source_frame,
            target_frame=target_frame,
            done=done,
            terminal_reward=terminal_reward,
        )
        for source_frame, target_frame, done in zip(
            source_frames, terminal_metadata_targets, done_values, strict=True
        )
    ]
    reward_tensor = torch.as_tensor(rewards, dtype=torch.float32).reshape(-1, 1, 1)

    intervene_values = [
        _strict_bool_from_frame(
            frame,
            ("intervene_flag",),
            default_intervene,
            field_name="intervene_flag",
        )
        for frame in source_frames
    ]
    intervene_tensor = torch.as_tensor(intervene_values, dtype=torch.bool).reshape(
        -1, 1, 1
    )
    intervene_tensor = intervene_tensor.expand_as(actions).contiguous()
    curr_obs = _build_obs(source_frames, state_only=state_only)
    next_obs = _build_obs(
        source_frames if target_frames is None else target_frames,
        state_only=state_only,
        use_next=target_frames is None,
    )
    _validate_obs_symmetry(curr_obs, next_obs)

    # LeRobot frames are single-action per step; every transition is valid.
    # Emit loss_mask with the caller-specified ``num_action_chunks`` trailing
    # dimension so demo batches concat cleanly with chunked-rollout batches
    # (which produce ``loss_mask`` shaped ``[T, B, num_action_chunks]`` via
    # ``EmbodiedRolloutResult.mark_last_step_with_loss_mask``). All chunk
    # positions are True because LeRobot has one valid action per step.
    loss_mask_tensor = torch.ones(
        actions.shape[0],
        actions.shape[1],
        num_action_chunks,
        dtype=torch.bool,
    )

    return Trajectory(
        max_episode_length=len(source_frames),
        model_weights_id=model_weights_id,
        actions=actions,
        intervene_flags=intervene_tensor,
        rewards=reward_tensor,
        terminations=termination_tensor,
        truncations=truncation_tensor,
        dones=done_tensor,
        loss_mask=loss_mask_tensor,
        forward_inputs={"action": actions.clone()},
        curr_obs=curr_obs,
        next_obs=next_obs,
    )


def _extract_episode_index(
    frame: dict[str, Any], frame_position: int
) -> int | None:
    """Return the integer ``episode_index`` for *frame* or ``None`` if absent.

    Raises ``ValueError`` for present-but-invalid values (NaN, non-integer
    float, …) so they can't silently corrupt grouping.
    """
    if "episode_index" not in frame:
        return None
    raw_value = frame["episode_index"]
    if _is_missing_value(raw_value):
        return None
    scalar_value = _scalar(raw_value)
    if scalar_value is None:
        raise ValueError(
            f"LeRobot frame at position {frame_position} has an unreadable "
            f"'episode_index' value: {raw_value!r}."
        )
    if isinstance(scalar_value, float):
        if not np.isfinite(scalar_value) or scalar_value != int(scalar_value):
            raise ValueError(
                f"LeRobot frame at position {frame_position} has a non-integer "
                f"'episode_index' value: {raw_value!r}."
            )
    return int(scalar_value)


def convert_lerobot_frames_to_trajectories(
    frames: Iterable[dict[str, Any]],
    *,
    default_intervene: bool = True,
    model_weights_id: str = "lerobot",
    terminal_reward: float = 1.0,
    state_only: bool = False,
    num_action_chunks: int = 1,
) -> list[Trajectory]:
    """Group LeRobot-style frames by episode and convert them to trajectories.

    Raises:
        ValueError: If frames have inconsistent ``episode_index`` presence
            (mixed present/missing would silently merge missing frames into
            episode 0, corrupting real episode boundaries — MAJ-9).
    """
    terminal_reward = _validate_terminal_reward(terminal_reward)
    frame_list = list(frames)
    extracted_indices: list[int | None] = [
        _extract_episode_index(frame, position)
        for position, frame in enumerate(frame_list)
    ]
    has_index = [index is not None for index in extracted_indices]
    if any(has_index) and not all(has_index):
        # Mixed presence is the dangerous case: missing frames would default
        # to bucket 0 and merge with real episode 0.
        missing_positions = [
            position
            for position, present in enumerate(has_index)
            if not present
        ]
        raise ValueError(
            "LeRobot frames have inconsistent 'episode_index' presence: "
            "some frames declare it while others do not. Mixed presence "
            "silently merges the unlabelled frames into episode 0. Missing "
            f"key at positions: {missing_positions}."
        )

    grouped_frames: dict[int, list[dict[str, Any]]] = {}
    for episode_index, frame in zip(extracted_indices, frame_list):
        # When no frame declared episode_index, treat the whole batch as a
        # single episode 0; the loader at ``load_lerobot_parquet_frames``
        # always assigns one so this is only reachable from direct callers
        # of ``convert_lerobot_frames_to_trajectories``.
        bucket = 0 if episode_index is None else episode_index
        grouped_frames.setdefault(bucket, []).append(frame)

    trajectories = []
    for episode_index in sorted(grouped_frames):
        episode_frames = grouped_frames[episode_index]
        trajectories.append(
            lerobot_episode_to_trajectory(
                episode_frames,
                default_intervene=default_intervene,
                model_weights_id=model_weights_id,
                terminal_reward=terminal_reward,
                state_only=state_only,
                num_action_chunks=num_action_chunks,
            )
        )
    _validate_lerobot_trajectory_observation_schema(trajectories)
    return trajectories


def _validate_lerobot_trajectory_observation_schema(
    trajectories: Sequence[Trajectory],
) -> None:
    if not trajectories:
        return

    for field_name in ("curr_obs", "next_obs"):
        expected_keys = set(getattr(trajectories[0], field_name).keys())
        for index, trajectory in enumerate(trajectories[1:], start=1):
            keys = set(getattr(trajectory, field_name).keys())
            if keys != expected_keys:
                missing = sorted(expected_keys - keys)
                extra = sorted(keys - expected_keys)
                changed_keys = sorted(set(missing) | set(extra))
                raise ValueError(
                    "LeRobot observation schema must be uniform across episodes; "
                    f"trajectory_id={index}, field={field_name}, "
                    f"changed_keys={changed_keys}, missing_keys={missing}, "
                    f"extra_keys={extra}."
                )


def write_lerobot_frames_to_replay_buffer(
    frames: Iterable[dict[str, Any]],
    save_path: str,
    *,
    seed: int = 1234,
    default_intervene: bool = True,
    model_weights_id: str = "lerobot",
    terminal_reward: float = 1.0,
    state_only: bool = False,
    num_action_chunks: int = 1,
) -> None:
    """Write LeRobot-style frames into a RLinf replay-buffer checkpoint."""
    trajectories = convert_lerobot_frames_to_trajectories(
        frames,
        default_intervene=default_intervene,
        model_weights_id=model_weights_id,
        terminal_reward=terminal_reward,
        state_only=state_only,
        num_action_chunks=num_action_chunks,
    )
    if not trajectories:
        raise ValueError("No valid LeRobot episodes were converted.")

    buffer = TrajectoryReplayBuffer(
        seed=seed,
        enable_cache=False,
        auto_save=True,
        auto_save_path=save_path,
        trajectory_format="pt",
    )
    try:
        buffer.add_trajectories(trajectories)
    finally:
        buffer.close()


def _find_lerobot_dataset_roots(root: Path) -> list[Path]:
    data_dir = root / "data"
    if data_dir.is_dir() and any(data_dir.glob("**/*.parquet")):
        return [root]

    dataset_roots = set()
    for parquet_file in root.glob("**/data/**/*.parquet"):
        # Walk up from the parquet file to the nearest `data` directory,
        # so nested paths like `<root>/foo/data/bar/data/chunk-000/x.parquet`
        # resolve to the innermost dataset (`<root>/foo/data/bar`).
        for ancestor in parquet_file.parents:
            if ancestor.name == "data":
                dataset_root = ancestor.parent
                if dataset_root.is_relative_to(root):
                    dataset_roots.add(dataset_root)
                break
    return sorted(dataset_roots)


def _validate_required_columns(parquet_file: Path, columns: set[str]) -> None:
    for keys in _REQUIRED_PARQUET_KEYS:
        if not any(key in columns for key in keys):
            key_names = ", ".join(keys)
            raise ValueError(f"{parquet_file} is missing required column: {key_names}")


def _read_lerobot_parquet(
    parquet_file: Path, *, state_only: bool
) -> list[dict[str, Any]]:
    import pandas as pd

    columns: set[str] | None = None
    try:
        import pyarrow.parquet as pq

        columns = set(pq.read_schema(parquet_file).names)
    except ImportError:
        columns = None
        # state_only without column projection would read the entire parquet
        # (including images) into RAM and then drop columns post-hoc, which
        # defeats the memory purpose of the mode (MAJ-13).
        if state_only:
            raise ImportError(
                "state_only=True requires pyarrow to project state columns "
                "from the parquet file without materialising image data in "
                "RAM. Install pyarrow (e.g. `pip install pyarrow`) and retry."
            )

    if columns is not None:
        _validate_required_columns(parquet_file, columns)
        if state_only:
            selected_columns = [
                column for column in _STATE_ONLY_PARQUET_KEYS if column in columns
            ]
            table = pd.read_parquet(parquet_file, columns=selected_columns)
        else:
            table = pd.read_parquet(parquet_file)
    else:
        table = pd.read_parquet(parquet_file)
        _validate_required_columns(parquet_file, set(table.columns))
    return table.to_dict("records")


def load_lerobot_parquet_frames(
    dataset_path: str, *, state_only: bool = False
) -> list[dict[str, Any]]:
    """Load frames from a local LeRobot parquet dataset directory.

    The loader accepts a single LeRobot root with parquet files under
    ``data/**`` or a collector parent such as ``collected_data/`` containing
    ``rank_*/id_*/data/**`` roots. The function intentionally avoids importing
    ``lerobot`` so it can run in lightweight environments that only have parquet
    support.
    """
    root = Path(dataset_path)
    dataset_roots = _find_lerobot_dataset_roots(root)
    if not dataset_roots:
        raise FileNotFoundError(f"No LeRobot parquet files found under {root}")

    frames: list[dict[str, Any]] = []
    next_episode_index = 0
    for dataset_root in dataset_roots:
        parquet_files = sorted((dataset_root / "data").glob("**/*.parquet"))
        # Tag every loaded record with its source parquet path so downstream
        # validators can include the file in error messages (multi-parquet
        # datasets become "bisect by hand" without it).
        root_frames: list[dict[str, Any]] = []
        for parquet_file in parquet_files:
            file_records = _read_lerobot_parquet(parquet_file, state_only=state_only)
            for record in file_records:
                record["_lerobot_parquet_file"] = str(parquet_file)
            root_frames.extend(file_records)

        episode_map: dict[int, int] = {}
        for position, frame in enumerate(sorted(root_frames, key=_frame_order)):
            local_episode_index = _extract_episode_index(frame, position)
            if local_episode_index is None:
                # Parquet schema is validated to contain ``episode_index``
                # above, so a None here means the column exists but holds an
                # explicit null — reject to avoid silent merging.
                raise ValueError(
                    "LeRobot parquet frame at position "
                    f"{position} (parquet_file={frame.get('_lerobot_parquet_file')}, "
                    f"root={dataset_root}) has a null "
                    "'episode_index'; cannot merge unlabelled frames."
                )
            if local_episode_index not in episode_map:
                episode_map[local_episode_index] = next_episode_index
                next_episode_index += 1
            converted_frame = dict(frame)
            converted_frame["_lerobot_dataset_root"] = str(dataset_root)
            converted_frame["episode_index"] = episode_map[local_episode_index]
            frames.append(converted_frame)
    return sorted(frames, key=_frame_order)


def convert_lerobot_dataset_to_replay_buffer(
    dataset_path: str,
    save_path: str,
    *,
    seed: int = 1234,
    default_intervene: bool = True,
    model_weights_id: str = "lerobot",
    terminal_reward: float = 1.0,
    state_only: bool = False,
    num_action_chunks: int = 1,
) -> None:
    """Convert a local LeRobot parquet dataset into a replay-buffer checkpoint."""
    frames = load_lerobot_parquet_frames(dataset_path, state_only=state_only)
    write_lerobot_frames_to_replay_buffer(
        frames,
        save_path,
        seed=seed,
        default_intervene=default_intervene,
        model_weights_id=model_weights_id,
        terminal_reward=terminal_reward,
        state_only=state_only,
        num_action_chunks=num_action_chunks,
    )


def _finite_float(value: str) -> float:
    """``argparse`` type that rejects NaN/Inf so they can't reach the buffer."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a valid float."
        ) from exc
    if not np.isfinite(parsed):
        raise argparse.ArgumentTypeError(
            f"{value!r} must be finite (not NaN/Inf)."
        )
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a local LeRobot parquet dataset to a RLinf replay buffer."
    )
    parser.add_argument(
        "--dataset-path", required=True, help="Local LeRobot dataset path."
    )
    parser.add_argument(
        "--save-path",
        required=True,
        help="Output TrajectoryReplayBuffer checkpoint path.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--default-intervene",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mark frames without intervene_flag as expert/intervention data.",
    )
    parser.add_argument("--model-weights-id", default="lerobot")
    parser.add_argument("--terminal-reward", type=_finite_float, default=1.0)
    parser.add_argument(
        "--state-only",
        action="store_true",
        help=(
            "Ignore image fields and convert only state/action data. Use this "
            "only when the training config uses a state-only actor model."
        ),
    )
    parser.add_argument(
        "--num-action-chunks",
        type=int,
        default=1,
        help=(
            "Trailing chunk dimension for the per-step loss_mask. MUST match "
            "the training rollout's actor.model.num_action_chunks; otherwise "
            "demo + rollout concat fails with a trailing-dim mismatch."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for local LeRobot-to-replay-buffer conversion."""
    args = _parse_args()
    convert_lerobot_dataset_to_replay_buffer(
        dataset_path=args.dataset_path,
        save_path=args.save_path,
        seed=args.seed,
        default_intervene=args.default_intervene,
        model_weights_id=args.model_weights_id,
        terminal_reward=args.terminal_reward,
        state_only=args.state_only,
        num_action_chunks=args.num_action_chunks,
    )


if __name__ == "__main__":
    main()
