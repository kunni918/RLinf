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

"""Lightweight SAC demo-buffer validation and chunk-mask helpers."""

from __future__ import annotations

from typing import Any, Optional

import torch
from omegaconf import DictConfig

from rlinf.config import SupportedModel, validate_demo_buffer_load_mode


def chunk_sample_weights(
    batch: dict, dtype: torch.dtype
) -> Optional[torch.Tensor]:
    """Per-sample ``[bsz, 1]`` weight derived from ``batch['loss_mask']``.

    A chunk sample is fully valid (weight 1.0) only when every sub-step of
    the chunk is valid; chunks padded by auto-reset are excluded
    (weight 0.0). Returns ``None`` when no mask is present so callers keep
    their existing unweighted reduction.
    """
    mask = batch.get("loss_mask")
    if mask is None:
        return None
    return mask.all(dim=-1, keepdim=True).to(dtype=dtype)


def apply_chunk_mask(
    per_sample_loss: torch.Tensor, weights: Optional[torch.Tensor]
) -> torch.Tensor:
    """Reduce ``per_sample_loss`` to a scalar, weighting by ``weights`` if
    provided (else a plain mean). When all weights are zero the result is
    ``0`` rather than ``NaN`` so the optimizer step is a no-op."""
    if weights is None:
        return per_sample_loss.mean()
    feature_size = per_sample_loss.numel() // per_sample_loss.shape[0]
    denom = (weights.sum() * feature_size).clamp(min=1.0)
    return (per_sample_loss * weights).sum() / denom

_SAC_REQUIRED_TRAJECTORY_FIELDS = (
    "actions",
    "rewards",
    "terminations",
    "truncations",
    "dones",
)


def required_demo_obs_keys(model_cfg: DictConfig) -> set[str]:
    required_keys = {"states"}
    model_type = str(model_cfg.get("model_type"))
    requires_images = model_type == SupportedModel.CNN_POLICY.value
    if model_type == SupportedModel.FLOW_POLICY.value:
        requires_images = model_cfg.get("input_type", "mixed") != "state"
    elif model_type == SupportedModel.OPENPI.value:
        requires_images = True

    if requires_images:
        required_keys.add("main_images")
        if int(model_cfg.get("image_num", 1)) > 1:
            required_keys.add("extra_view_images")
    return required_keys


def demo_buffer_load_kwargs(
    demo_cfg: DictConfig, *, rank: int, world_size: int
) -> dict[str, Any]:
    load_mode = validate_demo_buffer_load_mode(demo_cfg)
    if load_mode == "replicate":
        return {"is_distributed": False}
    return {
        "is_distributed": True,
        "local_rank": rank,
        "world_size": world_size,
    }


def _small_shard_hint(load_mode: str) -> str:
    if load_mode == "replicate":
        return "Use more demo trajectories or lower min_buffer_size."
    return (
        "Use more demo trajectories, lower min_buffer_size, or set "
        "algorithm.demo_buffer.load_mode: replicate for small demo sets."
    )


def _field_shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(dim) for dim in shape)


def _validate_image_obs_shape(
    *,
    key: str,
    obs_shape: tuple[int, ...],
    expected_extra_view_count: int | None,
    load_path: str,
    rank: int,
    trajectory_id: int,
    obs_name: str,
) -> None:
    if key == "main_images":
        if len(obs_shape) != 5:
            raise ValueError(
                "Loaded demo_buffer main_images has invalid image shape: "
                f"path={load_path}, rank={rank}, trajectory_id={trajectory_id}, "
                f"{obs_name}.{key}, expected [T, B, H, W, C], shape={obs_shape}."
            )
    elif key == "extra_view_images":
        if len(obs_shape) != 6:
            raise ValueError(
                "Loaded demo_buffer extra_view_images has invalid image shape: "
                f"path={load_path}, rank={rank}, trajectory_id={trajectory_id}, "
                f"{obs_name}.{key}, expected [T, B, N, H, W, C], shape={obs_shape}."
            )
        actual_extra_view_count = obs_shape[2]
        if actual_extra_view_count != expected_extra_view_count:
            raise ValueError(
                "Loaded demo_buffer extra_view_images do not match actor "
                "image_num: "
                f"path={load_path}, rank={rank}, trajectory_id={trajectory_id}, "
                f"{obs_name}.{key}, expected extra views from image_num="
                f"{expected_extra_view_count}, shape={obs_shape}."
            )


def _expected_action_shapes(model_cfg: DictConfig) -> set[tuple[int, ...]]:
    model_type = str(model_cfg.get("model_type"))
    if model_type == SupportedModel.OPENPI.value:
        openpi_cfg = model_cfg.get("openpi", {})
        noise_dim = openpi_cfg.get("dsrl_action_noise_dim", None)
        if noise_dim is None:
            return set()
        action_chunk = openpi_cfg.get(
            "action_chunk", model_cfg.get("num_action_chunks", None)
        )
        expected = {(int(noise_dim),)}
        if action_chunk is not None:
            expected.add((int(action_chunk), int(noise_dim)))
        return expected

    action_dim = model_cfg.get("action_dim", None)
    return {(int(action_dim),)} if action_dim is not None else set()


def _expected_state_dim(model_cfg: DictConfig) -> int | None:
    model_type = str(model_cfg.get("model_type"))
    if model_type == SupportedModel.OPENPI.value:
        dsrl_state_dim = model_cfg.get("openpi", {}).get("dsrl_state_dim", None)
        return int(dsrl_state_dim) if dsrl_state_dim is not None else None
    state_dim = model_cfg.get("state_dim", model_cfg.get("obs_dim", None))
    return int(state_dim) if state_dim is not None else None


def _validate_required_trajectory_fields(
    trajectory: Any,
    *,
    load_path: str,
    rank: int,
    trajectory_id: int,
    model_cfg: DictConfig,
) -> tuple[int, int]:
    missing_fields = [
        field
        for field in _SAC_REQUIRED_TRAJECTORY_FIELDS
        if getattr(trajectory, field, None) is None
    ]
    if missing_fields:
        raise ValueError(
            "Loaded demo_buffer trajectory is missing SAC fields: "
            f"path={load_path}, rank={rank}, trajectory_id={trajectory_id}, "
            f"missing_fields={missing_fields}."
        )

    action_shape = _field_shape(trajectory.actions)
    if action_shape is None or len(action_shape) < 3:
        raise ValueError(
            "Loaded demo_buffer trajectory has invalid actions shape: "
            f"path={load_path}, rank={rank}, trajectory_id={trajectory_id}, "
            f"shape={action_shape}. Expected actions with [T, B, ...] prefix."
        )
    expected_prefix = action_shape[:2]
    expected_action_shapes = _expected_action_shapes(model_cfg)
    if expected_action_shapes:
        actual_action_shape = action_shape[2:] if len(action_shape) >= 3 else ()
        if actual_action_shape not in expected_action_shapes:
            raise ValueError(
                "Loaded demo_buffer trajectory actions do not match actor action_dim: "
                f"path={load_path}, rank={rank}, trajectory_id={trajectory_id}, "
                f"expected_trailing_shape={sorted(expected_action_shapes)}, "
                f"actual_shape={action_shape}."
            )
    expected_T, expected_B = expected_prefix
    # dones/terminations/truncations may carry one extra entry per rollout
    # epoch (see EmbodiedRolloutResult and replay_buffer._flatten_trajectory),
    # so their first dim can legitimately exceed `expected_T` by a divisor.
    fields_with_epoch_padding = {"dones", "terminations", "truncations"}
    for field in _SAC_REQUIRED_TRAJECTORY_FIELDS[1:]:
        field_shape = _field_shape(getattr(trajectory, field))
        valid = (
            field_shape is not None
            and len(field_shape) >= 2
            and field_shape[1] == expected_B
        )
        if valid:
            if field in fields_with_epoch_padding:
                extra = field_shape[0] - expected_T
                valid = extra == 0 or (extra > 0 and expected_T % extra == 0)
            else:
                valid = field_shape[0] == expected_T
        if not valid:
            raise ValueError(
                "Loaded demo_buffer trajectory field has inconsistent [T, B] "
                "prefix: "
                f"path={load_path}, rank={rank}, trajectory_id={trajectory_id}, "
                f"field={field}, expected_prefix={expected_prefix}, "
                f"shape={field_shape}."
            )
    return expected_prefix


def _validate_obs_schema(
    obs: dict[str, Any],
    *,
    obs_name: str,
    required_obs_keys: set[str],
    load_path: str,
    rank: int,
    trajectory_id: int,
    model_type: str,
    expected_state_dim: int | None,
    expected_extra_view_count: int | None,
    expected_prefix: tuple[int, int],
) -> None:
    missing_keys = [
        key for key in required_obs_keys if key not in obs or obs[key] is None
    ]
    if missing_keys:
        available_obs_keys = sorted(
            key for key, value in obs.items() if value is not None
        )
        raise ValueError(
            "Loaded demo_buffer is incompatible with actor observation schema: "
            f"path={load_path}, rank={rank}, trajectory_id={trajectory_id}, "
            f"model_type={model_type}, {obs_name} missing {missing_keys}, "
            f"required_obs_keys={sorted(required_obs_keys)}, "
            f"available_obs_keys={available_obs_keys}. Convert image demos without "
            "--state-only or use a state-only actor model/config."
        )
    for key in required_obs_keys:
        obs_shape = _field_shape(obs[key])
        if obs_shape is None or len(obs_shape) < 2 or obs_shape[:2] != expected_prefix:
            raise ValueError(
                "Loaded demo_buffer observation field has inconsistent [T, B] "
                "prefix: "
                f"path={load_path}, rank={rank}, trajectory_id={trajectory_id}, "
                f"{obs_name}.{key}, expected_prefix={expected_prefix}, "
                f"shape={obs_shape}."
            )
        if key == "states" and expected_state_dim is not None:
            actual_state_shape = obs_shape[2:] if len(obs_shape) >= 3 else ()
            if actual_state_shape != (expected_state_dim,):
                raise ValueError(
                    "Loaded demo_buffer observation states do not match actor "
                    "obs_dim/state_dim: "
                    f"path={load_path}, rank={rank}, trajectory_id={trajectory_id}, "
                    f"{obs_name}.{key}, expected obs_dim/state_dim="
                    f"{expected_state_dim}, shape={obs_shape}."
                )
        if key in {"main_images", "extra_view_images"}:
            _validate_image_obs_shape(
                key=key,
                obs_shape=obs_shape,
                expected_extra_view_count=expected_extra_view_count,
                load_path=load_path,
                rank=rank,
                trajectory_id=trajectory_id,
                obs_name=obs_name,
            )


def validate_loaded_demo_buffer(
    demo_buffer: Any,
    *,
    load_path: str,
    min_demo_buffer_size: int,
    rank: int,
    world_size: int,
    model_cfg: DictConfig,
    load_mode: str,
) -> None:
    if demo_buffer is None:
        return
    if min_demo_buffer_size > 0 and not demo_buffer.is_ready(min_demo_buffer_size):
        raise ValueError(
            "Loaded demo_buffer shard is smaller than min_buffer_size: "
            f"path={load_path}, rank={rank}, world_size={world_size}, "
            f"loaded_trajectories={len(demo_buffer)}, "
            f"min_buffer_size={min_demo_buffer_size}. {_small_shard_hint(load_mode)}"
        )

    if not demo_buffer._trajectory_id_list:
        raise ValueError(
            "Loaded demo_buffer shard is empty: "
            f"path={load_path}, rank={rank}, world_size={world_size}. "
            f"{_small_shard_hint(load_mode)}"
        )

    image_num = int(model_cfg.get("image_num", 1))
    if image_num < 1:
        raise ValueError(
            "Loaded demo_buffer actor image_num must be >= 1: "
            f"path={load_path}, rank={rank}, image_num={image_num}."
        )
    required_obs_keys = required_demo_obs_keys(model_cfg)
    model_type = str(model_cfg.model_type)
    expected_state_dim = _expected_state_dim(model_cfg)
    expected_extra_view_count = (
        image_num - 1 if "extra_view_images" in required_obs_keys else None
    )
    for trajectory_id in demo_buffer._trajectory_id_list:
        model_weights_id = demo_buffer._trajectory_index[trajectory_id][
            "model_weights_id"
        ]
        trajectory = demo_buffer._load_trajectory(trajectory_id, model_weights_id)
        expected_prefix = _validate_required_trajectory_fields(
            trajectory,
            load_path=load_path,
            rank=rank,
            trajectory_id=trajectory_id,
            model_cfg=model_cfg,
        )
        for obs_name in ("curr_obs", "next_obs"):
            obs = getattr(trajectory, obs_name, None)
            if not isinstance(obs, dict):
                raise ValueError(
                    "Loaded demo_buffer trajectory is missing observation dict: "
                    f"path={load_path}, rank={rank}, trajectory_id={trajectory_id}, "
                    f"field={obs_name}."
                )
            _validate_obs_schema(
                obs,
                obs_name=obs_name,
                required_obs_keys=required_obs_keys,
                load_path=load_path,
                rank=rank,
                trajectory_id=trajectory_id,
                model_type=model_type,
                expected_state_dim=expected_state_dim,
                expected_extra_view_count=expected_extra_view_count,
                expected_prefix=expected_prefix,
            )
