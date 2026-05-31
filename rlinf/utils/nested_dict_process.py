# Copyright 2025 The RLinf Authors.
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

from typing import Any

import numpy as np
import torch

# Keys that we have already warned about in concat_batch, so each missing key
# only produces a single warning per process (avoid log spam in the replay /
# demo batch pipeline).
_CONCAT_BATCH_WARNED_KEYS: set[str] = set()


def update_nested_cfg(base_cfg, override_cfg):
    for key, value in override_cfg.items():
        if (
            key in base_cfg
            and isinstance(base_cfg[key], dict)
            and isinstance(value, dict)
        ):
            update_nested_cfg(base_cfg[key], value)
        else:
            base_cfg[key] = value
    return base_cfg


def copy_dict_tensor(next_extracted_obs: dict):
    """
    Recursively clones all torch tensors in a dict.
    """
    ret = {}
    for key, value in next_extracted_obs.items():
        if isinstance(value, torch.Tensor):
            ret[key] = value.clone()
        elif isinstance(value, dict):
            ret[key] = copy_dict_tensor(value)
        else:
            ret[key] = value
    return ret


def clone_nested_to_cpu(value: Any):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: clone_nested_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_nested_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_nested_to_cpu(item) for item in value)
    return value


def put_tensor_device(data_dict, device):
    if data_dict is None:
        return None

    if isinstance(data_dict, torch.Tensor):
        return data_dict.to(device=device).contiguous()
    for key, value in data_dict.items():
        if isinstance(value, dict):
            data_dict[key] = put_tensor_device(value, device)
        if isinstance(value, torch.Tensor):
            data_dict[key] = value.to(device=device).contiguous()
    return data_dict


def split_dict_to_chunk(data: dict, split_size, dim=0):
    splited_list = [{} for _ in range(split_size)]
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            split_vs = [
                chunk.contiguous() for chunk in torch.chunk(value, split_size, dim=dim)
            ]
        elif value is None:
            split_vs = [None for _ in range(split_size)]
        elif isinstance(value, dict):
            split_vs = split_dict_to_chunk(value, split_size, dim)
        else:
            raise ValueError(f"{key=}, {type(value)} is not supported.")
        for split_id in range(split_size):
            splited_list[split_id][key] = (
                split_vs[split_id].contiguous()
                if isinstance(split_vs[split_id], torch.Tensor)
                else split_vs[split_id]
            )
    return splited_list


# Tensor keys for which a missing side in strict-mode concat is auto-filled
# with a documented default rather than raising. ``loss_mask`` is the one
# field that legitimately appears on only one side (e.g. LeRobot demos always
# emit it; pre-existing rollouts may not) and where a default ("count every
# sample") is unambiguous.
_CONCAT_BATCH_FILLABLE_DEFAULTS: dict[str, dict] = {
    "loss_mask": {"value": True, "dtype": torch.bool},
}

# Dict-valued keys that are documented as opt-in (asymmetric presence is
# legitimate) and therefore exempt from the symmetric dict-key check in
# strict mode. ``forward_inputs`` is opt-in: LeRobot demos always populate
# ``{"action": ...}`` while many rollouts emit an empty dict that
# ``convert_trajectories_to_batch`` then omits entirely.
_CONCAT_BATCH_LENIENT_DICT_KEYS: frozenset = frozenset({"forward_inputs"})


def _peer_batch_dim(side: dict) -> int | None:
    """Return the leading (batch) dim of any tensor on ``side`` (or None).

    Used when auto-filling a fillable tensor key on the side that lacks it:
    the trailing shape comes from the OTHER side's value, but the leading
    batch dim must match this side's existing tensors so the cat stays
    consistent.
    """
    for value in side.values():
        if isinstance(value, torch.Tensor):
            return value.shape[0]
    return None


def _fill_missing_tensor_default(
    reference: torch.Tensor, spec: dict, *, batch_dim: int | None = None
) -> torch.Tensor:
    """Create a fill tensor matching ``reference``'s trailing shape.

    The leading batch dim is ``batch_dim`` when provided (so the missing
    side's batch size aligns with its other tensors) and ``reference``'s
    dim 0 otherwise.
    """
    dtype = spec.get("dtype", reference.dtype)
    shape = list(reference.shape)
    if batch_dim is not None:
        shape[0] = batch_dim
    return torch.full(
        shape,
        fill_value=spec["value"],
        dtype=dtype,
        device=reference.device,
    )


def concat_batch(data1, data2, *, strict: bool = True):
    """Concatenate two batch dicts along the batch (dim=0) dimension.

    Args:
        data1: First batch dict.  Iteration drives which keys are emitted.
        data2: Second batch dict.  Must contain the same tensor keys as
            ``data1`` when ``strict=True`` (with the exception of the
            documented fillable keys in
            :data:`_CONCAT_BATCH_FILLABLE_DEFAULTS`, currently
            ``loss_mask`` — missing side is auto-filled with an
            all-true mask rather than raising, so mixing legacy rollouts
            with new LeRobot demos still works).
        strict: When ``True`` (default), raise ``ValueError`` whenever a
            tensor key in ``data1`` is missing from ``data2`` (and vice
            versa) UNLESS the key is fillable.  When ``False``, falls back
            to legacy behaviour that skips missing keys (with a one-time
            warning per key for dict values).

    Raises:
        ValueError: If ``strict`` is ``True`` and the two dicts disagree on
            a non-fillable tensor key at any nesting level.
    """
    batch = {}
    # Strict check: BOTH tensor keys and dict keys (e.g. ``forward_inputs``)
    # in either side must be present in both, or be in the fillable set.
    # The earlier check covered only tensor keys, so a ``forward_inputs``
    # dict present on only one side could silently disappear under
    # ``strict=True`` (CRIT-R4-5).
    if strict:
        tensor_keys_1 = {
            key for key, value in data1.items() if isinstance(value, torch.Tensor)
        }
        tensor_keys_2 = {
            key for key, value in data2.items() if isinstance(value, torch.Tensor)
        }
        dict_keys_1 = {
            key for key, value in data1.items() if isinstance(value, dict)
        }
        dict_keys_2 = {
            key for key, value in data2.items() if isinstance(value, dict)
        }
        fillable = set(_CONCAT_BATCH_FILLABLE_DEFAULTS.keys())
        # ``forward_inputs`` (and any future opt-in dict) is exempt from the
        # symmetric check; the remaining ``curr_obs`` / ``next_obs`` cases
        # MUST be uniform or silent observation-schema drift becomes
        # possible.
        missing_in_data2 = sorted(tensor_keys_1 - tensor_keys_2 - fillable)
        missing_in_data1 = sorted(tensor_keys_2 - tensor_keys_1 - fillable)
        dict_missing_in_data2 = sorted(
            (dict_keys_1 - dict_keys_2) - _CONCAT_BATCH_LENIENT_DICT_KEYS
        )
        dict_missing_in_data1 = sorted(
            (dict_keys_2 - dict_keys_1) - _CONCAT_BATCH_LENIENT_DICT_KEYS
        )
        if (
            missing_in_data2
            or missing_in_data1
            or dict_missing_in_data2
            or dict_missing_in_data1
        ):
            raise ValueError(
                "concat_batch: keys disagree between the two batches "
                "(data1 / data2 are the two arguments passed to concat_batch); "
                f"tensor keys only in data1={missing_in_data2}, "
                f"tensor keys only in data2={missing_in_data1}, "
                f"dict keys only in data1={dict_missing_in_data2}, "
                f"dict keys only in data2={dict_missing_in_data1}. "
                "Producers must emit the same set of keys on both sides, or "
                "extend _CONCAT_BATCH_FILLABLE_DEFAULTS if a missing key has "
                "a safe documented default. (Passing strict=False silently "
                "drops missing keys and is only for legacy code paths.)"
            )

    for key, value in data1.items():
        if isinstance(value, torch.Tensor):
            if key not in data2:
                if strict and key in _CONCAT_BATCH_FILLABLE_DEFAULTS:
                    # Auto-fill data2's missing side with a tensor whose
                    # batch dim matches data2's peer tensors and whose
                    # trailing dims match data1's reference shape.
                    fill = _fill_missing_tensor_default(
                        value,
                        _CONCAT_BATCH_FILLABLE_DEFAULTS[key],
                        batch_dim=_peer_batch_dim(data2),
                    )
                    batch[key] = torch.cat([value, fill], dim=0)
                # Otherwise (strict=False) silently skip.
                continue
            batch[key] = torch.cat([data1[key], data2[key]], dim=0)
        elif isinstance(value, dict):
            # NOTE: added this for dealing with different keys in demo data.
            if key not in data2:
                if key not in _CONCAT_BATCH_WARNED_KEYS:
                    _CONCAT_BATCH_WARNED_KEYS.add(key)
                    # Lazy import to avoid pulling rlinf.scheduler.worker (and
                    # its heavy deps) at module import time. This only runs
                    # once per missing key, inside a worker where that import
                    # is essentially free.
                    from rlinf.utils.logging import get_logger

                    get_logger().warning(
                        "concat_batch: key '%s' not found in data2 (value type: %s), "
                        "skipping. This warning is only emitted once per key.",
                        key,
                        type(value).__name__,
                    )
                continue
            batch[key] = concat_batch(data1[key], data2[key], strict=strict)

    # Symmetric pass for fillable tensor keys that exist only in data2.
    if strict:
        for key, value in data2.items():
            if (
                isinstance(value, torch.Tensor)
                and key not in data1
                and key in _CONCAT_BATCH_FILLABLE_DEFAULTS
            ):
                fill = _fill_missing_tensor_default(
                    value,
                    _CONCAT_BATCH_FILLABLE_DEFAULTS[key],
                    batch_dim=_peer_batch_dim(data1),
                )
                batch[key] = torch.cat([fill, value], dim=0)
    return batch


def stack_list_of_dict_tensor(list_of_dict: list, dim=0):
    if len(list_of_dict) == 0:
        return {}
    keys = list_of_dict[0].keys()

    ret = {}
    for key in keys:
        _v0 = list_of_dict[0][key]
        if isinstance(_v0, torch.Tensor):
            v_list = [d[key] for d in list_of_dict]
            ret[key] = torch.stack(v_list, dim=dim)
        elif isinstance(_v0, dict):
            v_list = [d[key] for d in list_of_dict]
            ret[key] = stack_list_of_dict_tensor(v_list)
        elif _v0 is None:
            pass
        else:
            raise ValueError(f"{key=}, {type(_v0)} is not supported!")
    return ret


def cat_list_of_dict_tensor(list_of_dict: list, dim=0):
    if len(list_of_dict) == 0:
        return {}
    keys = list_of_dict[0].keys()

    ret = {}
    for key in keys:
        _v0 = list_of_dict[0][key]
        if _v0 is None:
            continue

        v_list = [d[key] for d in list_of_dict]

        if isinstance(_v0, torch.Tensor):
            ret[key] = torch.cat(v_list, dim=dim)
        elif isinstance(_v0, np.ndarray):
            ret[key] = np.concatenate([v for v in v_list if v is not None], axis=dim)
        elif isinstance(_v0, list):
            assert dim == 0, f"{key=} is list, dim !=0 is not supported!"
            ret[key] = [item for sub in v_list if sub is not None for item in sub]
        elif isinstance(_v0, dict):
            ret[key] = cat_list_of_dict_tensor(v_list, dim=dim)
        else:
            raise ValueError(f"{key=}, {type(_v0)} is not supported!")

    return ret


def split_dict(
    batch: dict[str, Any],
    split_sizes: list[int],
) -> list[dict[str, Any]]:
    """Split one batch dict into size-specified sub-batches along dim-0.

    Tensor values are chunked on dim-0; list values are sliced proportionally;
    nested dict values are split recursively.

    Args:
        batch: Dict.
        split_sizes: Batch sizes for each destination rank.

    Returns:
        A list of splited batches, one item per destination rank.
    """
    count = len(split_sizes)
    total_size = sum(split_sizes)
    splitted_batches = [{} for _ in range(count)]
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            assert value.shape[0] == total_size, (
                f"Tensor field '{key}' expected batch size {total_size}, got {value.shape[0]}."
            )
            splitted_values = torch.split(value, split_sizes, dim=0)
            for i in range(count):
                splitted_batches[i][key] = splitted_values[i].contiguous()
        elif isinstance(value, list):
            length = len(value)
            assert length == total_size, (
                f"List field '{key}' expected length {total_size}, got {length}."
            )
            begin = 0
            for i, size in enumerate(split_sizes):
                splitted_batches[i][key] = value[begin : begin + size]
                begin += size
        elif isinstance(value, dict):
            splitted_sub_batches = split_dict(value, split_sizes)
            for i in range(count):
                splitted_batches[i][key] = splitted_sub_batches[i]
        else:
            for i in range(count):
                splitted_batches[i][key] = value

    return splitted_batches
