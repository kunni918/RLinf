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

"""Import-safe RealWorld chunk stepping helpers."""

import copy

import numpy as np
import torch


def _mask_inactive_actions(actions, valid_step: torch.Tensor):
    if valid_step.all():
        return actions
    invalid = ~valid_step.cpu()
    if isinstance(actions, torch.Tensor):
        masked_actions = actions.clone()
        masked_actions[invalid.to(device=masked_actions.device)] = 0
        return masked_actions
    masked_actions = np.array(actions, copy=True)
    masked_actions[invalid.numpy()] = 0
    return masked_actions


def _zero_invalid_env_values(value: torch.Tensor, valid_step: torch.Tensor):
    invalid = ~valid_step.to(device=value.device, dtype=torch.bool)
    if not invalid.any():
        return value
    value = value.clone()
    value[invalid] = 0
    return value


def _mask_invalid_info_values(infos: dict, valid_step: torch.Tensor) -> dict:
    invalid = ~valid_step.cpu()
    if not invalid.any():
        return infos
    infos = dict(infos)
    if "intervene_action" in infos:
        intervene_action = infos["intervene_action"]
        if isinstance(intervene_action, torch.Tensor):
            intervene_action = intervene_action.clone()
            intervene_action[invalid.to(device=intervene_action.device)] = 0
        else:
            intervene_action = np.array(intervene_action, copy=True)
            intervene_action[invalid.numpy()] = 0
        infos["intervene_action"] = intervene_action
    if "intervene_flag" in infos:
        intervene_flag = infos["intervene_flag"]
        if isinstance(intervene_flag, torch.Tensor):
            intervene_flag = intervene_flag.clone()
            intervene_flag[invalid.to(device=intervene_flag.device)] = False
        else:
            intervene_flag = np.array(intervene_flag, copy=True)
            intervene_flag[invalid.numpy()] = False
        infos["intervene_flag"] = intervene_flag
    return infos


def run_realworld_chunk_step(env, chunk_actions):
    """Run a fixed-size RealWorld action chunk with invalid-tail padding."""
    chunk_size = chunk_actions.shape[1]
    obs_list = []
    infos_list = []
    chunk_rewards = []
    raw_chunk_terminations = []
    raw_chunk_truncations = []
    raw_chunk_intervene_actions = []
    raw_chunk_intervene_flag = []
    valid_step_masks = []
    active_step_mask = torch.ones(env.num_envs, dtype=torch.bool)

    for step_idx in range(chunk_size):
        valid_step = active_step_mask.clone()
        actions = _mask_inactive_actions(chunk_actions[:, step_idx], valid_step)
        extracted_obs, step_reward, terminations, truncations, infos = env.step(
            actions, auto_reset=False
        )
        step_reward = _zero_invalid_env_values(step_reward, valid_step)
        terminations = _zero_invalid_env_values(terminations, valid_step).to(
            dtype=torch.bool
        )
        truncations = _zero_invalid_env_values(truncations, valid_step).to(
            dtype=torch.bool
        )
        infos = _mask_invalid_info_values(infos, valid_step)
        infos["_valid_step"] = valid_step
        if "intervene_action" in infos:
            raw_chunk_intervene_actions.append(infos["intervene_action"])
            raw_chunk_intervene_flag.append(infos["intervene_flag"])
        dones = torch.logical_or(terminations, truncations) & valid_step.to(
            device=terminations.device, dtype=torch.bool
        )
        if dones.any() and env.auto_reset:
            infos["terminal_chunk_index"] = torch.full(
                (env.num_envs,), step_idx, dtype=torch.long
            )
            extracted_obs, infos = env._handle_auto_reset(
                dones.cpu().numpy(), extracted_obs, infos
            )
            infos["_valid_step"] = valid_step

        obs_list.append(extracted_obs)
        infos_list.append(infos)
        chunk_rewards.append(step_reward)
        raw_chunk_terminations.append(terminations)
        raw_chunk_truncations.append(truncations)
        valid_step_masks.append(valid_step)

        # Narrow active envs whenever any env terminates so that subsequent
        # substeps are treated as padding. Without this, when env.auto_reset is
        # False the loop would keep calling env.step on already-done envs and
        # record post-terminal frames as legitimate transitions.
        if dones.any():
            active_step_mask = active_step_mask & ~dones.cpu()
        if not active_step_mask.any():
            break

    if obs_list and len(obs_list) < chunk_size:
        pad_count = chunk_size - len(obs_list)
        zero_reward = torch.zeros_like(chunk_rewards[-1])
        zero_termination = torch.zeros_like(raw_chunk_terminations[-1])
        zero_truncation = torch.zeros_like(raw_chunk_truncations[-1])
        zero_valid_step = torch.zeros(env.num_envs, dtype=torch.bool)
        zero_intervene_action = (
            torch.zeros_like(raw_chunk_intervene_actions[-1])
            if raw_chunk_intervene_actions
            else None
        )
        zero_intervene_flag = (
            torch.zeros_like(raw_chunk_intervene_flag[-1], dtype=torch.bool)
            if raw_chunk_intervene_flag
            else None
        )
        for _ in range(pad_count):
            obs_list.append(copy.deepcopy(obs_list[-1]))
            infos_list.append({"_valid_step": zero_valid_step.clone()})
            chunk_rewards.append(zero_reward.clone())
            raw_chunk_terminations.append(zero_termination.clone())
            raw_chunk_truncations.append(zero_truncation.clone())
            valid_step_masks.append(zero_valid_step.clone())
            if zero_intervene_action is not None and zero_intervene_flag is not None:
                raw_chunk_intervene_actions.append(zero_intervene_action.clone())
                raw_chunk_intervene_flag.append(zero_intervene_flag.clone())

    chunk_rewards = torch.stack(chunk_rewards, dim=1)
    raw_chunk_terminations = torch.stack(raw_chunk_terminations, dim=1)
    raw_chunk_truncations = torch.stack(raw_chunk_truncations, dim=1)

    infos_last = infos_list[-1] if infos_list else {}
    if valid_step_masks:
        infos_last["chunk_valid_step"] = torch.stack(valid_step_masks, dim=1)
    if raw_chunk_intervene_actions:
        infos_last["chunk_intervene_action"] = torch.stack(
            raw_chunk_intervene_actions, dim=1
        ).reshape(env.num_envs, -1)
        infos_last["chunk_intervene_flag"] = torch.stack(
            raw_chunk_intervene_flag, dim=1
        )
        infos_list[-1] = infos_last

    return (
        obs_list,
        chunk_rewards,
        raw_chunk_terminations.clone(),
        raw_chunk_truncations.clone(),
        infos_list,
    )
