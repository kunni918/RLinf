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

import asyncio
import copy
import gc
from collections import defaultdict
from typing import Any, Literal

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    EnvOutput,
    RolloutResult,
    Trajectory,
)
from rlinf.envs import get_env_cls
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.wrappers import RecordVideo
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.comm_mapping import CommMapper
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.nested_dict_process import (
    clone_nested_to_cpu,
    copy_dict_tensor,
    split_dict,
    update_nested_cfg,
)
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.history_manager import HistoryManager


class EnvWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.train_video_cnt = 0
        self.eval_video_cnt = 0
        self.should_stop = False

        self.env_list = []
        self.eval_env_list = []

        self.last_obs_list = []
        self.last_intervened_info_list = []
        self._prefetched_train_bootstrap: list[EnvOutput] | None = None
        self.rollout_epoch = self.cfg.algorithm.get("rollout_epoch", 1)
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        self.collect_transitions = self.cfg.rollout.get("collect_transitions", False)
        self.collect_prev_infos = self.cfg.rollout.get("collect_prev_infos", True)
        self.stage_num = self.cfg.rollout.pipeline_stage_num

        self.reward_mode = self.cfg.get("reward", {}).get("reward_mode", "per_step")
        self.history_reward_assign = self.cfg.get("reward", {}).get(
            "history_reward_assign", False
        )
        self.use_reward_model = self.cfg.get("reward", {}).get(
            "use_reward_model", False
        )
        self.use_realworld_reward = self.cfg.get("reward", {}).get(
            "standalone_realworld", False
        )
        self.use_external_reward_model = (
            self.use_reward_model and not self.use_realworld_reward
        )
        self.env_infos_reward_keys = ("success", "episode", "final_info")
        if self.use_external_reward_model:
            self.reward_weight = self.cfg.reward.get("reward_weight", 1.0)
            self.env_reward_weight = self.cfg.reward.get("env_reward_weight", 0.0)

        # Env configurations
        self.only_eval = getattr(self.cfg.runner, "only_eval", False)
        train_env_cfg = self.cfg.env.get("train", None)
        eval_env_cfg = self.cfg.env.eval
        self.enable_offload = (
            train_env_cfg.get("enable_offload", False)
            if train_env_cfg is not None
            else eval_env_cfg.get("enable_offload", False)
        )
        self.enable_eval = self.cfg.runner.val_check_interval > 0 or self.only_eval
        if not self.only_eval:
            if train_env_cfg is None:
                raise ValueError(
                    "env.train config is required when runner.only_eval=False."
                )
            self.train_num_envs_per_stage = (
                self.cfg.env.train.total_num_envs // self._world_size // self.stage_num
            )
        if self.enable_eval:
            self.eval_num_envs_per_stage = (
                self.cfg.env.eval.total_num_envs // self._world_size // self.stage_num
            )
        self.n_train_chunk_steps = 0
        if not self.only_eval:
            self.n_train_chunk_steps = (
                self.cfg.env.train.max_steps_per_rollout_epoch
                // self.cfg.actor.model.num_action_chunks
            )
        self.n_eval_chunk_steps = (
            self.cfg.env.eval.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        self.actor_split_num = self.get_actor_split_num()

        if not self.only_eval:
            self.train_prev_done: list[torch.Tensor] = [
                torch.zeros(self.train_num_envs_per_stage, dtype=torch.bool)
                for _ in range(self.stage_num)
            ]
        if self.enable_eval:
            self.eval_prev_done: list[torch.Tensor] = [
                torch.zeros(self.eval_num_envs_per_stage, dtype=torch.bool)
                for _ in range(self.stage_num)
            ]

    def init_worker(self):
        self.dst_rank_map = self._setup_dst_rank_map()
        self.src_rank_map = self._setup_src_rank_map()

        self.log_info(f"Env worker initialized with dst_rank_map: {self.dst_rank_map}")
        self.log_info(f"Env worker initialized with src_rank_map: {self.src_rank_map}")

        # This is a barrier to ensure all envs' initial setup upon import is done
        # Essential for RealWorld env to ensure initial ROS node setup is done
        self.broadcast(
            True,
            groups=[(self._group_name, list(range(self._world_size)))],
        )

        self.update_env_cfg()

        if not self.only_eval:
            train_env_cls = get_env_cls(self.cfg.env.train.env_type, self.cfg.env.train)
            self.env_list = self._setup_env_and_wrappers(
                env_cls=train_env_cls,
                env_cfg=self.cfg.env.train,
                num_envs_per_stage=self.train_num_envs_per_stage,
            )
        if self.enable_eval:
            eval_env_cls = get_env_cls(self.cfg.env.eval.env_type, self.cfg.env.eval)
            self.eval_env_list = self._setup_env_and_wrappers(
                env_cls=eval_env_cls,
                env_cfg=self.cfg.env.eval,
                num_envs_per_stage=self.eval_num_envs_per_stage,
            )

        if not self.only_eval:
            self._init_env()
            if self.reward_mode == "history_buffer":
                self.train_history_managers = [
                    HistoryManager(self.cfg.reward, self.train_num_envs_per_stage)
                    for _ in range(self.stage_num)
                ]
                self.history_lengths = [{} for _ in range(self.stage_num)]

    def update_env_cfg(self):
        if not self.only_eval:
            # train env
            train_override_cfgs = self.cfg.env.train.get("override_cfgs", None)
            if train_override_cfgs is not None:
                assert len(train_override_cfgs) > self._rank, (
                    f"{len(train_override_cfgs)=} > {self._rank=}"
                )

                general_train_override_cfg = OmegaConf.to_container(
                    self.cfg.env.train.get("override_cfg", {}), resolve=True
                )
                override_cfg = OmegaConf.to_container(
                    train_override_cfgs[self._rank], resolve=True
                ).copy()

                base_cfg = {}
                base_cfg = update_nested_cfg(base_cfg, general_train_override_cfg)
                base_cfg = update_nested_cfg(base_cfg, override_cfg)
                setattr(self.cfg.env.train, "override_cfg", OmegaConf.create(base_cfg))
        self._inject_realworld_reward_cfg(self.cfg.env.train)
        eval_override_cfgs = self.cfg.env.eval.get("override_cfgs", None)
        if eval_override_cfgs is not None:
            assert len(eval_override_cfgs) > self._rank, (
                f"{len(eval_override_cfgs)=} > {self._rank=}"
            )

            general_eval_override_cfg = OmegaConf.to_container(
                self.cfg.env.eval.get("override_cfg", {}), resolve=True
            )
            eval_override_cfg = OmegaConf.to_container(
                eval_override_cfgs[self._rank], resolve=True
            ).copy()
            base_eval_cfg = {}
            base_eval_cfg = update_nested_cfg(base_eval_cfg, general_eval_override_cfg)
            base_eval_cfg = update_nested_cfg(base_eval_cfg, eval_override_cfg)
            setattr(self.cfg.env.eval, "override_cfg", OmegaConf.create(base_eval_cfg))
        self._inject_realworld_reward_cfg(self.cfg.env.eval)

    def _inject_realworld_reward_cfg(self, env_cfg: DictConfig):
        if not (self.use_reward_model and self.use_realworld_reward):
            return
        if env_cfg.env_type != "realworld":
            return

        reward_placements = self._component_placement.get_strategy(
            "reward"
        ).get_placement(Cluster())
        assert len(reward_placements) > 0, (
            "Reward placement must contain at least one worker."
        )
        reward_placement = reward_placements[0]
        reward_hardware_ranks = self._component_placement.get_hardware_ranks("reward")
        assert len(reward_hardware_ranks) > 0, (
            "Reward placement must contain at least one hardware rank."
        )

        override_cfg = OmegaConf.to_container(
            env_cfg.get("override_cfg", {}), resolve=True
        )
        override_cfg["use_reward_model"] = True
        override_cfg["reward_worker_cfg"] = OmegaConf.to_container(
            self.cfg.reward, resolve=True
        )
        override_cfg["reward_worker_hardware_rank"] = reward_hardware_ranks[0]
        override_cfg["reward_worker_node_rank"] = reward_placement.cluster_node_rank
        override_cfg["reward_worker_node_group"] = reward_placement.node_group_label
        override_cfg["reward_image_key"] = env_cfg.main_image_key
        setattr(env_cfg, "override_cfg", OmegaConf.create(override_cfg))

    def _setup_env_and_wrappers(self, env_cls, env_cfg, num_envs_per_stage: int):
        env_list = []

        for stage_id in range(self.stage_num):
            env = env_cls(
                cfg=env_cfg,
                num_envs=num_envs_per_stage,
                seed_offset=self._rank * self.stage_num + stage_id,
                total_num_processes=self._world_size * self.stage_num,
                worker_info=self.worker_info,
            )
            if env_cfg.video_cfg.save_video:
                env = RecordVideo(env, env_cfg.video_cfg)
            if env_cfg.get("data_collection", None) and getattr(
                env_cfg.data_collection, "enabled", False
            ):
                from rlinf.envs.wrappers import CollectEpisode

                env = CollectEpisode(
                    env,
                    save_dir=env_cfg.data_collection.save_dir,
                    rank=self._rank,
                    num_envs=num_envs_per_stage,
                    export_format=getattr(
                        env_cfg.data_collection, "export_format", "pickle"
                    ),
                    robot_type=getattr(env_cfg.data_collection, "robot_type", "panda"),
                    fps=getattr(env_cfg.data_collection, "fps", 10),
                    only_success=getattr(
                        env_cfg.data_collection, "only_success", False
                    ),
                    finalize_interval=getattr(
                        env_cfg.data_collection, "finalize_interval", 100
                    ),
                )
            env_list.append(env)
        return env_list

    def _setup_dst_rank_map(self) -> dict[str, list[tuple[int, int]]]:
        """Compute destination rank map for this env worker.

        This mapping supports both one-to-many and many-to-one env/rollout/reward layouts.
        The returned ranks are used as communication counterparts for both sending
        env outputs and receiving results from rollout and reward workers.

        Returns:
            Destination rank map for this env worker.
            The key is the channel name (e.g. "rollout_train", "reward_train", "rollout_eval"), and the value is a ordered list of tuples of (dst_rank, batch_size).
        """
        dst_rank_map = {}
        if not self.only_eval:
            dst_rank_map = {
                "rollout_train": CommMapper.get_dst_ranks(
                    batch_size=self.cfg.env.train.total_num_envs // self.stage_num,
                    src_world_size=self._component_placement.get_world_size("env"),
                    dst_world_size=self._component_placement.get_world_size("rollout"),
                    src_rank=self._rank,
                ),
            }
            if self.cfg.get("reward", {}).get("use_reward_model", False):
                dst_rank_map.update(
                    {
                        "reward_train": CommMapper.get_dst_ranks(
                            batch_size=self.cfg.env.train.total_num_envs
                            // self.stage_num,
                            src_world_size=self._component_placement.get_world_size(
                                "env"
                            ),
                            dst_world_size=self._component_placement.get_world_size(
                                "reward"
                            ),
                            src_rank=self._rank,
                        ),
                    }
                )

        if self.enable_eval:
            dst_rank_map.update(
                {
                    "rollout_eval": CommMapper.get_dst_ranks(
                        batch_size=self.cfg.env.eval.total_num_envs // self.stage_num,
                        src_world_size=self._component_placement.get_world_size("env"),
                        dst_world_size=self._component_placement.get_world_size(
                            "rollout"
                        ),
                        src_rank=self._rank,
                    ),
                }
            )
        return dst_rank_map

    def _setup_src_rank_map(self) -> dict[str, list[tuple[int, int]]]:
        """Compute source rank map for this env worker.

        This mapping supports both one-to-many and many-to-one env/rollout/reward layouts.
        The returned ranks are used as communication counterparts for both receiving results from rollout and reward workers and sending action chunks.

        Returns:
            Source rank map for this env worker.
            The key is the channel name (e.g. "rollout_train", "reward_train", "rollout_eval"), and the value is a ordered list of tuples of (src_rank, batch_size).
        """
        src_rank_map = {}
        if not self.only_eval:
            src_rank_map = {
                "rollout_train": CommMapper.get_src_ranks(
                    batch_size=self.cfg.env.train.total_num_envs // self.stage_num,
                    src_world_size=self._component_placement.get_world_size("rollout"),
                    dst_world_size=self._component_placement.get_world_size("env"),
                    dst_rank=self._rank,
                ),
            }
            if self.cfg.get("reward", {}).get("use_reward_model", False):
                src_rank_map.update(
                    {
                        "reward_train": CommMapper.get_src_ranks(
                            batch_size=self.cfg.env.train.total_num_envs
                            // self.stage_num,
                            src_world_size=self._component_placement.get_world_size(
                                "reward"
                            ),
                            dst_world_size=self._component_placement.get_world_size(
                                "env"
                            ),
                            dst_rank=self._rank,
                        ),
                    }
                )
        if self.enable_eval:
            src_rank_map.update(
                {
                    "rollout_eval": CommMapper.get_src_ranks(
                        batch_size=self.cfg.env.eval.total_num_envs // self.stage_num,
                        src_world_size=self._component_placement.get_world_size(
                            "rollout"
                        ),
                        dst_world_size=self._component_placement.get_world_size("env"),
                        dst_rank=self._rank,
                    ),
                }
            )
        return src_rank_map

    def _init_env(self):
        for i in range(self.stage_num):
            if self.cfg.env.train.auto_reset:
                extracted_obs, _ = self.env_list[i].reset()
                self.last_obs_list.append(extracted_obs)
                self.last_intervened_info_list.append((None, None))
            if self.enable_offload and hasattr(self.env_list[i], "offload"):
                self.env_list[i].offload()

    @Worker.timer("env_interact_step")
    def env_interact_step(
        self, chunk_actions: torch.Tensor, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """
        This function is used to interact with the environment.
        """
        chunk_actions = prepare_actions(
            raw_chunk_actions=chunk_actions,
            env_type=self.cfg.env.train.env_type,
            model_type=self.cfg.actor.model.model_type,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
            wm_env_type=self.cfg.env.train.get("wm_env_type", None),
        )
        env_info = {}

        obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list = (
            self.env_list[stage_id].chunk_step(chunk_actions)
        )
        if isinstance(obs_list, (list, tuple)):
            extracted_obs = obs_list[-1] if obs_list else None
        if isinstance(infos_list, (list, tuple)):
            infos = infos_list[-1] if infos_list else None
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
        final_obs = (
            self._build_chunk_final_obs(obs_list, infos_list)
            if self.use_external_reward_model
            or (self.cfg.env.train.auto_reset and chunk_dones.any())
            else (
                infos["final_observation"]
                if isinstance(infos, dict) and "final_observation" in infos
                else None
            )
        )
        if not self.cfg.env.train.auto_reset:
            if self.cfg.env.train.ignore_terminations:
                if chunk_truncations.any():
                    assert chunk_truncations.any(dim=1).all()
                    self._collect_chunk_final_episode_info(
                        env_info, infos_list, chunk_truncations
                    )
                    if not env_info and "episode" in infos:
                        for key in infos["episode"]:
                            env_info[key] = infos["episode"][key].cpu()
            else:
                if "episode" in infos:
                    for key in infos["episode"]:
                        env_info[key] = infos["episode"][key].cpu()
        elif chunk_dones.any():
            self._collect_chunk_final_episode_info(env_info, infos_list, chunk_dones)

        intervene_actions = None
        intervene_flags = None
        if isinstance(infos, dict):
            intervene_actions = infos.get(
                "chunk_intervene_action", infos.get("intervene_action")
            )
            intervene_flags = infos.get(
                "chunk_intervene_flag", infos.get("intervene_flag")
            )
            if (
                intervene_actions is None
                and self.cfg.env.train.auto_reset
                and chunk_dones.any()
                and "final_info" in infos
                and "intervene_action" in infos["final_info"]
            ):
                intervene_actions = infos["final_info"]["intervene_action"]
                intervene_flags = infos["final_info"]["intervene_flag"]

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=final_obs,
            rewards=chunk_rewards,
            env_infos=self._build_chunk_reward_env_infos(
                infos, infos_list, chunk_dones
            ),
            dones=chunk_dones,
            terminations=chunk_terminations,
            truncations=chunk_truncations,
            intervene_actions=intervene_actions,
            intervene_flags=intervene_flags,
        )
        return env_output, env_info

    def env_evaluate_step(
        self, raw_actions: torch.Tensor, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """
        This function is used to evaluate the environment.
        """
        chunk_actions = prepare_actions(
            raw_chunk_actions=raw_actions,
            env_type=self.cfg.env.eval.env_type,
            model_type=self.cfg.actor.model.model_type,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
            wm_env_type=self.cfg.env.eval.get("wm_env_type", None),
        )
        env_info = {}

        obs_list, _, chunk_terminations, chunk_truncations, infos_list = (
            self.eval_env_list[stage_id].chunk_step(chunk_actions)
        )
        if isinstance(obs_list, (list, tuple)):
            extracted_obs = obs_list[-1] if obs_list else None
        if isinstance(infos_list, (list, tuple)):
            infos = infos_list[-1] if infos_list else None
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
        final_obs = (
            self._build_chunk_final_obs(obs_list, infos_list)
            if self.use_external_reward_model
            or (self.cfg.env.eval.auto_reset and chunk_dones.any())
            else (
                infos["final_observation"]
                if isinstance(infos, dict) and "final_observation" in infos
                else None
            )
        )

        current_dones = chunk_dones.any(dim=1)  # [num_envs] bool
        if self.cfg.env.eval.auto_reset:
            newly_done_steps = chunk_dones
        else:
            prev = self.eval_prev_done[stage_id].to(current_dones.device)
            newly_done = current_dones & ~prev
            self.eval_prev_done[stage_id] = prev | current_dones
            newly_done_steps = chunk_dones & newly_done[:, None]

        if newly_done_steps.any():
            self._collect_chunk_final_episode_info(
                env_info, infos_list, newly_done_steps
            )
            if not env_info and isinstance(infos, dict) and "episode" in infos:
                newly_done = newly_done_steps.any(dim=1)
                for key in infos["episode"]:
                    env_info[key] = infos["episode"][key][newly_done].cpu()

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=final_obs,
        )
        return env_output, env_info

    def _collect_chunk_final_episode_info(self, env_info, infos_list, done_steps):
        if not isinstance(infos_list, (list, tuple)):
            return

        for step_idx, step_infos in enumerate(infos_list):
            if not isinstance(step_infos, dict) or "final_info" not in step_infos:
                continue
            final_info = step_infos["final_info"]
            if not isinstance(final_info, dict) or "episode" not in final_info:
                continue
            done_mask = done_steps[:, step_idx]
            if not done_mask.any():
                continue
            for key, value in final_info["episode"].items():
                selected_value = value[done_mask].cpu()
                if key in env_info:
                    env_info[key] = torch.cat([env_info[key], selected_value], dim=0)
                else:
                    env_info[key] = selected_value

    def _build_chunk_reward_env_infos(self, last_infos, infos_list, done_steps):
        if not isinstance(last_infos, dict):
            return None
        env_infos = dict(last_infos)
        if not isinstance(infos_list, (list, tuple)) or done_steps is None:
            return env_infos

        env_infos.pop("final_info", None)
        final_info = None
        for step_idx, step_infos in enumerate(infos_list):
            if not isinstance(step_infos, dict) or "final_info" not in step_infos:
                continue
            done_mask = done_steps[:, step_idx]
            if done_mask.any():
                final_info = self._merge_nested_batch_values_by_mask(
                    final_info, step_infos["final_info"], done_mask
                )
        if final_info is not None:
            env_infos["final_info"] = final_info
        return env_infos

    @staticmethod
    def _merge_nested_batch_values_by_mask(base, update, mask):
        """Merge per-env nested values from update where mask is true."""
        if base is None:
            return EnvWorker._empty_masked_nested_batch_value(update, mask)
        if isinstance(base, dict) and isinstance(update, dict):
            merged = {key: copy.deepcopy(value) for key, value in base.items()}
            for key, update_value in update.items():
                if key in merged:
                    merged[key] = EnvWorker._merge_nested_batch_values_by_mask(
                        merged[key], update_value, mask
                    )
                else:
                    merged[key] = EnvWorker._empty_masked_nested_batch_value(
                        update_value, mask
                    )
            return merged
        if isinstance(base, torch.Tensor) and isinstance(update, torch.Tensor):
            if base.shape[:1] != update.shape[:1] or base.shape[:1] != mask.shape[:1]:
                raise ValueError(
                    "Cannot merge final_info tensors with mismatched batch shapes: "
                    f"base_shape={tuple(base.shape)}, "
                    f"update_shape={tuple(update.shape)}, "
                    f"mask_shape={tuple(mask.shape)}."
                )
            merged = base.clone()
            base_mask = mask.to(device=merged.device, dtype=torch.bool)
            update_mask = mask.to(device=update.device, dtype=torch.bool)
            merged[base_mask] = update[update_mask].to(
                device=merged.device, dtype=merged.dtype
            )
            return merged
        if isinstance(base, np.ndarray) and isinstance(update, np.ndarray):
            if base.shape[:1] != update.shape[:1] or base.shape[:1] != mask.shape[:1]:
                raise ValueError(
                    "Cannot merge final_info arrays with mismatched batch shapes: "
                    f"base_shape={base.shape}, update_shape={update.shape}, "
                    f"mask_shape={tuple(mask.shape)}."
                )
            merged = base.copy()
            np_mask = mask.detach().cpu().numpy().astype(bool)
            merged[np_mask] = update[np_mask]
            return merged
        return copy.deepcopy(update) if bool(mask.any()) else copy.deepcopy(base)

    @staticmethod
    def _empty_masked_nested_batch_value(update, mask):
        if isinstance(update, dict):
            return {
                key: EnvWorker._empty_masked_nested_batch_value(value, mask)
                for key, value in update.items()
            }
        if isinstance(update, torch.Tensor):
            masked = torch.zeros_like(update)
            if update.shape[:1] != mask.shape[:1]:
                raise ValueError(
                    "Cannot build masked final_info tensor with mismatched shape: "
                    f"update_shape={tuple(update.shape)}, "
                    f"mask_shape={tuple(mask.shape)}."
                )
            update_mask = mask.to(device=update.device, dtype=torch.bool)
            masked[update_mask] = update[update_mask]
            return masked
        if isinstance(update, np.ndarray):
            masked = np.zeros_like(update)
            if update.shape[:1] != mask.shape[:1]:
                raise ValueError(
                    "Cannot build masked final_info array with mismatched shape: "
                    f"update_shape={update.shape}, mask_shape={tuple(mask.shape)}."
                )
            np_mask = mask.detach().cpu().numpy().astype(bool)
            masked[np_mask] = update[np_mask]
            return masked
        return copy.deepcopy(update) if bool(mask.any()) else None

    @staticmethod
    def _mask_chunk_tensor(
        value: torch.Tensor | None,
        valid_steps: torch.Tensor,
        *,
        field_name: str,
    ) -> torch.Tensor | None:
        if value is None:
            return None
        if not isinstance(value, torch.Tensor):
            return value
        if valid_steps.ndim != 2:
            raise ValueError(
                f"chunk_valid_step must be rank-2 [B, chunk_steps], got "
                f"shape={tuple(valid_steps.shape)}."
            )
        batch_size, chunk_steps = valid_steps.shape
        if value.shape[:1] != (batch_size,):
            return value
        mask = valid_steps.to(device=value.device, dtype=torch.bool)
        if value.ndim >= 3 and value.shape[1] == chunk_steps:
            view_shape = (batch_size, chunk_steps) + (1,) * (value.ndim - 2)
            return value.masked_fill(~mask.view(view_shape), 0)
        if value.ndim == 2 and value.shape[1] % chunk_steps == 0:
            action_dim = value.shape[1] // chunk_steps
            reshaped = value.reshape(batch_size, chunk_steps, action_dim)
            return reshaped.masked_fill(~mask.unsqueeze(-1), 0).reshape_as(value)
        raise ValueError(
            f"Cannot align {field_name} shape={tuple(value.shape)} with "
            f"chunk_valid_step shape={tuple(valid_steps.shape)}."
        )

    def _mask_invalid_chunk_actions(
        self, rollout_result: RolloutResult, env_output: EnvOutput
    ) -> RolloutResult:
        valid_steps = self._get_chunk_valid_steps(env_output)
        if valid_steps is None:
            return rollout_result
        rollout_result.actions = self._mask_chunk_tensor(
            rollout_result.actions, valid_steps, field_name="actions"
        )
        if (
            rollout_result.save_flags is not None
            and rollout_result.save_flags.shape == valid_steps.shape
        ):
            rollout_result.save_flags = rollout_result.save_flags & valid_steps.to(
                device=rollout_result.save_flags.device
            )
        if rollout_result.forward_inputs:
            forward_inputs = dict(rollout_result.forward_inputs)
            for key in ("action", "model_action"):
                if key in forward_inputs:
                    forward_inputs[key] = self._mask_chunk_tensor(
                        forward_inputs[key],
                        valid_steps,
                        field_name=f"forward_inputs.{key}",
                    )
            rollout_result.forward_inputs = forward_inputs
        return rollout_result

    @staticmethod
    def _get_chunk_valid_steps(env_output: EnvOutput) -> torch.Tensor | None:
        if not isinstance(env_output.env_infos, dict):
            return None
        valid_steps = env_output.env_infos.get("chunk_valid_step")
        if valid_steps is None:
            return None
        return torch.as_tensor(valid_steps, dtype=torch.bool)

    def _mask_last_invalid_chunk_actions(
        self, rollout_result: EmbodiedRolloutResult, env_output: EnvOutput
    ) -> None:
        valid_steps = self._get_chunk_valid_steps(env_output)
        if valid_steps is None or not rollout_result.actions:
            return
        rollout_result.mark_last_step_with_loss_mask(valid_steps)
        rollout_result.actions[-1] = self._mask_chunk_tensor(
            rollout_result.actions[-1], valid_steps, field_name="actions"
        )
        if rollout_result.intervene_flags:
            rollout_result.intervene_flags[-1] = self._mask_chunk_tensor(
                rollout_result.intervene_flags[-1],
                valid_steps,
                field_name="intervene_flags",
            )
        if rollout_result.forward_inputs:
            last_forward_inputs = dict(rollout_result.forward_inputs[-1])
            for key in ("action", "model_action"):
                if key in last_forward_inputs:
                    last_forward_inputs[key] = self._mask_chunk_tensor(
                        last_forward_inputs[key],
                        valid_steps,
                        field_name=f"forward_inputs.{key}",
                    )
            rollout_result.forward_inputs[-1] = last_forward_inputs

    def _update_last_actions_from_env_output(
        self, rollout_result: EmbodiedRolloutResult, env_output: EnvOutput
    ) -> None:
        if (
            env_output.intervene_actions is not None
            and env_output.intervene_flags is not None
        ):
            intervene_actions = env_output.intervene_actions
            intervene_flags = env_output.intervene_flags
            valid_steps = self._get_chunk_valid_steps(env_output)
            if valid_steps is not None:
                intervene_actions = self._mask_chunk_tensor(
                    intervene_actions,
                    valid_steps,
                    field_name="intervene_actions",
                )
                intervene_flags = self._mask_chunk_tensor(
                    intervene_flags,
                    valid_steps,
                    field_name="intervene_flags",
                )
            rollout_result.update_last_actions(intervene_actions, intervene_flags)
        self._mask_last_invalid_chunk_actions(rollout_result, env_output)

    def _build_chunk_final_obs(self, obs_list, infos_list):
        """Build per-env terminal observations for a whole chunk.

        Matches the old wrapper semantics:
        - default to the last rollout observation for each env
        - if an env terminated earlier in the chunk, replace that env's observation
          with the true `final_observation` captured at that substep
        """
        if not isinstance(obs_list, (list, tuple)) or len(obs_list) == 0:
            return None

        last_obs = obs_list[-1]
        if not isinstance(last_obs, dict):
            return None

        merged_final_obs = copy_dict_tensor(last_obs)

        if not isinstance(infos_list, (list, tuple)):
            return merged_final_obs

        for step_infos in infos_list:
            if not isinstance(step_infos, dict):
                continue
            if (
                "final_observation" not in step_infos
                or "_final_observation" not in step_infos
            ):
                continue

            final_obs = step_infos["final_observation"]
            reset_mask = step_infos["_final_observation"]
            if final_obs is None or reset_mask is None:
                continue
            reset_mask = (
                reset_mask.detach().cpu().numpy()
                if isinstance(reset_mask, torch.Tensor)
                else np.asarray(reset_mask)
            )
            done_mask = (
                reset_mask.any(axis=-1)
                if reset_mask.ndim > 1
                else reset_mask.astype(bool)
            )
            if not done_mask.any():
                continue

            for key, value in merged_final_obs.items():
                if key not in final_obs:
                    continue

                final_value = final_obs[key]
                if isinstance(value, torch.Tensor) and isinstance(
                    final_value, torch.Tensor
                ):
                    dst_mask = torch.as_tensor(done_mask, device=value.device)
                    src_mask = dst_mask.to(device=final_value.device)
                    merged_final_obs[key][dst_mask] = final_value[src_mask]
                elif isinstance(value, np.ndarray) and isinstance(
                    final_value, np.ndarray
                ):
                    merged_final_obs[key][done_mask] = final_value[done_mask]

        return merged_final_obs

    def recv_chunk_actions(self, input_channel: Channel, mode="train") -> np.ndarray:
        """Receive and merge chunked actions for the current env worker.

        The method fetches one action shard from each mapped rollout source rank
        under a deterministic channel key pattern and concatenates them on the
        batch dimension.

        Args:
            input_channel: Channel carrying rollout->env action chunks.
            mode: Rollout mode, either ``"train"`` or ``"eval"``.

        Returns:
            Concatenated action chunk array with shape ``[num_envs_per_stage, ...]``.
        """
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        src_ranks_and_sizes = self.src_rank_map[f"rollout_{mode}"]
        chunk_action = []
        for src_rank, expected_size in src_ranks_and_sizes:
            action_i = input_channel.get(
                key=CommMapper.build_channel_key(
                    src_rank, self._rank, extra=f"{mode}_actions"
                ),
            )
            if isinstance(action_i, torch.Tensor):
                action_i = action_i.detach().cpu().numpy()
            else:
                action_i = np.asarray(action_i)
            assert action_i.shape[0] == expected_size, (
                f"Expected action shard size {expected_size} from rollout rank {src_rank}, "
                f"got shape {action_i.shape}."
            )
            chunk_action.append(action_i)
        chunk_action = np.concatenate(chunk_action, axis=0)
        expected_total_size = sum(size for _, size in src_ranks_and_sizes)
        assert chunk_action.shape[0] == expected_total_size, (
            f"Expected concatenated action size {expected_total_size}, got {chunk_action.shape[0]}."
        )
        return chunk_action

    @Worker.timer("recv_rollout_results")
    def recv_rollout_results(
        self, input_channel: Channel, mode="train"
    ) -> RolloutResult:
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        src_ranks_and_sizes = self.src_rank_map[f"rollout_{mode}"]
        rollout_results: list[RolloutResult] = []

        def _infer_rollout_batch_size(rollout_result: RolloutResult) -> int:
            for field_name in (
                "actions",
                "prev_logprobs",
                "prev_values",
                "bootstrap_values",
                "versions",
            ):
                value = getattr(rollout_result, field_name, None)
                if isinstance(value, torch.Tensor):
                    return value.shape[0]
            if rollout_result.forward_inputs:
                first_tensor = next(iter(rollout_result.forward_inputs.values()))
                if isinstance(first_tensor, torch.Tensor):
                    return first_tensor.shape[0]
            raise ValueError("Cannot infer batch size from rollout result.")

        for src_rank, expected_size in src_ranks_and_sizes:
            rollout_result = input_channel.get(
                key=CommMapper.build_channel_key(
                    src_rank, self._rank, extra=f"{mode}_rollout_results"
                ),
            )

            actual_size = _infer_rollout_batch_size(rollout_result)
            assert actual_size == expected_size, (
                f"Expected rollout result size {expected_size} from rollout rank {src_rank}, "
                f"got batch size {actual_size}."
            )

            rollout_results.append(rollout_result)

        return RolloutResult.merge_rollout_results(rollout_results)

    @Worker.timer("compute_bootstrap_rewards")
    def compute_bootstrap_rewards(
        self,
        env_output: EnvOutput,
        bootstrap_values: torch.Tensor | None,
        reward_model_output: torch.Tensor | None,
    ) -> torch.Tensor | None:
        rewards = env_output.rewards
        if rewards is None:
            return None

        if reward_model_output is not None:
            reward_model_output = reward_model_output.to(
                device=rewards.device, dtype=rewards.dtype
            )
            if (
                reward_model_output.dim() == 1
                and rewards.dim() == 2
                and reward_model_output.shape[0] == rewards.shape[0]
            ):
                reward_model_output = reward_model_output[:, None]
            if (
                reward_model_output.dim() == rewards.dim()
                and reward_model_output.shape[:1] == rewards.shape[:1]
                and reward_model_output.shape[1:] == (1,)
                and rewards.shape[1:] != (1,)
            ):
                reward_model_output = reward_model_output.expand_as(rewards)
            if reward_model_output.shape != rewards.shape:
                raise ValueError(
                    "reward_model_output shape must match env rewards shape: "
                    f"reward_model_output_shape={tuple(reward_model_output.shape)}, "
                    f"rewards_shape={tuple(rewards.shape)}."
                )
            rewards = (
                self.env_reward_weight * rewards
                + self.reward_weight * reward_model_output
            )

        valid_steps = self._get_chunk_valid_steps(env_output)
        if valid_steps is not None:
            if valid_steps.shape != rewards.shape:
                raise ValueError(
                    "chunk_valid_step shape must match rewards shape: "
                    f"chunk_valid_step_shape={tuple(valid_steps.shape)}, "
                    f"rewards_shape={tuple(rewards.shape)}."
                )
            rewards = rewards.masked_fill(
                ~valid_steps.to(device=rewards.device, dtype=torch.bool), 0
            )

        adjusted_rewards = rewards.clone()
        if (
            bootstrap_values is None
            or not self.cfg.env.train.auto_reset
            or env_output.dones is None
        ):
            return adjusted_rewards

        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")
        if bootstrap_type == "standard":
            bootstrap_steps = env_output.truncations
        else:
            bootstrap_steps = env_output.dones

        if bootstrap_steps is None:
            return adjusted_rewards

        bootstrap_steps = bootstrap_steps.to(
            device=adjusted_rewards.device, dtype=torch.bool
        )
        if bootstrap_steps.dim() != adjusted_rewards.dim():
            raise ValueError(
                "bootstrap mask rank must match rewards rank: "
                f"bootstrap_steps_shape={tuple(bootstrap_steps.shape)}, "
                f"rewards_shape={tuple(adjusted_rewards.shape)}."
            )
        if not bootstrap_steps.any():
            return adjusted_rewards
        bootstrap_values = bootstrap_values.reshape(-1).to(
            device=adjusted_rewards.device, dtype=torch.float32
        )
        if bootstrap_steps.dim() == 1:
            adjusted_rewards[bootstrap_steps] += (
                self.cfg.algorithm.gamma * bootstrap_values[bootstrap_steps]
            ).to(adjusted_rewards.dtype)
            return adjusted_rewards

        if bootstrap_steps.dim() != 2:
            raise ValueError(
                "bootstrap mask rank must be 1 or 2: "
                f"bootstrap_steps_shape={tuple(bootstrap_steps.shape)}."
            )
        has_bootstrap = bootstrap_steps.any(dim=1)
        first_bootstrap_step = bootstrap_steps.to(torch.int64).argmax(dim=1)
        rows = torch.arange(adjusted_rewards.shape[0], device=adjusted_rewards.device)
        rows = rows[has_bootstrap]
        cols = first_bootstrap_step[has_bootstrap].to(device=adjusted_rewards.device)
        adjusted_rewards[rows, cols] += (
            self.cfg.algorithm.gamma * bootstrap_values[has_bootstrap]
        ).to(adjusted_rewards.dtype)
        return adjusted_rewards

    def finish_rollout(self, mode="train"):
        # reset
        if mode == "train":
            for i in range(self.stage_num):
                if self.cfg.env.train.video_cfg.save_video and isinstance(
                    self.env_list[i], RecordVideo
                ):
                    self.env_list[i].flush_video()
                self.env_list[i].update_reset_state_ids()
        elif mode == "eval":
            for i in range(self.stage_num):
                if self.cfg.env.eval.video_cfg.save_video and isinstance(
                    self.eval_env_list[i], RecordVideo
                ):
                    self.eval_env_list[i].flush_video()
                if not self.cfg.env.eval.auto_reset:
                    self.eval_env_list[i].update_reset_state_ids()

    def send_env_batch(
        self,
        rollout_channel: Channel,
        env_batch: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
    ) -> None:
        """Send split env batches to mapped rollout ranks.

        Each destination rank receives one split batch via a stable key built from
        ``src_rank``, ``dst_rank`` and ``mode``.

        Args:
            rollout_channel: Channel carrying env->rollout outputs.
            env_batch: Env output dictionary for one pipeline stage.
            mode: Rollout mode, either ``"train"`` or ``"eval"``.
        """
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        dst_ranks_and_sizes = self.dst_rank_map[f"rollout_{mode}"]
        split_sizes = [size for _, size in dst_ranks_and_sizes]
        env_batches = split_dict(env_batch, split_sizes)
        for (rank, _), env_batch_i in zip(dst_ranks_and_sizes, env_batches):
            rollout_channel.put(
                item=env_batch_i,
                key=CommMapper.build_channel_key(self._rank, rank, extra=f"{mode}_obs"),
            )

    def send_reward_input(
        self,
        send_channel: Channel,
        reward_input: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
    ):
        dst_ranks_and_sizes = self.dst_rank_map[f"reward_{mode}"]
        split_sizes = [size for _, size in dst_ranks_and_sizes]
        reward_input_batches = split_dict(reward_input, split_sizes)
        for (rank, _), reward_input_i in zip(dst_ranks_and_sizes, reward_input_batches):
            send_channel.put(
                item=reward_input_i,
                key=CommMapper.build_channel_key(
                    self._rank, rank, extra=f"{mode}_reward_input"
                ),
                async_op=True,
            )

    @Worker.timer("recv_reward_results")
    def recv_reward_results(self, recv_channel: Channel) -> torch.Tensor:
        reward_results: list[torch.Tensor] = []
        src_ranks_and_sizes = self.src_rank_map["reward_train"]
        for src_rank, expected_size in src_ranks_and_sizes:
            rewards = recv_channel.get(
                key=CommMapper.build_channel_key(
                    src_rank, self._rank, extra="reward_output"
                ),
            )
            if rewards is None:
                rewards = torch.zeros(expected_size, dtype=torch.float32)
            actual_size = rewards.shape[0]
            assert actual_size == expected_size, (
                f"Expected reward result size {expected_size} from reward rank {src_rank}, "
                f"got batch size {actual_size}."
            )
            reward_results.append(rewards)
        return torch.cat(reward_results, dim=0)

    @Worker.timer("get_reward_model_output")
    def get_reward_model_output(
        self,
        env_output: EnvOutput,
        send_channel: Channel,
        recv_channel: Channel,
        stage_id: int | None = None,
        last_run: bool = False,
    ):
        if self.reward_mode in {"per_step", "history_buffer"}:
            observations = (
                env_output.final_obs
                if env_output.final_obs is not None
                else env_output.obs
            )
        elif self.reward_mode == "terminal" and env_output.final_obs is not None:
            observations = env_output.final_obs
        else:
            return None
        reward_input = dict(observations)
        if env_output.env_infos is not None:
            reward_input["env_infos"] = self._select_reward_env_infos(
                env_output.env_infos
            )

        dones = env_output.dones
        if dones is not None:
            if getattr(dones, "ndim", 0) > 1:
                dones = dones.any(dim=1)
            reward_input.update({"dones": dones})

        if self.reward_mode == "history_buffer":
            if stage_id is None:
                raise ValueError("stage_id is required for history-buffer reward.")
            history_manager = self.train_history_managers[stage_id]
            history_manager.append_to_history_entries(observations)
            history_input, history_lengths = history_manager.build_history_input(
                dones=dones
            )
            reward_input["history_input"] = history_input
            self.history_lengths[stage_id] = dict(history_lengths)

        if last_run:
            reward_input.update(
                {
                    "last_run": torch.ones(
                        (self.train_num_envs_per_stage, 1), dtype=torch.bool
                    )
                }
            )
        self.send_reward_input(send_channel=send_channel, reward_input=reward_input)
        reward_output = self.recv_reward_results(recv_channel=recv_channel)
        if self.reward_mode != "terminal" or reward_output is None:
            return reward_output
        return self._scatter_terminal_reward_output(
            env_output=env_output, reward_output=reward_output
        )

    def _select_reward_env_infos(self, env_infos: dict[str, Any]) -> dict[str, Any]:
        reward_env_infos = {}
        for key in self.env_infos_reward_keys:
            if key not in env_infos:
                continue
            reward_env_infos[key] = clone_nested_to_cpu(env_infos[key])
        return reward_env_infos

    def _scatter_terminal_reward_output(
        self,
        env_output: EnvOutput,
        reward_output: torch.Tensor,
    ) -> torch.Tensor:
        if env_output.rewards is None or env_output.dones is None:
            return reward_output

        done_envs = env_output.dones.any(dim=1)
        sparse_rewards = torch.zeros_like(env_output.rewards, dtype=reward_output.dtype)
        if not done_envs.any():
            return sparse_rewards

        done_steps = env_output.dones.to(torch.int64).argmax(dim=1)
        sparse_rewards[done_envs, done_steps[done_envs]] = (
            reward_output[done_envs].reshape(-1).to(sparse_rewards.dtype)
        )
        return sparse_rewards

    def assign_history_reward(self, stage_id: int, reward_model_output: torch.Tensor):
        reward_assign_lengths = [
            min(
                history_buffer_length[env_id]
                for history_buffer_length in self.history_lengths[stage_id].values()
            )
            for env_id in range(self.train_num_envs_per_stage)
        ]
        rollout_rewards = self.rollout_results[stage_id].rewards
        rollout_rewards_length = len(rollout_rewards)
        reward_assign_lengths = [
            min(reward_assign_length, rollout_rewards_length)
            for reward_assign_length in reward_assign_lengths
        ]
        if not any(reward_assign_lengths):
            return
        reward = (self.reward_weight * reward_model_output).to(
            rollout_rewards[-1].dtype
        )
        for env_id, reward_assign_length in enumerate(reward_assign_lengths):
            for reward_assign_step in range(2, reward_assign_length + 1):
                rollout_rewards[-reward_assign_step][env_id] += reward[env_id]

    def bootstrap_step(self) -> list[EnvOutput]:
        def get_zero_dones() -> torch.Tensor:
            return (
                torch.zeros((self.train_num_envs_per_stage,), dtype=bool)
                .unsqueeze(1)
                .repeat(1, self.cfg.actor.model.num_action_chunks)
            )

        env_outputs: list[EnvOutput] = []
        if not self.cfg.env.train.auto_reset:
            for stage_id in range(self.stage_num):
                self.env_list[stage_id].is_start = True
                extracted_obs, infos = self.env_list[stage_id].reset()
                dones = get_zero_dones()
                terminations = dones.clone()
                truncations = dones.clone()

                env_output = EnvOutput(
                    obs=extracted_obs,
                    dones=dones,
                    terminations=terminations,
                    truncations=truncations,
                    final_obs=(
                        infos["final_observation"]
                        if "final_observation" in infos
                        else None
                    ),
                    intervene_actions=None,
                    intervene_flags=None,
                )
                env_outputs.append(env_output)
        else:
            dones = get_zero_dones()
            terminations = dones.clone()
            truncations = dones.clone()

            for stage_id in range(self.stage_num):
                env_output = EnvOutput(
                    obs=self.last_obs_list[stage_id],
                    rewards=None,
                    dones=dones,
                    terminations=terminations,
                    truncations=truncations,
                    intervene_actions=self.last_intervened_info_list[stage_id][0],
                    intervene_flags=self.last_intervened_info_list[stage_id][1],
                )
                env_outputs.append(env_output)

        return env_outputs

    def _send_train_bootstrap(
        self, rollout_channel: Channel, env_outputs: list[EnvOutput]
    ) -> None:
        for stage_id in range(self.stage_num):
            env_output: EnvOutput = env_outputs[stage_id]
            env_batch = env_output.to_dict()
            self.send_env_batch(
                rollout_channel,
                {
                    "obs": env_batch["obs"],
                    "final_obs": env_batch["final_obs"],
                },
            )

    def _bootstrap_and_send_train(self, rollout_channel: Channel) -> list[EnvOutput]:
        env_outputs = self.bootstrap_step()
        self._send_train_bootstrap(rollout_channel, env_outputs)
        return env_outputs

    def prefetch_train_bootstrap(self, rollout_channel: Channel) -> None:
        """Prepare and send the first env batch for the next training rollout."""
        if self._prefetched_train_bootstrap is not None:
            raise RuntimeError(
                "A prefetched train bootstrap already exists. "
                "Call interact() to consume it before prefetching again."
            )
        self._prefetched_train_bootstrap = self._bootstrap_and_send_train(
            rollout_channel
        )

    def record_env_metrics(
        self, env_metrics: dict[str, list], env_info: dict[str, Any], epoch: int
    ):
        for key, value in env_info.items():
            if (
                not self.cfg.env.train.auto_reset
                and not self.cfg.env.train.ignore_terminations
            ):
                if key in env_metrics and len(env_metrics[key]) > epoch:
                    env_metrics[key][epoch] = value
                else:
                    env_metrics[key].append(value)
            else:
                env_metrics[key].append(value)

    def store_last_obs_and_intervened_info(self, env_output_list: list[EnvOutput]):
        self.last_obs_list = [env_output.obs for env_output in env_output_list]
        self.last_intervened_info_list = [
            (env_output.intervene_actions, env_output.intervene_flags)
            for env_output in env_output_list
        ]

    async def send_rollout_trajectories(
        self, rollout_result: EmbodiedRolloutResult, channel: Channel
    ):
        trajectories: list[Trajectory] = rollout_result.to_splited_trajectories(
            self.actor_split_num
        )
        rollout_result.clear()
        for trajectory in trajectories:
            channel.put(trajectory, async_op=True)
        del trajectories
        gc.collect()

    @Worker.timer("run_interact_once")
    async def _run_interact_once(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
        actor_channel: Channel | None,
        *,
        cooperative_yield: bool,
    ) -> dict[str, torch.Tensor]:
        self.rollout_results: list[EmbodiedRolloutResult] = [
            EmbodiedRolloutResult(
                max_episode_length=self.cfg.env.train.max_episode_steps,
            )
            for _ in range(self.stage_num)
        ]
        env_metrics = defaultdict(list)

        for epoch in range(self.rollout_epoch):
            if epoch == 0 and self._prefetched_train_bootstrap is not None:
                env_outputs = self._prefetched_train_bootstrap
                self._prefetched_train_bootstrap = None
            else:
                env_outputs = self._bootstrap_and_send_train(rollout_channel)

            for chunk_step_idx in range(self.n_train_chunk_steps):
                for stage_id in range(self.stage_num):
                    if cooperative_yield:
                        await asyncio.sleep(0)

                    env_output = env_outputs[stage_id]
                    curr_obs = env_output.obs
                    self._update_last_actions_from_env_output(
                        self.rollout_results[stage_id], env_output
                    )

                    reward_model_output = None
                    if reward_channel is not None and chunk_step_idx != 0:
                        reward_model_output = self.get_reward_model_output(
                            env_output,
                            send_channel=reward_channel,
                            recv_channel=input_channel,
                            stage_id=stage_id,
                        )
                        if reward_model_output is not None:
                            env_metrics["reward_model_output"].append(
                                reward_model_output.detach().float().reshape(-1).cpu()
                            )

                    rollout_result = self.recv_rollout_results(
                        input_channel, mode="train"
                    )
                    rewards = self.compute_bootstrap_rewards(
                        env_output, rollout_result.bootstrap_values, reward_model_output
                    )
                    chunk_step_result = ChunkStepResult(
                        actions=rollout_result.forward_inputs.get("action", None),
                        prev_logprobs=(
                            rollout_result.prev_logprobs
                            if self.collect_prev_infos
                            else None
                        ),
                        prev_values=(
                            rollout_result.prev_values
                            if self.collect_prev_infos
                            else None
                        ),
                        forward_inputs=rollout_result.forward_inputs,
                        versions=rollout_result.versions,
                        dones=env_output.dones,
                        truncations=env_output.truncations,
                        terminations=env_output.terminations,
                        rewards=rewards,
                    )
                    self.rollout_results[stage_id].append_step_result(chunk_step_result)
                    if (
                        self.reward_mode == "history_buffer"
                        and self.history_reward_assign
                        and reward_model_output is not None
                    ):
                        self.assign_history_reward(stage_id, reward_model_output)
                    if rollout_result.save_flags is not None:
                        self.rollout_results[stage_id].mark_last_step_with_flags(
                            rollout_result.save_flags
                        )

                    env_output, env_info = self.env_interact_step(
                        rollout_result.actions, stage_id
                    )
                    self._mask_last_invalid_chunk_actions(
                        self.rollout_results[stage_id], env_output
                    )
                    env_batch = env_output.to_dict()
                    self.send_env_batch(
                        rollout_channel,
                        {
                            "obs": env_batch["obs"],
                            "final_obs": env_batch["final_obs"],
                        },
                    )
                    if self.collect_transitions:
                        next_obs = (
                            env_output.final_obs
                            if env_output.dones.any() and self.cfg.env.train.auto_reset
                            else env_output.obs
                        )
                        self.rollout_results[stage_id].append_transitions(
                            curr_obs, next_obs
                        )

                    env_outputs[stage_id] = env_output
                    self.record_env_metrics(env_metrics, env_info, epoch)

            for stage_id in range(self.stage_num):
                env_output = env_outputs[stage_id]
                self._update_last_actions_from_env_output(
                    self.rollout_results[stage_id], env_output
                )

                reward_model_output = None
                if reward_channel is not None:
                    last_run = epoch == self.rollout_epoch - 1
                    reward_model_output = self.get_reward_model_output(
                        env_output,
                        send_channel=reward_channel,
                        recv_channel=input_channel,
                        stage_id=stage_id,
                        last_run=last_run,
                    )
                    if reward_model_output is not None:
                        env_metrics["reward_model_output"].append(
                            reward_model_output.detach().float().reshape(-1).cpu()
                        )
                rollout_result = self.recv_rollout_results(input_channel, mode="train")
                rewards = self.compute_bootstrap_rewards(
                    env_output, rollout_result.bootstrap_values, reward_model_output
                )
                chunk_step_result = ChunkStepResult(
                    prev_values=(
                        rollout_result.prev_values if self.collect_prev_infos else None
                    ),
                    dones=env_output.dones,
                    truncations=env_output.truncations,
                    terminations=env_output.terminations,
                    rewards=rewards,
                )
                self.rollout_results[stage_id].append_step_result(chunk_step_result)
                if (
                    self.reward_mode == "history_buffer"
                    and self.history_reward_assign
                    and reward_model_output is not None
                ):
                    self.assign_history_reward(stage_id, reward_model_output)

            self.store_last_obs_and_intervened_info(env_outputs)
            self.finish_rollout()

        if actor_channel is not None:
            for stage_id in range(self.stage_num):
                await self.send_rollout_trajectories(
                    self.rollout_results[stage_id], actor_channel
                )
            # reduce memory peak
            self.rollout_results = []
            gc.collect()

        for key, value in env_metrics.items():
            env_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return env_metrics

    @Worker.timer("interact")
    async def interact(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
        actor_channel: Channel | None = None,
    ):
        env_metrics = await self._run_interact_once(
            input_channel,
            rollout_channel,
            reward_channel,
            actor_channel,
            cooperative_yield=False,
        )

        for env in self.env_list:
            if self.enable_offload and hasattr(env, "offload"):
                env.offload()

        return env_metrics

    def evaluate(self, input_channel: Channel, rollout_channel: Channel):
        eval_metrics = defaultdict(list)

        for eval_rollout_epoch in range(self.cfg.algorithm.eval_rollout_epoch):
            if not self.cfg.env.eval.auto_reset or eval_rollout_epoch == 0:
                for stage_id in range(self.stage_num):
                    self.eval_env_list[stage_id].is_start = True
                    self.eval_prev_done[stage_id] = torch.zeros(
                        self.eval_num_envs_per_stage, dtype=torch.bool
                    )
                    extracted_obs, infos = self.eval_env_list[stage_id].reset()
                    env_output = EnvOutput(
                        obs=extracted_obs,
                        final_obs=(
                            infos["final_observation"]
                            if "final_observation" in infos
                            else None
                        ),
                    )
                    env_batch = env_output.to_dict()
                    self.send_env_batch(
                        rollout_channel,
                        {
                            "obs": env_batch["obs"],
                            "final_obs": env_batch["final_obs"],
                        },
                        mode="eval",
                    )

            for eval_step in range(self.n_eval_chunk_steps):
                for stage_id in range(self.stage_num):
                    raw_chunk_actions = self.recv_chunk_actions(
                        input_channel, mode="eval"
                    )
                    env_output, env_info = self.env_evaluate_step(
                        raw_chunk_actions, stage_id
                    )

                    for key, value in env_info.items():
                        eval_metrics[key].append(value)

                    if self.cfg.env.eval.auto_reset:
                        if (
                            eval_rollout_epoch
                            == self.cfg.algorithm.eval_rollout_epoch - 1
                            and eval_step == self.n_eval_chunk_steps - 1
                        ):
                            continue
                    else:
                        if eval_step == self.n_eval_chunk_steps - 1:
                            continue
                    env_batch = env_output.to_dict()
                    self.send_env_batch(
                        rollout_channel,
                        {
                            "obs": env_batch["obs"],
                            "final_obs": env_batch["final_obs"],
                        },
                        mode="eval",
                    )

            self.finish_rollout(mode="eval")
        for stage_id in range(self.stage_num):
            if self.cfg.env.eval.get("enable_offload", False) and hasattr(
                self.eval_env_list[stage_id], "offload"
            ):
                self.eval_env_list[stage_id].offload()

        for key, value in eval_metrics.items():
            eval_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return eval_metrics

    def get_actor_split_num(self):
        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)
        return split_num
