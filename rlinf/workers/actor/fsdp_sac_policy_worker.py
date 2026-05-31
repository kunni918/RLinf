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


import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from rlinf.config import (
    SupportedModel,
    validate_demo_buffer_load_mode,
    validate_embodied_sac_model_type,
)
from rlinf.data.embodied_buffer_dataset import (
    PreloadReplayBufferDataset,
    ReplayBufferDataset,
    replay_buffer_collate_fn,
)
from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.modules.entropy_tunning import EntropyTemperature
from rlinf.scheduler import Channel, Worker
from rlinf.utils import drq
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_split_num,
)
from rlinf.utils.nested_dict_process import (
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.utils.utils import clear_memory
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.actor.sac_demo_buffer_utils import (
    apply_chunk_mask,
    chunk_sample_weights,
    demo_buffer_load_kwargs,
    validate_loaded_demo_buffer,
)


class EmbodiedSACFSDPPolicy(EmbodiedFSDPActor):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # SAC-specific initialization
        self.replay_buffer = None
        self.target_model = None
        self.entropy_temp = None
        self.demo_buffer = None
        self.alpha_optimizer = None
        self.update_step = 0
        self.enable_drq = bool(getattr(self.cfg.actor, "enable_drq", False))

    def init_worker(self):
        # MAJ-R4-7: under ``only_eval`` skip the WHOLE training-only SAC
        # stack (replay/demo buffer, dataloader, entropy temperature,
        # alpha optimizer, dataloader iterator, soft target init). Eval
        # still needs the online model + target model for Q evaluation,
        # so we initialize those — but skip everything that requires the
        # training config sections (``algorithm.replay_buffer``,
        # ``algorithm.demo_buffer``, ``algorithm.entropy_tuning``). This
        # lets shipped eval-only configs load cleanly without bringing
        # in training-only validation / requirements.
        only_eval = bool(self.cfg.runner.only_eval)
        if not only_eval:
            validate_embodied_sac_model_type(self.cfg.actor.model, self.cfg.algorithm)
        self.setup_model_and_optimizer(initialize_target=True)
        if not only_eval:
            self.setup_sac_components()
            self.soft_update_target_model(tau=1.0)
        if self.use_dsrl:
            self._init_target_shadow()
        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            # CRIT-R8: under ``only_eval`` the optimizers are None
            # because we skipped ``build_optimizers`` (no training-only
            # config sections required). ``offload_optimizer()`` would
            # dereference ``self.optimizer.state`` → AttributeError.
            # Param offload is still valid (the model exists).
            if not only_eval:
                self.offload_optimizer()
        if self.cfg.actor.get("compile_model", False):
            self.model = torch.compile(
                self.model, mode="default"
            )  # max-autotune-no-cudagraphs
            self.target_model = torch.compile(self.target_model, mode="default")

    def setup_model_and_optimizer(self, initialize_target=False) -> None:
        """Setup model, lr_scheduler, optimizer and grad_scaler."""
        """Add initializing target model logic."""
        module = self.model_provider_func()
        if initialize_target:
            target_module = self.model_provider_func()

        # Enable gradient checkpointing if configured
        if self.cfg.actor.model.get("gradient_checkpointing", False):
            self.logger.info("[FSDP] Enabling gradient checkpointing")
            module.gradient_checkpointing_enable()
            if initialize_target:
                target_module.gradient_checkpointing_enable()
        else:
            self.logger.info("[FSDP] Gradient checkpointing is disabled")

        # build model, optimizer, lr_scheduler, grad_scaler
        self.model = self._strategy.wrap_model(
            model=module, device_mesh=self._device_mesh
        )
        # When precision is null (e.g. Pi0), detect actual dtype from wrapped model
        if self.torch_dtype is None:
            self.torch_dtype = next(self.model.parameters()).dtype
        if initialize_target:
            self.target_model = self._strategy.wrap_model(
                model=target_module, device_mesh=self._device_mesh
            )
            self.target_model.requires_grad_(False)
            self.target_model_initialized = True

        self.use_dsrl = self.cfg.actor.model.get("openpi", {}).get("use_dsrl", False)
        # MAJ-R4-7: SAC training-only initialization MUST be skipped under
        # ``runner.only_eval``. The eval config schema does not require
        # ``actor.critic_optim`` / ``actor.optim`` / ``algorithm.entropy_tuning``
        # sections (e.g. ``realworld_eval.yaml``), so attempting to build
        # optimizers / entropy temperature would crash with KeyError before
        # reaching the deliberately-skipped ``setup_sac_components``.
        if self.cfg.runner.only_eval:
            self.optimizer = None
            self.qf_optimizer = None
        else:
            use_dsrl = self.use_dsrl
            if use_dsrl:
                # DSRL: separate actor/critic encoders into different optimizer groups
                param_filters = {
                    "critic": ["critic_image_encoder", "critic_state_encoder", "q_head"]
                }
            else:
                param_filters = {"critic": ["encoders", "encoder", "q_head", "state_proj"]}
            filtered_optim_config = {"critic": self.cfg.actor.critic_optim}
            optimizers = self.build_optimizers(
                model=self.model,
                main_optim_config=self.cfg.actor.optim,
                param_filters=param_filters,
                filtered_optim_config=filtered_optim_config,
            )
            self.optimizer = optimizers[0]
            self.qf_optimizer = optimizers[1]

            # SAC alpha
            # Initialize temperature parameter for automatic entropy tuning
            alpha_type = self.cfg.algorithm.entropy_tuning.get(
                "alpha_type", "softplus"
            )  # supported type: ["softplus","exp","fixed_alpha"]
            self.entropy_temp = EntropyTemperature(
                initial_alpha=self.cfg.algorithm.entropy_tuning.get("initial_alpha", 0.01),
                alpha_type=alpha_type,
                device=self.device,
                dtype=self.torch_dtype,
            )
            if alpha_type != "fixed_alpha":
                self.target_entropy = self.cfg.algorithm.entropy_tuning.get(
                    "target_entropy",
                    -self.cfg.actor.model.action_dim,
                )

                self.alpha_optimizer = torch.optim.Adam(
                    self.entropy_temp.parameters(),
                    lr=self.cfg.algorithm.entropy_tuning.optim.lr,
                )

            self.build_lr_schedulers()

        self.grad_scaler = self.build_grad_scaler(
            self.cfg.actor.fsdp_config.grad_scaler
        )

    def build_lr_schedulers(self):
        self.lr_scheduler = self.build_lr_scheduler(
            self.optimizer, self.cfg.actor.optim
        )
        self.qf_lr_scheduler = self.build_lr_scheduler(
            self.qf_optimizer, self.cfg.actor.critic_optim
        )
        if self.alpha_optimizer is not None:
            self.alpha_lr_scheduler = self.build_lr_scheduler(
                self.alpha_optimizer, self.cfg.algorithm.entropy_tuning.optim
            )

    def setup_sac_components(self):
        """Initialize SAC-specific components"""
        # Initialize replay buffer
        seed = self.cfg.actor.get("seed", 1234)
        auto_save_path = self.cfg.algorithm.replay_buffer.get("auto_save_path", None)
        if auto_save_path is None:
            auto_save_path = os.path.join(
                self.cfg.runner.logger.log_path, f"replay_buffer/rank_{self._rank}"
            )
        else:
            auto_save_path = os.path.join(auto_save_path, f"rank_{self._rank}")
        self.replay_buffer = TrajectoryReplayBuffer(
            seed=seed,
            enable_cache=self.cfg.algorithm.replay_buffer.enable_cache,
            cache_size=self.cfg.algorithm.replay_buffer.cache_size,
            sample_window_size=self.cfg.algorithm.replay_buffer.sample_window_size,
            auto_save=self.cfg.algorithm.replay_buffer.get("auto_save", False),
            auto_save_path=auto_save_path,
            trajectory_format=self.cfg.algorithm.replay_buffer.get(
                "trajectory_format", "pt"
            ),
        )

        min_demo_buffer_size = 0
        if self.cfg.algorithm.get("demo_buffer", None) is not None:
            auto_save_path = self.cfg.algorithm.demo_buffer.get("auto_save_path", None)
            if auto_save_path is None:
                auto_save_path = os.path.join(
                    self.cfg.runner.logger.log_path, f"demo_buffer/rank_{self._rank}"
                )
            else:
                auto_save_path = os.path.join(auto_save_path, f"rank_{self._rank}")
            self.demo_buffer = TrajectoryReplayBuffer(
                seed=seed,
                enable_cache=self.cfg.algorithm.demo_buffer.enable_cache,
                cache_size=self.cfg.algorithm.demo_buffer.cache_size,
                sample_window_size=self.cfg.algorithm.demo_buffer.sample_window_size,
                auto_save=self.cfg.algorithm.demo_buffer.get("auto_save", False),
                auto_save_path=auto_save_path,
                trajectory_format="pt",
            )
            min_demo_buffer_size = self.cfg.algorithm.demo_buffer.min_buffer_size
            if self.cfg.algorithm.demo_buffer.get("load_path", None) is not None:
                load_path = self.cfg.algorithm.demo_buffer.load_path
                self.demo_buffer.load_checkpoint(
                    load_path,
                    **demo_buffer_load_kwargs(
                        self.cfg.algorithm.demo_buffer,
                        rank=self._rank,
                        world_size=self._world_size,
                    ),
                )
                validate_loaded_demo_buffer(
                    self.demo_buffer,
                    load_path=load_path,
                    min_demo_buffer_size=min_demo_buffer_size,
                    rank=self._rank,
                    world_size=self._world_size,
                    model_cfg=self.cfg.actor.model,
                    load_mode=validate_demo_buffer_load_mode(
                        self.cfg.algorithm.demo_buffer
                    ),
                )

        if self.cfg.algorithm.replay_buffer.get("enable_preload", False):
            buffer_dataset_cls = PreloadReplayBufferDataset
        else:
            buffer_dataset_cls = ReplayBufferDataset
        self.buffer_dataset = buffer_dataset_cls(
            replay_buffer=self.replay_buffer,
            demo_buffer=self.demo_buffer,
            batch_size=self.cfg.actor.global_batch_size // self._world_size,
            min_replay_buffer_size=self.cfg.algorithm.replay_buffer.min_buffer_size,
            min_demo_buffer_size=min_demo_buffer_size,
            prefetch_size=self.cfg.algorithm.replay_buffer.get("prefetch_size", 10),
        )
        self.buffer_dataloader = DataLoader(
            self.buffer_dataset,
            batch_size=1,
            num_workers=0,
            drop_last=True,
            collate_fn=replay_buffer_collate_fn,
        )
        self.buffer_dataloader_iter = iter(self.buffer_dataloader)

        self.critic_actor_ratio = self.cfg.algorithm.get("critic_actor_ratio", 1)
        self.critic_subsample_size = self.cfg.algorithm.get("critic_subsample_size", -1)
        self.critic_sample_generator = torch.Generator(self.device)
        self.critic_sample_generator.manual_seed(seed)

        self.target_update_type = self.cfg.algorithm.get("target_update_type", "all")
        assert self.target_update_type in ["all", "q_head_only"], (
            f"{self.target_update_type=} is not suppported!"
        )

    def _init_target_shadow(self):
        """Create persistent float32 shadow of target model parameters.

        bfloat16 has only 7 mantissa bits (ULP ~0.002 at magnitude 0.3).
        With tau=0.005, per-step EMA delta can be smaller than ULP/2, so
        storing back to bf16 each step rounds away the update. The shadow
        keeps the accumulated EMA state in float32 (ULP ~3.6e-8) across
        steps, preventing precision loss.
        """
        self._target_shadow_f32 = {}
        for name, param in self.target_model.named_parameters():
            self._target_shadow_f32[name] = param.data.float().clone()

    def soft_update_target_model(self, tau: Optional[float] = None):
        """Soft update target model parameters.

        For DSRL (bfloat16 models), uses a persistent float32 shadow buffer
        to prevent EMA precision loss. For non-DSRL SAC, uses direct EMA
        on model parameters.
        """
        if tau is None:
            tau = self.cfg.algorithm.tau

        assert self.target_model_initialized

        with torch.no_grad():
            if not hasattr(self, "_target_shadow_f32"):
                # Non-DSRL path (or before shadow init): direct EMA update
                for (name1, online_param), (name2, target_param) in zip(
                    self.model.named_parameters(),
                    self.target_model.named_parameters(),
                ):
                    assert name1 == name2
                    if "q_head" not in name1:
                        if self.target_update_type == "all":
                            target_param.data.mul_(1.0 - tau)
                            target_param.data.add_(online_param.data * tau)
                        else:
                            target_param.data.mul_(0.0)
                            target_param.data.add_(online_param.data)
                    else:
                        target_param.data.mul_(1.0 - tau)
                        target_param.data.add_(online_param.data * tau)
            else:
                # DSRL path: float32 shadow buffer for bf16 precision
                for (name1, online_param), (name2, target_param) in zip(
                    self.model.named_parameters(),
                    self.target_model.named_parameters(),
                ):
                    assert name1 == name2
                    if "q_head" not in name1 and self.target_update_type != "all":
                        shadow = self._target_shadow_f32[name1]
                        shadow.copy_(online_param.data.float())
                        target_param.data.copy_(shadow.to(target_param.data.dtype))
                    else:
                        shadow = self._target_shadow_f32[name1]
                        shadow.mul_(1.0 - tau).add_(
                            online_param.data.float(), alpha=tau
                        )
                        target_param.data.copy_(shadow.to(target_param.data.dtype))

    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        """
        Receive rollout trajectories from rollout workers.

        Args:
            input_channel: The input channel to read from.
        """
        clear_memory(sync=False)

        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []

        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)

        self.replay_buffer.add_trajectories(recv_list)

        if self.demo_buffer is not None:
            intervene_traj_list = []
            for traj in recv_list:
                assert isinstance(traj, Trajectory)
                intervene_trajs = traj.extract_intervene_traj()
                if intervene_trajs is not None:
                    intervene_traj_list.extend(intervene_trajs)

            if len(intervene_traj_list) > 0:
                self.demo_buffer.add_trajectories(intervene_traj_list)

    @Worker.timer("forward_critic")
    def forward_critic(self, batch):
        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"
        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")
        agg_q = self.cfg.algorithm.get("agg_q", "min")
        use_dsrl = self.cfg.actor.model.get("openpi", {}).get("use_dsrl", False)
        if use_dsrl:
            num_action_chunks = self.cfg.actor.model.get("num_action_chunks", 1)
            discount = self.cfg.algorithm.gamma**num_action_chunks
            rewards_for_bootstrap = batch["rewards"][:, 0:1].to(self.torch_dtype)
        else:
            discount = self.cfg.algorithm.gamma
            rewards_for_bootstrap = (
                batch["rewards"].sum(dim=-1, keepdim=True).to(self.torch_dtype)
            )
        terminations = batch["terminations"].to(self.torch_dtype)

        curr_obs = batch["curr_obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]

        with torch.no_grad():
            kwargs = {}
            if SupportedModel(self.cfg.actor.model.model_type) in [
                SupportedModel.OPENVLA,
                SupportedModel.OPENVLA_OFT,
            ]:
                kwargs["temperature"] = (
                    self.cfg.algorithm.sampling_params.temperature_train
                )
            if use_dsrl:
                kwargs["train"] = True
            next_state_actions, next_state_log_pi, shared_feature = self.model(
                forward_type=ForwardType.SAC, obs=next_obs, **kwargs
            )
            if next_state_log_pi.ndim == 1:
                next_state_log_pi = next_state_log_pi.unsqueeze(-1)
            next_state_log_pi = next_state_log_pi.sum(dim=-1, keepdim=True)
            if not use_crossq:
                dsrl_kwargs = {"train": True} if use_dsrl else {}
                all_qf_next_target = self.target_model(
                    forward_type=ForwardType.SAC_Q,
                    obs=next_obs,
                    actions=next_state_actions,
                    shared_feature=None,
                    **dsrl_kwargs,
                )
                if self.critic_subsample_size > 0:
                    sample_idx = torch.randint(
                        0,
                        all_qf_next_target.shape[-1],
                        (self.critic_subsample_size,),
                        generator=self.critic_sample_generator,
                        device=self.device,
                    )
                    all_qf_next_target = all_qf_next_target.index_select(
                        dim=-1, index=sample_idx
                    )

                if agg_q == "min":
                    qf_next_target, _ = torch.min(
                        all_qf_next_target, dim=1, keepdim=True
                    )
                elif agg_q == "mean":
                    qf_next_target = torch.mean(all_qf_next_target, dim=1, keepdim=True)

                if self.cfg.algorithm.get("backup_entropy", True):
                    qf_next_target = (
                        qf_next_target - self.entropy_temp.alpha * next_state_log_pi
                    )
                    qf_next_target = qf_next_target.to(dtype=self.torch_dtype)
                if bootstrap_type == "always":
                    target_q_values = (
                        rewards_for_bootstrap + discount * qf_next_target
                    )  # [bsz, 1]
                elif bootstrap_type == "standard":
                    target_q_values = (
                        rewards_for_bootstrap
                        + (~(terminations.any(dim=-1, keepdim=True)))
                        * discount
                        * qf_next_target
                    )  # [bsz, 1]
                else:
                    raise NotImplementedError(f"{bootstrap_type=} is not supported!")

        if not use_crossq:
            dsrl_kwargs = {"train": True} if use_dsrl else {}
            all_data_q_values = self.model(
                forward_type=ForwardType.SAC_Q,
                obs=curr_obs,
                actions=actions,
                **dsrl_kwargs,
            )
        else:
            all_data_q_values, all_qf_next = self.model(
                forward_type=ForwardType.CROSSQ_Q,
                obs=curr_obs,
                actions=actions,
                next_obs=next_obs,
                next_actions=next_state_actions,
            )

            all_qf_next = all_qf_next.detach()
            if agg_q == "min":
                qf_next, _ = torch.min(all_qf_next, dim=1, keepdim=True)
            elif agg_q == "mean":
                qf_next = torch.mean(all_qf_next, dim=1, keepdim=True)
            if self.cfg.algorithm.get("backup_entropy", True):
                qf_next = qf_next - self.entropy_temp.alpha * next_state_log_pi
                qf_next = qf_next.to(dtype=self.torch_dtype)

            if bootstrap_type == "always":
                target_q_values = rewards_for_bootstrap + discount * qf_next  # [bsz, 1]
            elif bootstrap_type == "standard":
                target_q_values = (
                    rewards_for_bootstrap
                    + (~(terminations.any(dim=-1, keepdim=True))) * discount * qf_next
                )  # [bsz, 1]
            else:
                raise NotImplementedError(f"{bootstrap_type=} is not supported!")

        # Align dtype: bool ops with Python floats promote to float32,
        # which can mismatch with bfloat16 model outputs.
        target_q_values = target_q_values.to(dtype=all_data_q_values.dtype)
        per_element_se = F.mse_loss(
            all_data_q_values,
            target_q_values.expand_as(all_data_q_values),
            reduction="none",
        )
        weights = chunk_sample_weights(batch, per_element_se.dtype)
        critic_loss, valid_count = apply_chunk_mask(per_element_se, weights)
        return critic_loss, valid_count, {"q_data": all_data_q_values.mean().item()}

    @Worker.timer("forward_actor")
    def forward_actor(self, batch):
        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"
        if "actor_agg_q" in self.cfg.algorithm:
            agg_q = self.cfg.algorithm["actor_agg_q"]
        else:
            agg_q = self.cfg.algorithm.get("agg_q", "min")

        curr_obs = batch["curr_obs"]
        kwargs = {}
        if self.cfg.actor.model.model_type in ["openvla", "openvla_oft"]:
            kwargs["temperature"] = self.cfg.algorithm.sampling_params.temperature_train
        if self.use_dsrl:
            kwargs["train"] = True
        pi, log_pi, shared_feature = self.model(
            forward_type=ForwardType.SAC, obs=curr_obs, **kwargs
        )
        if log_pi.ndim == 1:
            log_pi = log_pi.unsqueeze(-1)
        log_pi = log_pi.sum(dim=-1, keepdim=True)  # sum over the chunk dimension
        if not use_crossq:
            dsrl_kwargs = {"train": True} if self.use_dsrl else {}
            all_qf_pi = self.model(
                forward_type=ForwardType.SAC_Q,
                obs=curr_obs,
                actions=pi,
                shared_feature=None,
                detach_encoder=True,
                **dsrl_kwargs,
            )
        else:
            all_qf_pi, _ = self.model(
                forward_type=ForwardType.CROSSQ_Q,
                obs=curr_obs,
                actions=pi,
                next_obs=None,
                next_actions=None,
                shared_feature=None,
                detach_encoder=True,
            )
        metrics = {
            f"q_value_{q_id}": all_qf_pi[..., q_id].mean().item()
            for q_id in range(self.cfg.actor.model.get("num_q_heads", 2))
        }
        if agg_q == "min":
            qf_pi, _ = torch.min(all_qf_pi, dim=1, keepdim=True)
        elif agg_q == "mean":
            qf_pi = torch.mean(all_qf_pi, dim=1, keepdim=True)
        metrics["q_pi"] = qf_pi.mean().item()
        actor_loss_per = (self.entropy_temp.alpha * log_pi) - qf_pi
        weights = chunk_sample_weights(batch, actor_loss_per.dtype)
        actor_loss, valid_count = apply_chunk_mask(actor_loss_per, weights)
        entropy, _ = apply_chunk_mask(-log_pi, weights)
        return actor_loss, entropy, valid_count, metrics

    @Worker.timer("forward_alpha")
    def forward_alpha(self, batch):
        curr_obs = batch["curr_obs"]
        with torch.no_grad():
            kwargs = {}
            if self.cfg.actor.model.model_type in ["openvla", "openvla_oft"]:
                kwargs["temperature"] = (
                    self.cfg.algorithm.sampling_params.temperature_train
                )
            if self.use_dsrl:
                kwargs["train"] = True
            _, log_pi, _ = self.model(
                forward_type=ForwardType.SAC, obs=curr_obs, **kwargs
            )
            if log_pi.ndim == 1:
                log_pi = log_pi.unsqueeze(-1)
            log_pi = log_pi.sum(dim=-1, keepdim=True)

        alpha = self.entropy_temp.compute_alpha()
        weights = chunk_sample_weights(batch, log_pi.dtype)
        log_pi_mean, valid_count = apply_chunk_mask(log_pi, weights)
        alpha_loss = -alpha * (log_pi_mean + self.target_entropy)
        if weights is not None:
            # Avoid letting `target_entropy` move alpha when the entire batch
            # was padded; `apply_chunk_mask` returns 0 in that case, but the
            # `(0 + target_entropy)` term would still drive a spurious update.
            alpha_loss = alpha_loss * (weights.sum() > 0).to(alpha_loss.dtype)
        return alpha_loss, valid_count

    @Worker.timer("update_one_epoch")
    def update_one_epoch(self, train_actor: bool = True):
        global_batch_size_per_rank = (
            self.cfg.actor.global_batch_size // self._world_size
        )

        with self.worker_timer("sample"):
            global_batch = next(self.buffer_dataloader_iter)

        train_micro_batch_list = split_dict_to_chunk(
            global_batch,
            global_batch_size_per_rank // self.cfg.actor.micro_batch_size,
        )

        # move train_micro_batch_list to device and apply DRQ for critic/actor/alpha passes
        for i, batch in enumerate(train_micro_batch_list):
            batch = put_tensor_device(batch, device=self.device)
            if self.enable_drq:
                drq.apply_drq(batch["curr_obs"], pad=4)
                drq.apply_drq(batch["next_obs"], pad=4)
            train_micro_batch_list[i] = batch

        self.qf_optimizer.zero_grad()
        gbs_critic_loss = []
        all_critic_metrics = {}
        critic_valid_local = 0
        for batch in train_micro_batch_list:
            critic_loss, critic_valid, critic_metrics = self.forward_critic(batch)
            critic_loss = critic_loss / self.gradient_accumulation
            # ALWAYS call backward — `apply_chunk_mask` now returns a
            # graph-connected zero when local valid==0, so backward is safe
            # and FSDP collectives stay consistent across ranks. Otherwise
            # the rank-local skip would deadlock peers that did call backward.
            critic_loss.backward()
            gbs_critic_loss.append(critic_loss.item() * self.gradient_accumulation)
            critic_valid_local += int(critic_valid)
            append_to_dict(all_critic_metrics, critic_metrics)
        all_critic_metrics = {
            f"critic/{key}": np.mean(value) for key, value in all_critic_metrics.items()
        }
        # FSDP / SAC collectives (clip_grad_norm_, optimizer.step on sharded
        # params, alpha all_reduce) MUST be entered identically on every rank
        # or the process group will deadlock. Gate them on the GLOBAL valid
        # count, not the per-rank local count. Ranks with locally zero valid
        # samples still enter every collective with zero grads — clip and
        # step are no-ops for zero grads but stay synchronised.
        critic_valid_total = self._global_valid_count(critic_valid_local)
        if critic_valid_total > 0:
            qf_grad_norm = self.model.clip_grad_norm_(
                max_norm=self.cfg.actor.critic_optim.clip_grad
            )

            self.qf_optimizer.step()
            self.qf_lr_scheduler.step()
        else:
            # Every rank saw only padded chunks -- skip optimizer.step() and
            # scheduler.step() so weight decay / momentum / LR schedule don't
            # advance on a synthetic zero loss.
            qf_grad_norm = 0.0

        metrics_data = {
            "sac/critic_loss": np.mean(gbs_critic_loss),
            "critic/lr": self.qf_optimizer.param_groups[0]["lr"],
            "critic/grad_norm": qf_grad_norm,
            "critic/valid_count": critic_valid_total,
            **all_critic_metrics,
        }

        actor_valid_total = 0
        if self.update_step % self.critic_actor_ratio == 0 and train_actor:
            # MAJ-R4-5: also zero the critic optimizer's params before the
            # actor pass. ``optimizer.step()`` does NOT clear grads, so
            # stale critic grads on shared ``q_head`` / ``q_target`` params
            # would otherwise contaminate the model-wide
            # ``clip_grad_norm_`` and the optimizer step below.
            self.qf_optimizer.zero_grad()
            self.optimizer.zero_grad()
            gbs_actor_loss = []
            gbs_entropy = []
            actor_valid_local = 0
            all_actor_metrics = {}
            for batch in train_micro_batch_list:
                actor_loss, entropy, actor_valid, q_metrics = self.forward_actor(batch)
                actor_loss = actor_loss / self.gradient_accumulation
                # Always backward; see critic loop comment for the FSDP
                # collective-consistency rationale.
                actor_loss.backward()
                gbs_actor_loss.append(actor_loss.item() * self.gradient_accumulation)
                gbs_entropy.append(entropy.item())
                actor_valid_local += int(actor_valid)
                append_to_dict(all_actor_metrics, q_metrics)
            all_actor_metrics = {
                f"actor/{key}": np.mean(value)
                for key, value in all_actor_metrics.items()
            }
            # Use the global valid count to gate FSDP-collective optimizer/clip
            # calls; otherwise ranks with locally zero valid samples would skip
            # the collective and the others would hang.
            actor_valid_total = self._global_valid_count(actor_valid_local)
            # MAJ-R4-5 (post-backward leg): actor.backward() writes fresh
            # grads on the critic's ``q_head`` params via the q_pi path
            # (encoder is detached but q_head is not). Zero those before
            # the model-wide ``clip_grad_norm_`` so the actor's clip
            # norm is not contaminated by q_head grads that no actor
            # optimizer.step will consume.
            self.qf_optimizer.zero_grad()
            if actor_valid_total > 0:
                actor_grad_norm = self.model.clip_grad_norm_(
                    max_norm=self.cfg.actor.optim.clip_grad
                )
                self.optimizer.step()
                self.lr_scheduler.step()
            else:
                actor_grad_norm = 0.0

            # Update temperature parameter if using automatic entropy tuning
            gbs_alpha_loss = [0]
            alpha_grad_norm = 0
            alpha_valid_total = 0
            if self.alpha_optimizer is not None:
                # MAJ-R4-5 (alpha-side): wipe both model optimizers'
                # accumulated grads so the alpha backward path doesn't
                # accidentally see stale grads via shared FSDP graphs.
                self.qf_optimizer.zero_grad()
                self.optimizer.zero_grad()
                self.alpha_optimizer.zero_grad()
                gbs_alpha_loss = []
                alpha_valid_local = 0
                for batch in train_micro_batch_list:
                    alpha_loss, alpha_valid = self.forward_alpha(batch)
                    alpha_loss = alpha_loss / self.gradient_accumulation
                    # Always backward; see critic loop. This also guarantees
                    # ``self.entropy_temp.base_alpha.grad`` is populated (even
                    # if zero) on every rank, so the subsequent all_reduce
                    # below doesn't crash on local-zero ranks where the grad
                    # would otherwise be None.
                    alpha_loss.backward()
                    gbs_alpha_loss.append(
                        alpha_loss.item() * self.gradient_accumulation
                    )
                    alpha_valid_local += int(alpha_valid)
                # Same FSDP-collective hazard: the alpha all_reduce must be
                # called identically on every rank to avoid deadlock.
                alpha_valid_total = self._global_valid_count(alpha_valid_local)
                if alpha_valid_total > 0:
                    torch.distributed.all_reduce(
                        self.entropy_temp.base_alpha.grad,
                        op=torch.distributed.ReduceOp.AVG,
                    )
                    alpha_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.entropy_temp.base_alpha,
                        self.cfg.algorithm.entropy_tuning.optim.clip_grad,
                    )
                    self.alpha_optimizer.step()
                    self.alpha_lr_scheduler.step()

            # Collect metrics
            metrics_data.update(
                {
                    "sac/actor_loss": np.mean(gbs_actor_loss),
                    "sac/alpha_loss": np.mean(gbs_alpha_loss),
                    "sac/alpha": self.entropy_temp.alpha,
                    "actor/lr": self.optimizer.param_groups[0]["lr"],
                    "actor/grad_norm": actor_grad_norm,
                    "actor/entropy": np.mean(gbs_entropy),
                    "actor/valid_count": actor_valid_total,
                    "alpha/grad_norm": alpha_grad_norm,
                    **all_actor_metrics,
                }
            )
        # Soft update target network. Skip the soft update entirely when EVERY
        # rank saw zero valid samples (critic_valid_total is the global
        # all-reduced value); otherwise the target network drifts toward the
        # current online weights even though no real learning signal was
        # applied this step. Gating on the global value keeps every rank
        # consistent so the soft-update collective is entered identically.
        if (
            self.target_model_initialized
            and critic_valid_total > 0
            and self.update_step % self.cfg.algorithm.get("target_update_freq", 1) == 0
        ):
            self.soft_update_target_model()

        return metrics_data

    def _global_valid_count(self, local_count: int) -> int:
        """All-reduce SUM the local per-rank valid sample count.

        FSDP-wrapped clip_grad_norm_(), optimizer.step() and the alpha
        all_reduce are collective: every rank must enter / skip them in
        lock-step or the process group deadlocks. We therefore gate those
        branches on the GLOBAL valid count, not the per-rank local one.

        Falls back to ``local_count`` for non-distributed runs (single GPU,
        unit tests).
        """
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            count_tensor = torch.tensor(
                [int(local_count)], dtype=torch.long, device=self.device
            )
            torch.distributed.all_reduce(
                count_tensor, op=torch.distributed.ReduceOp.SUM
            )
            return int(count_tensor.item())
        return int(local_count)

    def _global_all_ranks(self, local_bool: bool) -> bool:
        """All-reduce MIN the per-rank bool — True only if every rank is True.

        Used for branching decisions (replay-buffer readiness, actor-warmup)
        that must be taken identically on every rank to keep subsequent
        FSDP collectives in sync. CRIT-R4-2.
        """
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            flag_tensor = torch.tensor(
                [1 if local_bool else 0], dtype=torch.long, device=self.device
            )
            torch.distributed.all_reduce(
                flag_tensor, op=torch.distributed.ReduceOp.MIN
            )
            return bool(int(flag_tensor.item()))
        return bool(local_bool)

    def process_train_metrics(self, metrics):
        replay_buffer_stats = self.replay_buffer.get_stats()
        replay_buffer_stats = {
            f"replay_buffer/{key}": value for key, value in replay_buffer_stats.items()
        }
        append_to_dict(metrics, replay_buffer_stats)

        if self.demo_buffer is not None:
            demo_buffer_stats = self.demo_buffer.get_stats()
            demo_buffer_stats = {
                f"demo_buffer/{key}": value for key, value in demo_buffer_stats.items()
            }
            append_to_dict(metrics, demo_buffer_stats)
        # Average metrics across updates
        mean_metric_dict = {}
        for key, value in metrics.items():
            if isinstance(value, list) and len(value) > 0:
                # Convert tensor values to CPU and detach before computing mean
                cpu_values = []
                for v in value:
                    if isinstance(v, torch.Tensor):
                        cpu_values.append(v.detach().cpu().item())
                    else:
                        cpu_values.append(v)
                mean_metric_dict[key] = np.mean(cpu_values)
            else:
                # Handle single values
                if isinstance(value, torch.Tensor):
                    mean_metric_dict[key] = value.detach().cpu().item()
                else:
                    mean_metric_dict[key] = value

        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )
        return mean_metric_dict

    @Worker.timer("run_training")
    def run_training(self):
        """SAC training using replay buffer"""
        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

        # Check if replay buffer has enough samples. CRIT-R4-2: these
        # branches MUST be globally agreed across all FSDP ranks or the
        # process group deadlocks. One rank returning early while peers
        # enter ``update_one_epoch`` (which calls FSDP-collective
        # backward/clip/all_reduce) hangs the run. We all-reduce the
        # local readiness as an AND so every rank takes the same branch.
        min_buffer_size = self.cfg.algorithm.replay_buffer.get("min_buffer_size", 100)
        local_ready = bool(self.replay_buffer.is_ready(min_buffer_size))
        if not self._global_all_ranks(local_ready):
            self.log_on_first_rank(
                f"Replay buffer size {len(self.replay_buffer)} < {min_buffer_size} "
                "on some rank, skipping training"
            )
            return {}

        # Demo buffer readiness MUST also be globally agreed: the
        # ``EmbodiedBufferDataset`` blocks until BOTH replay and demo are
        # ready (per-rank), so a single rank with a demo shard short of
        # ``min_demo_buffer_size`` would block forever while peers enter
        # FSDP collectives. Gate globally before ``next(dataloader)``.
        if self.demo_buffer is not None:
            min_demo_buffer_size = self.cfg.algorithm.demo_buffer.get(
                "min_buffer_size", 0
            )
            local_demo_ready = bool(
                self.demo_buffer.is_ready(min_demo_buffer_size)
            )
            if not self._global_all_ranks(local_demo_ready):
                self.log_on_first_rank(
                    f"Demo buffer size {len(self.demo_buffer)} < "
                    f"{min_demo_buffer_size} on some rank, skipping training"
                )
                return {}

        # Delay actor training until buffer has enough samples — also
        # globally agreed (AND across ranks) so every rank skips the actor
        # path together.
        train_actor_steps = self.cfg.algorithm.get("train_actor_steps", 0)
        train_actor_steps = max(min_buffer_size, train_actor_steps)
        train_actor = self._global_all_ranks(
            bool(self.replay_buffer.is_ready(train_actor_steps))
        )

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )
        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        self.model.train()
        metrics = {}

        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            metrics_data = self.update_one_epoch(train_actor=train_actor)
            append_to_dict(metrics, metrics_data)
            # MAJ-R4-6: only advance ``update_step`` when the critic
            # optimizer actually stepped (i.e. the GLOBAL valid count
            # was positive). Advancing on an all-padded global batch
            # would skew ``critic_actor_ratio`` and ``target_update_freq``
            # scheduling. ``critic/valid_count`` is the metric we just
            # appended above.
            critic_valid_total = int(metrics_data.get("critic/valid_count", 0))
            if critic_valid_total > 0:
                self.update_step += 1

        mean_metric_dict = self.process_train_metrics(metrics)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return mean_metric_dict

    def compute_advantages_and_returns(self):
        """
        SAC doesn't compute advantages/returns like PPO.
        This method is kept for compatibility but returns empty metrics.
        """
        return {}

    def save_checkpoint(self, save_base_path, step):
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
            self.is_weight_offloaded = False
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)
            self.is_optimizer_offloaded = False

        # Save model
        self._strategy.save_checkpoint(
            model=self.model,
            optimizers=[self.optimizer, self.qf_optimizer],
            lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
            save_path=save_base_path,
            checkpoint_format="local_shard"
            if self.cfg.actor.fsdp_config.use_orig_params
            else "dcp",
        )

        # Save sac components
        # save alpha
        if self.alpha_optimizer is not None:
            alpha_save_path = os.path.join(save_base_path, "sac_components/alpha")
            self._strategy.save_checkpoint(
                model=self.entropy_temp,
                optimizers=self.alpha_optimizer,
                lr_schedulers=self.alpha_lr_scheduler,
                save_path=alpha_save_path,
                save_full_model_weights=False,
            )

        # save target model
        target_model_save_path = os.path.join(
            save_base_path, "sac_components/target_model"
        )
        os.makedirs(target_model_save_path, exist_ok=True)
        target_model_state_dict = self._strategy.get_model_state_dict(
            self.target_model, cpu_offload=False, full_state_dict=True
        )
        torch.save(
            target_model_state_dict,
            os.path.join(target_model_save_path, f"checkpoint_rank_{self._rank}.pt"),
        )

        # save replay buffer
        buffer_save_path = os.path.join(
            save_base_path, f"sac_components/replay_buffer/rank_{self._rank}"
        )
        self.replay_buffer.save_checkpoint(buffer_save_path)

    def load_checkpoint(self, load_base_path):
        # load model
        self._strategy.load_checkpoint(
            model=self.model,
            optimizers=[self.optimizer, self.qf_optimizer],
            lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
            load_path=load_base_path,
            checkpoint_format="local_shard"
            if self.cfg.actor.fsdp_config.use_orig_params
            else "dcp",
        )

        # load alpha
        if self.alpha_optimizer is not None:
            alpha_load_path = os.path.join(load_base_path, "sac_components/alpha")
            self._strategy.load_checkpoint(
                model=self.entropy_temp,
                optimizers=self.alpha_optimizer,
                lr_schedulers=self.alpha_lr_scheduler,
                load_path=alpha_load_path,
            )

        # load target model
        target_model_load_path = os.path.join(
            load_base_path, "sac_components/target_model"
        )
        target_model_state_dict = torch.load(
            os.path.join(target_model_load_path, f"checkpoint_rank_{self._rank}.pt")
        )
        self._strategy.load_model_with_state_dict(
            self.target_model,
            target_model_state_dict,
            cpu_offload=False,
            full_state_dict=True,
        )

        # load replay buffer
        buffer_load_path = os.path.join(
            load_base_path, f"sac_components/replay_buffer/rank_{self._rank}"
        )
        self.replay_buffer.load_checkpoint(buffer_load_path)
