Data Collection
===============

RLinf provides two data collection approaches targeting different downstream use cases:

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Approach
     - Entry Point
     - Typical Use
   * - **Episode Collection**
     - ``CollectEpisode`` wrapper
     - Reward model / value model training data
   * - **Real-robot Replay Buffer Collection**
     - ``collect_data.sh``
     - Real-robot RLPD prior data / policy initialization

----

Episode Data Collection
-----------------------

``CollectEpisode`` is a ``gymnasium.Wrapper`` that transparently wraps any
environment and automatically records step-level data during RL training or
evaluation, saving each completed episode to disk asynchronously.

Two output formats are supported:

- **pickle** — saves the complete raw buffer; suited for custom offline processing.
- **lerobot** — saves structured Parquet files with metadata; directly compatible
  with the LeRobot training pipeline.

Key Features
~~~~~~~~~~~~

- Supports both single environments and vectorized parallel environments
  (``num_envs > 1``).
- Compatible with auto-reset environments: the final pre-reset observation is
  correctly attributed to the current episode, and the post-reset observation is
  carried over to the next episode.
- All write operations run asynchronously in a background thread so they never
  block the RL training loop.
- The LeRobot writer is lazily initialized on the first episode write, with image
  shape, state dimension, and action dimension inferred automatically.
- LeRobot export can store ``image``, ``wrist_image``, and
  ``extra_view_image``. Stacked ``[N, H, W, C]`` wrist/extra-view tensors are
  fanned out by index (for example ``extra_view_image-0``,
  ``extra_view_image-1``, …).
- Set ``only_success=True`` to filter out failed episodes and save disk space.

.. warning::

   **Image channel order: RGB only.** ``CollectEpisode`` writes the raw
   image arrays it receives directly into LeRobot frames; the LeRobot →
   replay-buffer converter accepts those arrays **as-is and assumes
   RGB**. If your env exposes BGR frames (OpenCV / RealWorld cameras
   that natively output BGR), convert BGR → RGB before passing them
   into the env's observation, or the persisted demo buffer will train
   on color-swapped images. There is no runtime detection for channel
   order; this is a contract.

Constructor Arguments
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Argument
     - Type
     - Default
     - Description
   * - ``env``
     - ``gym.Env``
     - —
     - The gymnasium environment to wrap
   * - ``save_dir``
     - ``str``
     - —
     - Directory for saving episode data (created automatically)
   * - ``rank``
     - ``int``
     - ``0``
     - Worker rank for unique file naming in distributed settings
   * - ``num_envs``
     - ``int``
     - ``1``
     - Number of parallel environments
   * - ``show_goal_site``
     - ``bool``
     - ``True``
     - Show goal-site visualization in renders (for environments that support it)
   * - ``export_format``
     - ``str``
     - ``"pickle"``
     - Output format: ``"pickle"`` or ``"lerobot"``
   * - ``robot_type``
     - ``str``
     - ``"panda"``
     - Robot type written to LeRobot metadata (lerobot format only)
   * - ``fps``
     - ``int``
     - ``10``
     - Dataset frame rate written to LeRobot metadata (lerobot format only)
   * - ``only_success``
     - ``bool``
     - ``False``
     - Save only successful episodes
   * - ``finalize_interval``
     - ``int``
     - ``100``
     - Call ``writer.finalize()`` every N completed episodes as a checkpoint (``0`` disables; lerobot format only)

Usage Examples
~~~~~~~~~~~~~~

**Direct Python API:**

.. code-block:: python

   from rlinf.envs.wrappers.collect_episode import CollectEpisode

   env = CollectEpisode(
       env=base_env,
       save_dir="./collected_data",
       num_envs=8,
       export_format="lerobot",   # or "pickle"
       robot_type="panda",
       fps=10,
       only_success=True,
   )

   obs, info = env.reset()
   while not done:
       action = policy(obs)
       obs, reward, terminated, truncated, info = env.step(action)
   env.close()   # triggers final flush and finalize

**Via YAML configuration (simulation training):**

Add a ``data_collection`` block under ``env`` in your YAML config:

.. code-block:: yaml

  env:
    group_name: "EnvGroup"
    enable_offload: False

    eval:
      data_collection:
        enabled: True
        save_dir: ${runner.logger.log_path}/collected_data
        export_format: "lerobot"      # or "pickle"
        only_success: True
        robot_type: "panda"
        fps: 10

Then run the training script as usual; data is collected automatically:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_collect

Data Format Details
~~~~~~~~~~~~~~~~~~~

**pickle format**

Each episode is saved as a separate ``.pkl`` file with the naming convention:

.. code-block:: text

   rank_{rank}_env_{env_idx}_episode_{episode_id}_{success|fail}.pkl

Example: ``rank_0_env_3_episode_42_success.pkl``

The file contains a single dictionary:

.. code-block:: python

   {
       "rank":        int,   # worker rank
       "env_idx":     int,   # environment index
       "episode_id":  int,   # episode counter (per-env, monotonically increasing)
       "success":     bool,  # whether the episode succeeded
       "observations": list, # length = num_steps + 1 (includes the initial reset obs)
       "actions":     list,  # length = num_steps
       "rewards":     list,  # length = num_steps
       "terminated":  list,  # length = num_steps
       "truncated":   list,  # length = num_steps
       "infos":       list,  # length = num_steps
   }

.. note::

   The pickle format preserves the raw buffer exactly as recorded, making it
   suitable for custom offline RL or behaviour analysis pipelines.
   Index 0 in ``observations`` comes from ``reset()``; indices 1 through N come
   from successive ``step()`` calls.

**LeRobot format**

Data is stored as Parquet files alongside JSON metadata files. The
``CollectEpisode`` wrapper creates a separate LeRobot dataset root per
worker / writer batch under
``save_dir/rank_{rank}/id_{episodes_written}/``:

.. code-block:: text

   save_dir/
   └── rank_0/                   # one subdir per worker rank
       ├── id_0/                 # one subdir per writer batch (rotated via finalize_interval)
       │   ├── meta/
       │   │   ├── info.json           # dataset metadata (fps, robot_type, dimensions, …)
       │   │   ├── episodes.jsonl      # per-episode length and task description
       │   │   ├── tasks.jsonl         # deduplicated task list
       │   │   └── stats.json          # mean / std statistics for observations and actions
       │   └── data/
       │       └── chunk-000/
       │           ├── episode_000000.parquet
       │           ├── episode_000001.parquet
       │           └── ...
       └── id_1/                 # next batch after finalize_interval episodes
           └── ...

.. note::

   ``rank_*`` differentiates writers in distributed/vectorized rollouts.
   ``id_*`` rotates whenever ``writer.finalize()`` is invoked
   (e.g. every ``finalize_interval`` episodes), so a single rank may emit
   multiple LeRobot roots.

Parquet column schema:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Description
   * - ``image``
     - Main camera image (bytes + path), uint8
   * - ``next_image``
     - Next-step main camera image for replay-buffer conversion, uint8
   * - ``wrist_image`` / ``wrist_image-N``
     - Wrist camera image (bytes + path), uint8. Multi-view stacks are fanned
       out into ``wrist_image-0``, ``wrist_image-1``, ….
   * - ``next_wrist_image`` / ``next_wrist_image-N``
     - Next-step wrist camera image, using the same multi-view fan-out rule
   * - ``extra_view_image`` / ``extra_view_image-N``
     - Auxiliary camera image (bytes + path), uint8. Multi-view stacks are
       fanned out into ``extra_view_image-0``, ``extra_view_image-1``, …;
       empty when no extra view is present.
   * - ``next_extra_view_image`` / ``next_extra_view_image-N``
     - Next-step auxiliary camera image, using the same multi-view fan-out rule
   * - ``state``
     - Robot state vector, ``float32[state_dim]``
   * - ``next_state``
     - Next-step robot state vector, ``float32[state_dim]``
   * - ``actions``
     - Action vector, ``float32[action_dim]``
   * - ``rewards``
     - Transition reward aligned with ``actions``, ``float32[1]``
   * - ``timestamp``
     - Frame timestamp in seconds, ``float``
   * - ``frame_index``
     - Frame index within the episode, ``int64``
   * - ``episode_index``
     - Global episode index, ``int64``
   * - ``index``
     - Global frame index, ``int64``
   * - ``task_index``
     - Task index (references tasks.jsonl), ``int64``
   * - ``done``
     - Per-step done flag, ``bool`` (``True`` on the last step of each episode)
   * - ``terminated`` / ``truncated``
     - Gymnasium termination and truncation flags aligned with ``actions``
   * - ``is_success``
     - Whether the episode succeeded, ``bool``

Observation key lookup order (first match wins):

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Keys checked (in priority order)
   * - Main image
     - ``main_images`` → ``image`` → ``full_image``
   * - Wrist image
     - ``wrist_images`` → ``wrist_image`` (``[N, H, W, C]`` stacks fan out to
       ``wrist_image-0``, ``wrist_image-1``, …)
   * - Extra-view image
     - ``extra_view_images`` → ``extra_view_image`` (``[N, H, W, C]`` stacks
       fan out to ``extra_view_image-0``, ``extra_view_image-1``, …)
   * - State
     - ``states`` → ``state``

Images are automatically converted to uint8 (float [0, 1] arrays are multiplied
by 255; out-of-range arrays are cast directly).

All-null optional image columns are treated as absent during replay-buffer
conversion. Partially-null optional image columns fail fast so camera schemas do
not silently change inside one episode.

Success Detection Logic
~~~~~~~~~~~~~~~~~~~~~~~

The wrapper scans ``info`` dicts in reverse step order (most recent first). For
each step's info dict, it checks three sources in order —
``final_info`` → ``episode`` → root info — and within each source looks for keys
in order ``success_once`` → ``success_at_end`` → ``success``:

1. ``info["final_info"]["success_once"]`` / ``success_at_end`` / ``success``
2. ``info["episode"]["success_once"]`` / ``success_at_end`` / ``success``
3. ``info["success_once"]`` / ``info["success_at_end"]`` / ``info["success"]``

If none of the above keys is found across all recorded steps, the wrapper falls
back to the incrementally maintained ``_episode_success`` flag updated at each
step.

----

Real-robot Replay Buffer Collection
------------------------------------

Real-robot collection is used for RLPD (Reinforcement Learning from Prior Data)
or policy initialization. An operator uses a SpaceMouse or GELLO device to
demonstrate successful task completions; data is saved in
``TrajectoryReplayBuffer`` format for direct use in subsequent real-robot training.

Unlike large-scale parallel simulation collection, real-robot collection runs on
a single control node and stops automatically once the target number of successful
demonstrations is reached.

Core Components
~~~~~~~~~~~~~~~

- **Entry script**: ``examples/embodiment/collect_data.sh``
- **Collection logic**: ``examples/embodiment/collect_real_data.py`` (``DataCollector`` class)
- **Config file**: ``examples/embodiment/config/realworld_collect_data.yaml``

``DataCollector`` workflow:

1. Initialise ``RealWorldEnv`` and ``TrajectoryReplayBuffer``.
2. Loop over steps, reading the SpaceMouse intervention action from
   ``info["intervene_action"]``.
3. Construct a ``ChunkStepResult`` and append it to ``EmbodiedRolloutResult``.
4. When an episode ends (``done=True``) with reward ``>= 0.5``, count it as a
   success and write the trajectory to the buffer.
5. Stop automatically once ``num_data_episodes`` successes have been collected
   and finalise the buffer.

Configuration Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 42 15 43

   * - Parameter
     - Default
     - Description
   * - ``runner.num_data_episodes``
     - ``20``
     - Target number of successful demonstrations; stops when reached
   * - ``cluster.node_groups[0].hardware.configs[0].robot_ip``
     - —
     - IP address of the Franka robot. ``node_groups`` and ``configs`` are
       YAML lists; see :ref:`realworld-collect-yaml-shape` below.
   * - ``env.eval.use_spacemouse``
     - ``True``
     - Enable SpaceMouse intervention
   * - ``env.eval.no_gripper``
     - ``True`` (wrapper default when key is absent;
       not set by the shipped config)
     - Whether the real-world env uses a 6-DoF action without a gripper
       dimension. The single-arm wrapper applies ``GripperCloseEnv`` when
       this is True.
   * - ``env.eval.use_gello``
     - ``False``
     - Enable GELLO teleoperation (mutually exclusive with SpaceMouse)
   * - ``env.eval.gello_port``
     - —
     - Serial port of the GELLO device (required when ``use_gello`` is ``True``)
   * - ``env.eval.override_cfg.target_ee_pose``
     - —
     - Target end-effector pose ``[x, y, z, rx, ry, rz]``
   * - ``env.eval.override_cfg.success_hold_steps``
     - ``1``
     - Number of consecutive steps at goal pose required to declare success
   * - ``runner.record_task_description``
     - ``False`` (per shipped ``realworld_collect_data.yaml``)
     - Whether to include the task description string in observations

Data Format
~~~~~~~~~~~

After collection, data is saved to:

.. code-block:: text

   logs/{timestamp}/demos/

``TrajectoryReplayBuffer`` stores each trajectory as a ``.pt`` file.
Each saved ``.pt`` file holds a ``Trajectory`` dataclass
(:class:`rlinf.data.embodied_io_struct.Trajectory`) whose **top-level**
fields are serialized directly — there is no ``"transitions"`` wrapper:

.. code-block:: python

   # rlinf.data.embodied_io_struct.Trajectory
   Trajectory(
       max_episode_length=int,             # scalar
       model_weights_id=str,               # uuid derived from model versions
       actions=torch.Tensor,               # [T, B, action_dim] (or with chunk dim)
       intervene_flags=torch.Tensor,       # [T, B, ...] bool, per-step intervention
       rewards=torch.Tensor,               # [T, B, 1]
       terminations=torch.Tensor,          # [T, B, 1] bool
       truncations=torch.Tensor,           # [T, B, 1] bool
       dones=torch.Tensor,                 # [T, B, 1] bool
       loss_mask=Optional[torch.Tensor],   # [T, B, num_action_chunks] when set
       forward_inputs=dict,                # producer-specific forward kwargs
       curr_obs={                          # observation dict at step t
           "states":      ...,             # robot state, e.g. [T, B, 19]
           "main_images": ...,              # main camera, e.g. [T, B, H, W, C] uint8
           # other observation keys (wrist_images, extra_view_images, ...)
       },
       next_obs={                          # observation dict at step t+1
           "states":      ...,
           "main_images": ...,
       },
   )

.. note::

   ``intervene_flags`` semantics differ between the in-flight rollout and
   the persisted real-robot demo trajectory:

   - During the rollout (``EmbodiedRolloutResult``) the flags are
     **per-step booleans** sourced from ``info["intervene_flag"]`` (e.g.
     the SpaceMouse / GELLO intervention path).
   - When ``examples/embodiment/collect_real_data.py`` writes a
     successful trajectory to the replay buffer, every step is
     **overwritten to True** via ``torch.ones_like(...)`` so the
     trajectory is treated as fully expert by RLPD/SAC-demo training. This
     matches the operator contract: a saved successful real-robot demo is
     intended as expert teleop data regardless of which substeps the
     operator nudged.

   If your training expects the original per-step flags from a real-robot
   collect, read them from the rollout-side ``Trajectory`` before the
   ``collect_real_data`` post-processing, not from the persisted demo
   buffer.

Collect Replay Buffer And LeRobot Data Together
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``examples/embodiment/collect_real_data.py`` now supports writing the real-robot
replay buffer and the ``CollectEpisode`` export in the same run. With
``env.eval.data_collection.enabled=True``, successful demonstrations are saved twice:

- ``logs/{timestamp}/demos/`` as ``TrajectoryReplayBuffer`` trajectories for RLPD
- ``logs/{timestamp}/collected_data/`` as episode files in ``pickle`` or LeRobot format

To collect LeRobot-format data while still building the replay buffer, keep the
real-world collection config like this:

.. code-block:: yaml

   env:
    eval:
     data_collection:
       enabled: True
       save_dir: ${runner.logger.log_path}/collected_data
       export_format: "lerobot"
       only_success: True
       robot_type: "panda"
       fps: 10

Usage Steps
~~~~~~~~~~~

1. Activate the environment on the control node:

   .. code-block:: bash

      source <path_to_your_venv>/bin/activate

2. Edit ``examples/embodiment/config/realworld_collect_data.yaml`` to replace
   ``ROBOT_IP`` and ``TARGET_EE_POSE`` with your actual robot IP and target pose.

   .. _realworld-collect-yaml-shape:

   ``cluster.node_groups`` is a YAML list, and each entry's
   ``hardware.configs`` is also a list (one entry per robot on that node
   group). Match the structure of the shipped
   ``examples/embodiment/config/realworld_collect_data.yaml``:

   .. code-block:: yaml

      cluster:
        num_nodes: 1
        component_placement:
          env:
            node_group: franka
            placement: 0
        node_groups:
          - label: franka
            node_ranks: 0
            hardware:
              type: Franka
              configs:
                - robot_ip: "192.168.1.100"   # replace with actual IP
                  node_rank: 0

      env:
        eval:
          use_spacemouse: True
          override_cfg:
            target_ee_pose: [0.5, 0.0, 0.3, 0.0, 3.14, 0.0]
            success_hold_steps: 3

      runner:
        num_data_episodes: 50

3. Launch collection (an optional first argument overrides the config name):

   .. code-block:: bash

      bash examples/embodiment/collect_data.sh
      # or with a custom config name:
      bash examples/embodiment/collect_data.sh my_custom_config

4. Use the SpaceMouse (or GELLO) to operate the robot. Once ``num_data_episodes``
   successes are recorded the script saves the buffer and exits. Logs and data are
   written under ``logs/{timestamp}/``.

   To use GELLO instead of SpaceMouse, use the dedicated config:

   .. code-block:: bash

      bash examples/embodiment/collect_data.sh realworld_collect_data_gello

   See :doc:`../../examples/embodied/franka` for GELLO setup details.

----

Best Practices
--------------

**Episode collection (CollectEpisode)**

- Image data is large. If disk space is limited, use ``only_success=True`` to
  discard failed episodes.
- In distributed training, assign each worker a unique ``rank`` to prevent
  filename collisions.

**Real-robot replay buffer collection**

- Prioritise trajectory quality. If the success rate is low, relax
  ``success_hold_steps`` or set a more tolerant ``target_ee_pose``.
- After collection, load the buffer with ``TrajectoryReplayBuffer.load_checkpoint()``
  to verify the trajectory count before launching training.
- To append additional demonstrations, re-run the script pointing to the same
  ``demos`` directory. With ``auto_save=True``, the buffer writes incrementally
  without overwriting existing trajectories.

Visualization Tools
-------------------

After collection, you can inspect both output formats directly from the saved
artifacts under ``logs/{timestamp}/``.

**Replay buffer trajectories**

Use the existing replay-buffer visualizer to inspect trajectories in
``logs/{timestamp}/demos/``:

.. code-block:: bash

   python toolkits/replay_buffer/visualize.py \
       --replay_dir logs/{timestamp}/demos

**LeRobot datasets**

Use ``toolkits/lerobot/visualize_lerobot_dataset.py`` to expand a LeRobot
dataset into per-episode folders containing ``.jpg`` images and ``.txt`` step
metadata. The visualizer reads ``meta/info.json`` directly under the path
you pass, so point ``--dataset-path`` at one **concrete** LeRobot root
(``save_dir/rank_*/id_*``), not the parent ``collected_data/`` directory:

.. code-block:: bash

   python toolkits/lerobot/visualize_lerobot_dataset.py \
       --dataset-path logs/{timestamp}/collected_data/rank_0/id_0 \
       --output-dir logs/{timestamp}/collected_data_visualized/rank_0_id_0

To inspect every collected root, run the command once per
``rank_*/id_*`` directory (for example with a shell loop). The
replay-buffer converter, in contrast, accepts the parent
``collected_data/`` directory because it walks nested
``rank_*/id_*/data/**/*.parquet`` roots automatically.

The tool reads ``meta/info.json`` plus each ``episode_*.parquet`` file
within the supplied root, then creates output like
``episode_000000/step_000003_image.jpg`` and
``episode_000000/step_000003.txt`` for quick inspection.
