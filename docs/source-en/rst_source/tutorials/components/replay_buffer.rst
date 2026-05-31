Replay Buffer Tutorial
==============================

This tutorial focuses on **practical usage** and **configuration tips** for
`TrajectoryReplayBuffer`. For a fuller design overview and data-flow details,
see the API doc: :doc:`../../apis/replay_buffer`.

Quick Start
-----------

.. code-block:: python

   from rlinf.data.replay_buffer import TrajectoryReplayBuffer

   buffer = TrajectoryReplayBuffer(
       seed=1234,
       enable_cache=True,
       cache_size=5,
       sample_window_size=100,
       auto_save=True,
       auto_save_path="/path/to/buffer",
       trajectory_format="pt",
   )

Common Parameters
-----------------

- `auto_save_path`: trajectory storage directory when ``auto_save=True``.
  This argument is **required** when ``auto_save=True`` — the constructor
  asserts that it is non-empty
  (`rlinf/data/replay_buffer.py:270-279`). There is no fallback to the log
  directory.
- `trajectory_format`: `pt` (default) or `pkl`.
- `enable_cache` / `cache_size`: enable cache and set its size for throughput.
- `sample_window_size`: sample from the most recent N trajectories; 0 means all.
- `auto_save`: whether to persist new trajectories to disk **incrementally**
  inside ``add_trajectories``. When ``False``, the constructor force-enables
  the in-memory cache (sized to ``sample_window_size``) so trajectories are
  not lost. Explicit ``save_checkpoint()`` calls are **independent** of this
  flag and continue to work regardless of ``auto_save``.

Add Trajectories
----------------

.. code-block:: python

   # trajectories is List[Trajectory]
   buffer.add_trajectories(trajectories)

Key behavior during writes:

- generate `uuid` and `trajectory_id` for each trajectory
- update `_trajectory_index` and counters
- async save by background thread (when `auto_save=True`)

Sampling for Training
---------------------

.. code-block:: python

   batch = buffer.sample(num_chunks=256)
   # batch shape: [num_chunks, ...]

Sampling draws transitions within the window and returns a rollout-aligned batch dict.

Save and Load
-------------

.. code-block:: python

   buffer.save_checkpoint("/path/to/ckpt")

   buffer.load_checkpoint(
       load_path="/path/to/ckpt",
       is_distributed=True,
       local_rank=0,
       world_size=4,
   )

When saving a checkpoint, cached trajectories and metadata are saved into the checkpoint path.
Loading requires setting `load_path` to the checkpoint directory that contains both metadata
and trajectory files.
The trajectory data is saved in the format
``trajectory_{trajectory_id}_{model_weights_id}.{pt|pkl}``.

Convert Local LeRobot Datasets
------------------------------

``algorithm.demo_buffer.load_path`` expects a ``TrajectoryReplayBuffer``
checkpoint, not a raw LeRobot parquet directory. If you already have a local
LeRobot-format dataset root under ``data/**/*.parquet`` or a RLinf collector
parent such as ``collected_data/`` with nested ``rank_*/id_*/data/**/*.parquet``
roots, convert it first:

.. code-block:: bash

   python -m rlinf.data.lerobot_replay_buffer \
     --dataset-path /path/to/lerobot_dataset \
     --save-path /path/to/replay_buffer_demo \
     --num-action-chunks <actor.model.num_action_chunks>

.. warning::

   ``--num-action-chunks`` MUST match the training config's
   ``actor.model.num_action_chunks``. The persisted ``loss_mask`` trailing
   dim derives from this flag; a mismatch causes the strict
   ``concat_batch`` mode to reject demo + rollout mixing at training time.
   The flag defaults to ``1`` for backward compatibility, which only works
   for single-chunk SAC configs.

.. warning::

   **Image channel order: RGB only.** The converter's bytes / path / PIL
   code paths emit RGB via ``PIL.Image.convert("RGB")``. Raw numpy /
   torch tensor cells are accepted **as-is and assumed RGB**; the
   converter cannot detect BGR from a uint8 array. If your collector
   (e.g. OpenCV / RealWorld cameras that natively output BGR) writes raw
   arrays into LeRobot frames, convert BGR → RGB BEFORE passing data to
   ``CollectEpisode`` / the converter, or your demo buffer will silently
   train on color-swapped images.

The converter decodes image columns that contain arrays, PIL images, encoded
``bytes`` payloads, or paths relative to the local LeRobot dataset root. If an
image payload cannot be decoded, it fails fast by default. All-null optional
image columns are treated as absent; partially-null optional image columns fail
fast to keep camera schemas stable inside each episode. Use ``--state-only`` only
when you explicitly want a state/action-only demo buffer and the training config
uses a state-only actor model. Do not use ``--state-only`` for image-based
SAC/RLPD configs such as ``cnn_policy`` or image ``flow_policy`` configs:

.. code-block:: bash

   python -m rlinf.data.lerobot_replay_buffer \
     --dataset-path /path/to/collected_data \
     --save-path /path/to/replay_buffer_demo \
     --num-action-chunks <actor.model.num_action_chunks> \
     --state-only

Then replace ``load_path`` in an existing RLPD/SAC ``demo_buffer`` block:

.. code-block:: yaml

   algorithm:
     demo_buffer:
       enable_cache: True
       cache_size: 200
       min_buffer_size: 1
       sample_window_size: 200
       load_path: /path/to/replay_buffer_demo
       load_mode: shard  # or replicate for small demo sets on multiple actor ranks
       auto_save: False

RLinf LeRobot collection writes each action frame with explicit ``next_state`` /
``next_*`` observation fields, so terminal actions, rewards, ``terminated``,
``truncated``, and intervention flags stay aligned with the source action. Legacy
datasets without explicit next observations must include an observation-only
final frame; otherwise the converter fails fast instead of silently dropping the
last action.

**Terminal-flag contract.** The converter
(``_validate_terminal_frame_flags`` in ``rlinf/data/lerobot_replay_buffer.py``) enforces a strict
contract on terminal frames:

- When both ``terminated`` and ``truncated`` columns are present, the
  converter raises if they are **both true** in the same frame
  (gymnasium spec: terminal end vs. truncated horizon are mutually
  exclusive).
- ``done`` must agree with ``terminated or truncated`` whenever both
  ``done`` and the split flags are provided; mismatches raise instead
  of silently picking one source.
- On the final action frame, partial / incomplete terminal metadata
  (e.g. ``done`` missing while either of ``terminated`` / ``truncated``
  is present and inconclusive) is rejected so that the implicit
  ``default_done`` fallback can never override an explicit, ambiguous
  source. Set ``done``, or provide split flags that determine ``done``,
  to make the final transition unambiguous.

Multi-view columns such as ``wrist_image-0`` and ``extra_view_image-1`` are
stacked into canonical replay keys.

By default ``demo_buffer.load_mode: shard`` splits loaded demos across actor
ranks. If the converted demo buffer is smaller than the actor world size, use
``load_mode: replicate`` so every actor rank loads the full demo buffer.

.. warning::

   **Per-rank ``min_buffer_size`` caveat (shard mode).** Under
   ``load_mode: shard``, each actor rank receives roughly
   ``floor(N / world_size)`` trajectories
   (``TrajectoryReplayBuffer.load_checkpoint`` splits the trajectory
   list by ``local_rank`` / ``world_size``). The post-load validator
   (``validate_loaded_demo_buffer`` in
   ``rlinf/workers/actor/sac_demo_buffer_utils.py``) then hard-fails on
   any rank whose shard has fewer than
   ``algorithm.demo_buffer.min_buffer_size`` trajectories. Concretely,
   if you converted ``N`` demos and run with ``world_size`` actor ranks,
   keep ``min_buffer_size <= floor(N / world_size)``. If you cannot
   collect enough demos to satisfy that on every rank, either lower
   ``min_buffer_size`` or switch to ``load_mode: replicate`` so every
   rank sees the full ``N`` trajectories.

This is data plumbing for replay/demo-buffer initialization; it is not a full
HIL-SERL reproduction or a claim about training performance.

CLI Test
--------

.. code-block:: bash

   python rlinf/data/replay_buffer.py \
     --load-path /path/to/buffer \
     --num-chunks 1024 \
     --cache-size 10 \
     --enable-cache

This command loads a buffer checkpoint and samples once, printing batch keys and shapes.

Merge / Split Tool
------------------

Script path: `toolkits/replay_buffer/merge_or_split_replay_buffer.py`

.. code-block:: bash

   # Merge multiple ranks (interleaved by original trajectory_id)
   python toolkits/replay_buffer/merge_or_split_replay_buffer.py \
     --source-path /path/to/buffer \
     --save-path /path/to/merged \
     --copy

.. code-block:: bash

   # Split a single buffer by first N trajectories
   python toolkits/replay_buffer/merge_or_split_replay_buffer.py \
     --source-path /path/to/buffer \
     --save-path /path/to/split \
     --split-count 30 \
     --copy

Cleanup and Reset
-----------------

.. code-block:: python

   buffer.close()        # close async save thread
   buffer.clear()        # clear index and counters
   buffer.clear_cache()  # clear cache and close thread

Tips
----

- **Throughput**: enable cache and set `cache_size` to recent trajectories.
- **Data freshness**: use `sample_window_size` to limit the sampling window.

Visualization Tool
------------------

RLinf provides an interactive visualizer for inspecting trajectory data saved by the replay buffer.

Features
~~~~~~~~

- **Lazy loading**: Uses `TrajectoryReplayBuffer` to load trajectories on-demand, avoiding loading all data into memory
- **Auto-switching**: Automatically advances to the next trajectory when reaching the last frame
- **Jump to trajectory**: Type trajectory ID in the text box to jump directly
- **Multi-camera support**: View main, wrist, or extra camera views
- **Batch navigation**: Navigate between batch indices if B > 1
- **SSH/Headless support**: Save images for viewing in VSCode Remote SSH

Interactive Mode (Local Machine with Display)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python toolkits/replay_buffer/visualize.py \
       --replay_dir logs/my_run/replay_buffer/rank_0

Navigate with keyboard:

- ``←`` / ``→`` (or ``p`` / ``n``): Next/previous step (auto-switches trajectories at boundaries)
- ``↑`` / ``↓``: Next/previous trajectory
- ``b`` / ``v``: Switch between batch indices (if B > 1)
- ``s``: Save current view to image file
- ``Home`` / ``End``: Jump to first/last step
- ``q`` / ``Esc``: Quit
- Type trajectory ID in the text box to jump directly

SSH/Headless Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the headless interactive script:

.. code-block:: bash

   python toolkits/replay_buffer/visualize_headless.py \
       --replay_dir logs/my_run/replay_buffer/rank_0 \
       --output viz.png

Then in VSCode:

1. Open ``viz.png`` in the editor
2. Navigate using command-line prompts
3. The image updates automatically - VSCode will show the changes

**Commands:**

- ``n`` / ``next``: Next step (auto-switches to next trajectory at end)
- ``p`` / ``prev``: Previous step
- ``nt`` / ``nexttraj``: Next trajectory
- ``pt`` / ``prevtraj``: Previous trajectory
- ``j <id>``: Jump to trajectory ID (e.g., ``j 42``)
- ``info``: Show current position
- ``q`` / ``quit``: Exit

Auto-save with X11 Forwarding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have X11 forwarding enabled:

.. code-block:: bash

   python toolkits/replay_buffer/visualize.py \
       --replay_dir logs/my_run/replay_buffer/rank_0 \
       --save_image --output viz.png

Navigate with keyboard, and the image file updates automatically. Open ``viz.png`` in VSCode to see the current view.

Static Image Export
~~~~~~~~~~~~~~~~~~~

Save a single frame without interaction:

.. code-block:: bash

   python toolkits/replay_buffer/visualize.py \
       --replay_dir logs/my_run/replay_buffer/rank_0 \
       --save_image --output viz.png --no_display

Display Information
~~~~~~~~~~~~~~~~~~~

The visualizer shows:

- **Current observation** (left panel)
- **Next observation** (right panel)
- **Trajectory ID** and index position
- **Step** and **Batch** indices
- **Action**, **Reward**, and **Done** flag for each transition

View Different Camera Angles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Main camera (default)
   python toolkits/replay_buffer/visualize.py \
       --replay_dir logs/my_run/replay_buffer/rank_0 \
       --camera main_images

   # Wrist camera
   python toolkits/replay_buffer/visualize.py \
       --replay_dir logs/my_run/replay_buffer/rank_0 \
       --camera wrist_images

   # Extra view camera
   python toolkits/replay_buffer/visualize.py \
       --replay_dir logs/my_run/replay_buffer/rank_0 \
       --camera extra_view_images

Notes
~~~~~

- The tool uses ``TrajectoryReplayBuffer.load_checkpoint()`` to read metadata and index files.
- Trajectories are loaded lazily on-demand using the public
  ``load_trajectory(trajectory_id, model_weights_id)`` method; metadata is
  iterated via ``iter_trajectory_metadata()`` / ``get_trajectory_info(id)``,
  and the live id list comes from ``list_trajectory_ids()``. Do NOT
  reach into the private ``_trajectory_*`` attributes — they are not
  part of the public surface.
- Cache size is set to 5 trajectories to balance memory and performance.
- When you press ``→`` at the last frame of trajectory i, it automatically jumps to frame 0 of trajectory i+1.
- When you press ``←`` at the first frame of trajectory i, it automatically jumps to the last frame of trajectory i-1.
- Image files are saved at 150 DPI for good quality while keeping file size reasonable.

Public Replay-Buffer API
~~~~~~~~~~~~~~~~~~~~~~~~

External tools (visualizers, validators, ad-hoc scripts) should use the
following public methods on ``TrajectoryReplayBuffer``:

- ``list_trajectory_ids() -> list[int]`` — snapshot of trajectory ids in
  insertion order, taken under the index lock.
- ``iter_trajectory_metadata()`` — yields ``(trajectory_id,
  model_weights_id, num_samples)`` tuples without loading any payload.
- ``get_trajectory_info(trajectory_id) -> dict`` — returns the stored
  metadata dict (``model_weights_id``, ``num_samples``, ``shape``).
- ``load_trajectory(trajectory_id, model_weights_id) -> Trajectory`` —
  public wrapper around the on-disk load path.

Durability and resume safety
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- All persisted state (``metadata.json``, ``trajectory_index.json``,
  ``trajectory_*.pt``, ``.pkl``) is written via a temp file +
  ``os.fsync`` + ``os.replace`` so a Ctrl-C / SIGKILL / OOM mid-write
  cannot leave a half-written checkpoint at the final path.
- The persisted index references ONLY trajectories whose payload file is
  durable on disk. A concurrent ``add_trajectories`` whose ``.pt`` is
  still being written is excluded from the index until the save
  completes — resume will never load metadata that points at a missing
  file.
- ``close(wait=True)`` drains every pending save / metadata-flush future
  and re-raises the first error. CLIs / collectors MUST call
  ``buffer.close()`` (or wrap with ``try/finally``) so async persistence
  failures aren't silently swallowed.

LeRobot writer schema modes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``LeRobotDatasetWriter.create()`` exposes a ``transition_schema`` kwarg
that selects between two field sets:

- ``transition_schema=False`` (default — backward compatible): emits the
  minimal LeRobot schema (``state``, ``actions``, ``done``,
  ``is_success``, ``image``, ``intervene_flag``,
  ``observation.task_description``).
- ``transition_schema=True``: also emits ``next_state``, ``next_image``
  (and ``next_<wrist/extra_view>_image`` when configured), ``rewards``,
  ``terminated``, ``truncated``. This is the schema required by the
  RLPD/SAC demo-buffer pipeline; ``CollectEpisode`` opts in
  automatically.

If you build a custom writer for a non-RLinf consumer, leave
``transition_schema=False`` to preserve the legacy column set. Older
code that called ``LeRobotDatasetWriter.create(features=None)`` and
implicitly received the transition-rich schema MUST switch to passing
``transition_schema=True`` to keep that behavior.

SAC FSDP collective contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SAC actor worker (``EmbodiedSACFSDPPolicy``) gates several
collective-relevant decisions on a GLOBAL (all-reduced) view so every
rank takes the same branch and the FSDP process group does not deadlock:

- Replay-buffer / demo-buffer readiness is decided via global AND
  (``_global_all_ranks``). A single rank with a short shard skips the
  whole training step on every rank.
- ``clip_grad_norm_``, ``optimizer.step``, ``scheduler.step``, the
  alpha all-reduce, and the soft target-network update are gated on
  ``_global_valid_count``. Ranks with locally zero valid samples still
  enter ``backward()`` (via a graph-connected zero loss) so FSDP
  collectives remain consistent, but they do NOT advance the LR
  schedule when the global valid count is zero.
- The all-zero / partial mask paths sanitize NaN/Inf in
  ``per_sample_loss`` before reduction so a single rank's bad value
  cannot poison every rank via FSDP's grad averaging.
- The per-rank loss is scaled by ``world_size / global_valid_count`` so
  that FSDP's grad averaging recovers the global unweighted mean
  regardless of how unevenly valid samples are distributed.
