Replay Buffer 使用教程
==============================

本教程聚焦 `TrajectoryReplayBuffer` 的 **实际使用** 与 **配置建议**。
更完整的设计说明与数据流细节见 API 文档：:doc:`../../apis/replay_buffer`。

快速开始
--------

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

常用参数
--------

- `enable_cache` / `cache_size`：启用并控制缓存数量，用于提升采样吞吐。
- `sample_window_size`：仅在最近 N 条轨迹内采样；0 表示全量。
- `auto_save`：是否在 ``add_trajectories`` 中 **增量** 落盘。设为
  ``False`` 时构造器会强制启用内存缓存（``cache_size`` 与
  ``sample_window_size`` 相等），以避免轨迹丢失。显式调用
  ``save_checkpoint()`` 与该参数 **无关**，无论 ``auto_save``
  设置为何都会执行。
- `auto_save_path`：``auto_save=True`` 时的轨迹存储目录，
  **必填**。构造器会断言该参数非空
  (`rlinf/data/replay_buffer.py:270-279`)，
  不存在回退到 log 目录的默认行为。
- `trajectory_format`：`pt`（默认）或 `pkl`。

写入轨迹
--------

.. code-block:: python

   # trajectories 为 List[Trajectory]
   buffer.add_trajectories(trajectories)

写入阶段的关键行为：

- 为每条轨迹生成 `uuid` 与 `trajectory_id`
- 更新 `_trajectory_index` 与计数器
- 在后台线程异步保存轨迹文件（若 `auto_save=True`）

采样训练
--------

.. code-block:: python

   batch = buffer.sample(num_chunks=256)
   # batch 形状: [num_chunks, ...]

采样在滑动窗口内随机抽取 transition，并返回与 rollout 对齐的 batch 字典。

保存与加载
----------

.. code-block:: python

   buffer.save_checkpoint("/path/to/ckpt")

   buffer.load_checkpoint(
       load_path="/path/to/ckpt",
       is_distributed=True,
       local_rank=0,
       world_size=4,
   )

保存 checkpoint 时会把缓存轨迹与 metadata 一并写入 checkpoint 路径。
加载时需要设置 `load_path` 指向包含 metadata 和轨迹文件的 checkpoint 目录。
轨迹数据保存格式为
``trajectory_{trajectory_id}_{model_weights_id}.{pt|pkl}``。

转换本地 LeRobot 数据集
---------------------------

``algorithm.demo_buffer.load_path`` 需要指向 ``TrajectoryReplayBuffer``
checkpoint，而不是原始 LeRobot parquet 目录。若已有本地 LeRobot 格式数据集
root（``data/**/*.parquet``），或 RLinf 采集器生成的 ``collected_data/`` 父目录
（其下包含 ``rank_*/id_*/data/**/*.parquet`` root），先转换：

.. code-block:: bash

   python -m rlinf.data.lerobot_replay_buffer \
     --dataset-path /path/to/lerobot_dataset \
     --save-path /path/to/replay_buffer_demo \
     --num-action-chunks <actor.model.num_action_chunks>

.. warning::

   ``--num-action-chunks`` 必须与训练配置的 ``actor.model.num_action_chunks``
   一致。持久化的 ``loss_mask`` 末维由该 flag 决定；若不一致，严格模式的
   ``concat_batch`` 会在训练时拒绝 demo + rollout 混合。该 flag 默认 ``1``
   仅适配单 chunk SAC 配置，多 chunk 必须显式传入。

.. warning::

   **图像通道顺序：仅支持 RGB**。转换器的 bytes / path / PIL 路径会通过
   ``PIL.Image.convert("RGB")`` 输出 RGB；原始 numpy / torch 张量
   **按原样接收并默认 RGB**，无法从 uint8 数组推断 BGR。
   如果采集端（如 OpenCV / 默认输出 BGR 的 RealWorld 相机）将
   原始数组写入 LeRobot frame，请在传给 ``CollectEpisode`` /
   转换器之前自行 BGR → RGB；否则 demo buffer 会在不报错的情况下
   用颜色错位的图像训练。

转换器会解码 image 列中的数组、PIL image、编码后的 ``bytes`` payload，或相对
本地 LeRobot dataset root 的路径。若 image payload 无法解码，转换器默认会
fail fast。全为空值的可选图像列会视为不存在；同一 episode 内部分为空、
部分有值的可选图像列会 fail fast，确保相机 schema 稳定。仅在明确需要
state/action-only demo buffer，且训练配置使用 state-only actor model 时，
才使用 ``--state-only``。不要把 ``--state-only`` 产物用于 ``cnn_policy``
或 image ``flow_policy`` 等图像 SAC/RLPD 配置：

.. code-block:: bash

   python -m rlinf.data.lerobot_replay_buffer \
     --dataset-path /path/to/collected_data \
     --save-path /path/to/replay_buffer_demo \
     --num-action-chunks <actor.model.num_action_chunks> \
     --state-only

然后在已有 RLPD/SAC ``demo_buffer`` block 中替换 ``load_path``：

.. code-block:: yaml

   algorithm:
     demo_buffer:
       enable_cache: True
       cache_size: 200
       min_buffer_size: 1
       sample_window_size: 200
       load_path: /path/to/replay_buffer_demo
       load_mode: shard  # demo 数量少于 actor rank 时可改为 replicate
       auto_save: False

RLinf LeRobot 采集会在每个 action frame 中写入显式 ``next_state`` / ``next_*``
observation 字段，因此 terminal action、reward、``terminated``、``truncated`` 和
intervention flag 都会与 source action 对齐。旧数据若没有显式 next observation，
必须包含 observation-only final frame；否则转换器会 fail fast，而不是静默丢弃
最后一个 action。

**Terminal flag 校验合约。** 转换器
（``_validate_terminal_frame_flags``，位于 ``rlinf/data/lerobot_replay_buffer.py``）会在 terminal frame
上施加更严格的检查：

- 同时提供 ``terminated`` 与 ``truncated`` 列时，若同一帧 **两者都为
  True**，转换器会直接抛错（gymnasium 规范：terminal 与 truncation
  互斥）。
- 若 ``done`` 与拆分的 ``terminated`` / ``truncated`` 都存在，
  ``done`` 必须等于 ``terminated or truncated``，不一致时直接抛错而非
  静默选择其中一边。
- 最后一个 action frame 上若 terminal metadata **不完整**\ （例如
  ``done`` 缺失但其中一个拆分 flag 存在且无法决定）会被拒绝，避免
  ``default_done`` 隐式回退覆盖一个明确但不一致的 source。请显式设置
  ``done`` 或提供可决定 ``done`` 的拆分 flag。

``wrist_image-0``、``extra_view_image-1`` 等多视角列会被堆叠成
canonical replay key。

``demo_buffer.load_mode`` 默认是 ``shard``，会按 actor rank 切分已加载 demo。
如果转换后的 demo buffer 数量少于 actor world size，可设为 ``replicate``，让每个
actor rank 都加载完整 demo buffer。

.. warning::

   **shard 模式下的逐 rank ``min_buffer_size`` 注意事项。**
   ``load_mode: shard`` 模式下，每个 actor rank 大约只拿到
   ``floor(N / world_size)`` 条 trajectory
   （``TrajectoryReplayBuffer.load_checkpoint`` 按 ``local_rank`` /
   ``world_size`` 切分 trajectory 列表）。
   随后 ``validate_loaded_demo_buffer``
   （``rlinf/workers/actor/sac_demo_buffer_utils.py``）会对所有 rank
   逐个检查，任何一个 rank 的 shard 数量小于
   ``algorithm.demo_buffer.min_buffer_size`` 都会直接抛错。具体来说：
   若你转换得到 ``N`` 条 demo 并以 ``world_size`` 个 actor rank 启动，
   ``min_buffer_size`` 必须 ``<= floor(N / world_size)``。如果无法在每个
   rank 上满足，请降低 ``min_buffer_size`` 或改用 ``load_mode:
   replicate``，让每个 rank 拿到完整的 ``N`` 条 trajectory。

该功能只是 replay/demo-buffer 初始化的数据转换路径，不代表完整 HIL-SERL
复现，也不声明训练效果提升。

命令行测试
--------------

.. code-block:: bash

   python rlinf/data/replay_buffer.py \
     --load-path /path/to/buffer \
     --num-chunks 1024 \
     --cache-size 10 \
     --enable-cache

该命令会加载 buffer checkpoint 并进行一次采样，输出 batch 的 key 与 shape。

合并 / 拆分工具
-----------------

脚本位置：`toolkits/replay_buffer/merge_or_split_replay_buffer.py`

.. code-block:: bash

   # 合并多个 rank（按原 trajectory_id 交错）
   python toolkits/replay_buffer/merge_or_split_replay_buffer.py \
     --source-path /path/to/buffer \
     --save-path /path/to/merged \
     --copy

.. code-block:: bash

   # 拆分单个 buffer，取前 N 条轨迹
   python toolkits/replay_buffer/merge_or_split_replay_buffer.py \
     --source-path /path/to/buffer \
     --save-path /path/to/split \
     --split-count 30 \
     --copy

资源释放与重置
--------------

.. code-block:: python

   buffer.close()        # 关闭异步保存线程
   buffer.clear()        # 清空索引与计数
   buffer.clear_cache()  # 清空缓存并关闭线程

实践建议
--------

- **吞吐优先**：开启 `enable_cache`，`cache_size` 设为近期活跃轨迹数。
- **数据新鲜度**：使用 `sample_window_size` 限制采样窗口。

可视化工具
----------

RLinf 提供了交互式可视化工具，用于检查 replay buffer 保存的轨迹数据。

功能特性
~~~~~~~~

- **延迟加载**：使用 `TrajectoryReplayBuffer` 按需加载轨迹，避免将所有数据加载到内存
- **自动切换**：到达最后一帧时自动前进到下一条轨迹
- **跳转轨迹**：在文本框中输入轨迹 ID 直接跳转
- **多相机支持**：查看主相机、腕部相机或额外视角相机
- **批次导航**：在 B > 1 时可在批次索引间导航
- **SSH/无显示器模式支持**：保存图像以便查看

交互模式（本地机器有显示）
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python toolkits/replay_buffer/visualize.py \
       --replay_dir logs/my_run/replay_buffer/rank_0

键盘导航：

- ``←`` / ``→`` (或 ``p`` / ``n``)：上一步/下一步（在边界自动切换轨迹）
- ``↑`` / ``↓``：上一条/下一条轨迹
- ``b`` / ``v``：在批次索引间切换（如果 B > 1）
- ``s``：保存当前视图到图像文件
- ``Home`` / ``End``：跳转到第一步/最后一步
- ``q`` / ``Esc``：退出
- 在文本框中输入轨迹 ID 直接跳转

SSH/无显示器模式
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

使用无显示器交互脚本：

.. code-block:: bash

   python toolkits/replay_buffer/visualize_headless.py \
       --replay_dir logs/my_run/replay_buffer/rank_0 \
       --output viz.png

然后在 VSCode 中：

1. 在编辑器中打开 ``viz.png``
2. 使用命令行提示进行导航
3. 图像自动更新 - VSCode 会显示变化

**命令：**

- ``n`` / ``next``：下一步（在末尾自动切换到下一条轨迹）
- ``p`` / ``prev``：上一步
- ``nt`` / ``nexttraj``：下一条轨迹
- ``pt`` / ``prevtraj``：上一条轨迹
- ``j <id>``：跳转到轨迹 ID（例如 ``j 42``）
- ``info``：显示当前位置
- ``q`` / ``quit``：退出

带 X11 转发的自动保存
~~~~~~~~~~~~~~~~~~~~~

如果启用了 X11 转发：

.. code-block:: bash

   python toolkits/replay_buffer/visualize.py \
       --replay_dir logs/my_run/replay_buffer/rank_0 \
       --save_image --output viz.png

使用键盘导航，图像文件会自动更新。在 VSCode 中打开 ``viz.png`` 查看当前视图。

静态图像导出
~~~~~~~~~~~~

保存单帧而不进行交互：

.. code-block:: bash

   python toolkits/replay_buffer/visualize.py \
       --replay_dir logs/my_run/replay_buffer/rank_0 \
       --save_image --output viz.png --no_display

显示信息
~~~~~~~~

可视化工具显示：

- **当前观察** （左面板）
- **下一个观察** （右面板）
- **轨迹 ID** 和索引位置
- **步骤** 和 **批次** 索引
- 每个转换的 **动作**、 **奖励** 和 **完成** 标志

查看不同相机角度
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # 主相机（默认）
   python toolkits/replay_buffer/visualize.py \
       --replay_dir logs/my_run/replay_buffer/rank_0 \
       --camera main_images

   # 腕部相机
   python toolkits/replay_buffer/visualize.py \
       --replay_dir logs/my_run/replay_buffer/rank_0 \
       --camera wrist_images

   # 额外视角相机
   python toolkits/replay_buffer/visualize.py \
       --replay_dir logs/my_run/replay_buffer/rank_0 \
       --camera extra_view_images

注意事项
~~~~~~~~

- 工具使用 ``TrajectoryReplayBuffer.load_checkpoint()`` 读取元数据和索引文件。
- 轨迹通过公开方法 ``load_trajectory(trajectory_id, model_weights_id)``
  按需延迟加载；元数据通过 ``iter_trajectory_metadata()`` /
  ``get_trajectory_info(id)`` 访问，轨迹 id 列表通过
  ``list_trajectory_ids()`` 获取。**请勿访问私有 ``_trajectory_*``
  字段**，它们不属于公开接口。
- 缓存大小设置为 5 条轨迹以平衡内存和性能。
- 当在轨迹 i 的最后一帧按 ``→`` 时，会自动跳转到轨迹 i+1 的第 0 帧。
- 当在轨迹 i 的第一帧按 ``←`` 时，会自动跳转到轨迹 i-1 的最后一帧。
- 图像文件以 150 DPI 保存，在保持良好质量的同时控制文件大小。

公开 Replay-Buffer API
~~~~~~~~~~~~~~~~~~~~~~

外部工具（可视化器、validator、临时脚本）应使用
``TrajectoryReplayBuffer`` 的下列公开方法：

- ``list_trajectory_ids() -> list[int]``：按插入顺序返回轨迹 id 列表
  快照，在 index 锁内取出。
- ``iter_trajectory_metadata()``：迭代 ``(trajectory_id,
  model_weights_id, num_samples)`` 元组，不加载任何 payload。
- ``get_trajectory_info(trajectory_id) -> dict``：返回该轨迹存储的
  元数据 dict（``model_weights_id``、``num_samples``、``shape``）。
- ``load_trajectory(trajectory_id, model_weights_id) -> Trajectory``：
  对磁盘加载路径的公开封装。

持久化与恢复安全
~~~~~~~~~~~~~~~~

- 所有持久化状态（``metadata.json``、``trajectory_index.json``、
  ``trajectory_*.pt``、``.pkl``）都通过临时文件 +
  ``os.fsync`` + ``os.replace`` 原子写入：Ctrl-C / SIGKILL / OOM
  中断不会在最终路径留下半写文件。
- 持久化 index 只引用 payload 已落盘的轨迹。并发的
  ``add_trajectories``，其 ``.pt`` 还在写入时不会进入 index；
  resume 不会加载指向缺失文件的元数据。
- ``close(wait=True)`` 会 drain 所有待写 future 并重新抛出第一个
  异常。CLI / 采集器必须调用 ``buffer.close()``（或用
  ``try/finally``），否则异步持久化失败会被默默吞掉。

LeRobot writer schema 模式
~~~~~~~~~~~~~~~~~~~~~~~~~~

``LeRobotDatasetWriter.create()`` 通过 ``transition_schema`` kwarg
选择字段集合：

- ``transition_schema=False``（默认，向后兼容）：使用最小 LeRobot
  schema（``state``、``actions``、``done``、``is_success``、
  ``image``、``intervene_flag``、``observation.task_description``）。
- ``transition_schema=True``：额外写入 ``next_state``、``next_image``
  （以及配置了 wrist/extra_view 时的 ``next_<wrist/extra_view>_image``）、
  ``rewards``、``terminated``、``truncated``。这是 RLPD / SAC
  demo-buffer 流程所需的 schema；``CollectEpisode`` 会自动启用。

若你为非 RLinf 消费方编写自定义 writer，请保留
``transition_schema=False`` 维持遗留列集；旧代码若调用过
``LeRobotDatasetWriter.create(features=None)`` 并依赖隐式的
transition-rich schema，请显式传 ``transition_schema=True`` 以保留
原行为。

SAC FSDP 集体契约
~~~~~~~~~~~~~~~~~

SAC actor worker（``EmbodiedSACFSDPPolicy``）将若干涉及集体操作的
分支基于 GLOBAL（all-reduce 后）视图门控，确保所有秩走同一分支、
FSDP 进程组不会死锁：

- Replay / demo buffer 就绪通过 global AND（``_global_all_ranks``）
  决策。任何一秩 shard 偏短，整个训练步在每个秩上都跳过。
- ``clip_grad_norm_``、``optimizer.step``、``scheduler.step``、
  alpha 的 all_reduce，以及目标网络 soft update 都由
  ``_global_valid_count`` 门控。本地有效样本为 0 的秩仍然通过
  graph-connected 零损失调用 ``backward()``，保持 FSDP 集体一致；
  但 LR scheduler 不会在全局有效样本为 0 时推进。
- All-zero / partial mask 路径在 reduction 前对 ``per_sample_loss``
  执行 NaN/Inf sanitize，使单秩坏值无法通过 FSDP 梯度平均污染
  其他秩。
- 每秩损失按 ``world_size / global_valid_count`` 缩放，FSDP 梯度
  平均后恢复全局未加权均值，避免有效样本分布不均带来的偏差。
