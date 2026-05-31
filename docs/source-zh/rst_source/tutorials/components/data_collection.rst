数据采集
========

RLinf 提供两种数据采集方案，面向不同的下游用途：

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - 方式
     - 入口
     - 典型用途
   * - **Episode 采集**
     - ``CollectEpisode`` wrapper
     - 奖励模型 / 价值模型训练数据
   * - **真机 Replay Buffer 采集**
     - ``collect_data.sh``
     - 真机 RLPD 先验数据 / 策略初始化

----

Episode 数据采集
----------------

``CollectEpisode`` 是一个 ``gymnasium.Wrapper``，可以透明地包裹任意环境，
在 RL 训练或评估过程中自动逐 step 记录数据，并在 episode 结束时异步保存到磁盘。

支持保存为两种格式：

- **pickle** — 保存完整原始 buffer，适合自定义离线处理。
- **lerobot** — 保存结构化 Parquet + 元数据，可直接接入 LeRobot 训练流程。

核心特性
~~~~~~~~

- 支持单环境与向量化并行环境（``num_envs > 1``）。
- 兼容自动重置（auto-reset）环境：正确将重置前的最终观测归入当前 episode，
  将重置后的初始观测带入下一 episode。
- 写入操作在独立后台线程异步执行，不阻塞 RL 训练主循环。
- LeRobot writer 在第一条 episode 写入时懒初始化，自动推断图像尺寸、状态维度、动作维度。
- LeRobot 导出支持保存 ``image``、``wrist_image`` 与 ``extra_view_image``；
  当 ``wrist_images`` / ``extra_view_images`` 为 ``[N, H, W, C]`` 多视角堆叠时，
  会自动按索引展开成 ``wrist_image-0``、``extra_view_image-0`` …… 等列。
- ``only_success=True`` 可过滤失败 episode，节省磁盘空间。

.. warning::

   **图像通道顺序：仅支持 RGB**。``CollectEpisode`` 把环境给的原始图像
   数组直接写入 LeRobot frame；后续 LeRobot → replay-buffer 转换器对
   这些数组 **按原样接收并默认 RGB**。如果你的环境暴露的是 BGR
   （OpenCV / 默认输出 BGR 的 RealWorld 相机），必须在 obs 进入 env
   之前自行 BGR → RGB，否则持久化的 demo buffer 将以颜色错位的图像
   训练。运行时无法检测通道顺序，这是契约。

构造参数
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - 参数
     - 类型
     - 默认值
     - 说明
   * - ``env``
     - ``gym.Env``
     - —
     - 被包裹的 gymnasium 环境
   * - ``save_dir``
     - ``str``
     - —
     - episode 数据保存目录（自动创建）
   * - ``rank``
     - ``int``
     - ``0``
     - 分布式场景下的 Worker 编号，用于文件命名去重
   * - ``num_envs``
     - ``int``
     - ``1``
     - 并行环境数量
   * - ``show_goal_site``
     - ``bool``
     - ``True``
     - 是否在渲染中显示目标位置可视化（对支持的环境有效）
   * - ``export_format``
     - ``str``
     - ``"pickle"``
     - 保存格式：``"pickle"`` 或 ``"lerobot"``
   * - ``robot_type``
     - ``str``
     - ``"panda"``
     - 机器人类型，写入 LeRobot 元数据（仅 lerobot 格式有效）
   * - ``fps``
     - ``int``
     - ``10``
     - 数据集帧率，写入 LeRobot 元数据（仅 lerobot 格式有效）
   * - ``only_success``
     - ``bool``
     - ``False``
     - 仅保存成功的 episode
   * - ``finalize_interval``
     - ``int``
     - ``100``
     - 每写完 N 个 episode 主动调用 ``writer.finalize()`` 生成检查点（``0`` 表示禁用，仅 lerobot 格式有效）

使用示例
~~~~~~~~

**直接使用 wrapper：**

.. code-block:: python

   from rlinf.envs.wrappers.collect_episode import CollectEpisode

   env = CollectEpisode(
       env=base_env,
       save_dir="./collected_data",
       num_envs=8,
       export_format="lerobot",   # 或 "pickle"
       robot_type="panda",
       fps=10,
       only_success=True,
   )

   obs, info = env.reset()
   while not done:
       action = policy(obs)
       obs, reward, terminated, truncated, info = env.step(action)
   env.close()   # 触发最终写入并 finalize

**通过 YAML 配置启用（仿真训练场景）：**

在 YAML 配置文件的 ``env`` 部分添加 ``data_collection`` 配置：

.. code-block:: yaml

  env:
    group_name: "EnvGroup"
    enable_offload: False
    eval:
      data_collection:
        enabled: True
        save_dir: ${runner.logger.log_path}/collected_data
        export_format: "lerobot"      # 或 "pickle"
        only_success: True
        robot_type: "panda"
        fps: 10

然后正常启动训练脚本，数据会在训练过程中自动采集：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp_collect

数据格式详解
~~~~~~~~~~~~

**pickle 格式**

每个 episode 保存为一个独立的 ``.pkl`` 文件，命名规则：

.. code-block:: text

   rank_{rank}_env_{env_idx}_episode_{episode_id}_{success|fail}.pkl

示例：``rank_0_env_3_episode_42_success.pkl``

文件内容为一个字典：

.. code-block:: python

   {
       "rank":        int,   # Worker 编号
       "env_idx":     int,   # 环境索引
       "episode_id":  int,   # Episode 编号（单环境内自增）
       "success":     bool,  # 是否成功
       "observations": list, # 观测列表，长度 = num_steps + 1（含初始观测）
       "actions":     list,  # 动作列表，长度 = num_steps
       "rewards":     list,  # 奖励列表，长度 = num_steps
       "terminated":  list,  # 终止标志，长度 = num_steps
       "truncated":   list,  # 截断标志，长度 = num_steps
       "infos":       list,  # info 字典列表，长度 = num_steps
   }

.. note::

   pickle 格式完整保留原始 buffer 数据，适合需要自定义处理的场景（如离线 RL、行为分析）。
   观测列表中第 0 项来自 ``reset()``，第 1 至 N 项来自各 ``step()``。

**LeRobot 格式**

数据以 Parquet 文件存储，并附带 JSON 元数据。``CollectEpisode`` wrapper
会按 worker 和写入批次在
``save_dir/rank_{rank}/id_{episodes_written}/`` 下分别创建独立的
LeRobot dataset root：

.. code-block:: text

   save_dir/
   └── rank_0/                   # 每个 worker rank 一个子目录
       ├── id_0/                 # 每个写入批次一个子目录（受 finalize_interval 控制）
       │   ├── meta/
       │   │   ├── info.json           # 数据集元信息（fps、robot_type、维度等）
       │   │   ├── episodes.jsonl      # 每条 episode 的长度与任务描述
       │   │   ├── tasks.jsonl         # 去重后的任务列表
       │   │   └── stats.json          # 观测、动作的均值/方差统计
       │   └── data/
       │       └── chunk-000/
       │           ├── episode_000000.parquet
       │           ├── episode_000001.parquet
       │           └── ...
       └── id_1/                 # 下一批，触发 finalize_interval 后产生
           └── ...

.. note::

   ``rank_*`` 用于区分分布式 / 向量化 rollout 中的不同 writer；
   ``id_*`` 在每次调用 ``writer.finalize()``（例如每 ``finalize_interval``
   个 episode）时滚动，因此同一 rank 可能写出多个 LeRobot root。

每个 Parquet 文件的列结构：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 列名
     - 说明
   * - ``image``
     - 主摄像头图像（bytes + path），uint8
   * - ``next_image``
     - 用于 replay-buffer 转换的下一步主摄像头图像，uint8
   * - ``wrist_image`` / ``wrist_image-N``
     - 腕部摄像头图像（bytes + path），uint8；多视角堆叠时展开为
       ``wrist_image-0``、``wrist_image-1`` ……
   * - ``next_wrist_image`` / ``next_wrist_image-N``
     - 下一步腕部摄像头图像，使用相同的多视角展开规则
   * - ``extra_view_image`` / ``extra_view_image-N``
     - 额外视角图像（bytes + path），uint8；多视角堆叠时展开为
       ``extra_view_image-0``、``extra_view_image-1`` …… 无额外视角时列为空
   * - ``next_extra_view_image`` / ``next_extra_view_image-N``
     - 下一步额外视角图像，使用相同的多视角展开规则
   * - ``state``
     - 机器人状态向量，``float32[state_dim]``
   * - ``next_state``
     - 下一步机器人状态向量，``float32[state_dim]``
   * - ``actions``
     - 动作向量，``float32[action_dim]``
   * - ``rewards``
     - 与 ``actions`` 对齐的 transition reward，``float32[1]``
   * - ``timestamp``
     - 帧时间戳（秒），``float``
   * - ``frame_index``
     - episode 内帧序号，``int64``
   * - ``episode_index``
     - 全局 episode 编号，``int64``
   * - ``index``
     - 全局帧序号，``int64``
   * - ``task_index``
     - 任务编号（对应 tasks.jsonl），``int64``
   * - ``done``
     - 每步的结束标志，``bool``（episode 最后一步为 ``True``）
   * - ``terminated`` / ``truncated``
     - 与 ``actions`` 对齐的 Gymnasium 终止和截断标志
   * - ``is_success``
     - 该 episode 是否成功，``bool``

观测键的映射规则：wrapper 自动从 obs 字典中按以下优先级查找图像和状态：

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 字段
     - 查找键（按优先级）
   * - 主图像
     - ``main_images`` → ``image`` → ``full_image``
   * - 腕部图像
     - ``wrist_images`` → ``wrist_image``（``[N, H, W, C]`` 会按索引展开为
       ``wrist_image-0``、``wrist_image-1`` ……）
   * - 额外视角图像
     - ``extra_view_images`` → ``extra_view_image``（``[N, H, W, C]`` 会按索引
       展开为 ``extra_view_image-0``、``extra_view_image-1`` ……）
   * - 状态
     - ``states`` → ``state``

图像会自动转换为 uint8（浮点 [0, 1] 乘以 255，或直接截断转换）。

全为空值的可选图像列会在 replay-buffer 转换时视为不存在；同一 episode
内部分为空、部分有值的可选图像列会 fail fast，避免相机 schema 静默变化。

成功状态判断逻辑
~~~~~~~~~~~~~~~~

wrapper 从 info 字典中按以下优先级推断 episode 是否成功（从最后一步向前扫描，先找到则返回）：

对每步 info，依次检查以下三个来源（``final_info`` → ``episode`` → info 根层），
每个来源中按 ``success_once`` → ``success_at_end`` → ``success`` 顺序查找：

1. ``info["final_info"]["success_once"]`` / ``success_at_end`` / ``success``
2. ``info["episode"]["success_once"]`` / ``success_at_end`` / ``success``
3. ``info["success_once"]`` / ``info["success_at_end"]`` / ``info["success"]``

以上均未找到时，回退到逐步维护的 ``_episode_success`` 标志。

----

真机 Replay Buffer 数据采集
----------------------------

真机采集用于 RLPD（Reinforcement Learning from Prior Data）或策略初始化。
操作员通过 SpaceMouse 或 GELLO 等人工干预设备完成任务，数据以 ``Trajectory``
格式保存，可直接供后续真机训练使用。

与仿真的大规模并行采集不同，真机采集在单控制节点运行，按目标成功次数自动停止。

核心组件
~~~~~~~~

- **入口脚本**：``examples/embodiment/collect_data.sh``
- **采集逻辑**：``examples/embodiment/collect_real_data.py``（``DataCollector`` 类）
- **配置文件**：``examples/embodiment/config/realworld_collect_data.yaml``

``DataCollector`` 的工作流程：

1. 初始化 ``RealWorldEnv`` 和 ``TrajectoryReplayBuffer``。
2. 循环执行 step，从 ``info["intervene_action"]`` 读取 SpaceMouse 干预动作。
3. 构造 ``ChunkStepResult``，追加到 ``EmbodiedRolloutResult``。
4. episode 结束（``done=True``）且奖励 ``>= 0.5`` 时，记为一次成功，将轨迹写入 buffer。
5. 成功次数达到 ``num_data_episodes`` 后自动停止并 finalize buffer。

核心配置参数
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - 参数
     - 默认值
     - 说明
   * - ``runner.num_data_episodes``
     - ``20``
     - 目标成功轨迹数量，达到后自动停止
   * - ``cluster.node_groups[0].hardware.configs[0].robot_ip``
     - —
     - Franka 机器人 IP 地址。``node_groups`` 与 ``configs`` 都是 YAML 列表，
       结构详见下文 :ref:`zh-realworld-collect-yaml-shape`。
   * - ``env.eval.use_spacemouse``
     - ``True``
     - 是否启用 SpaceMouse 干预
   * - ``env.eval.no_gripper``
     - ``True``\ （key 缺省时 wrapper 默认值；
       shipped 配置未显式设置）
     - 是否使用不带夹爪维度的 6 维真机动作。单臂 wrapper 在该项为 True 时
       会套用 ``GripperCloseEnv``。
   * - ``env.eval.use_gello``
     - ``False``
     - 是否启用 GELLO 遥操作（与 SpaceMouse 互斥）
   * - ``env.eval.gello_port``
     - —
     - GELLO 设备串口路径（``use_gello`` 为 ``True`` 时必填）
   * - ``env.eval.override_cfg.target_ee_pose``
     - —
     - 任务目标末端位姿 ``[x, y, z, rx, ry, rz]``
   * - ``env.eval.override_cfg.success_hold_steps``
     - ``1``
     - 持续到达目标位姿多少 step 后判定为成功
   * - ``runner.record_task_description``
     - ``False``\ （与 shipped ``realworld_collect_data.yaml`` 一致）
     - 是否将任务描述写入观测

数据格式
~~~~~~~~

采集完成后，数据保存在：

.. code-block:: text

   logs/{timestamp}/demos/

``TrajectoryReplayBuffer`` 使用 ``.pt`` 格式存储每条轨迹。
每个 ``.pt`` 文件保存的是 ``Trajectory`` dataclass
(:class:`rlinf.data.embodied_io_struct.Trajectory`)，序列化的字段都在
**顶层**，没有 ``"transitions"`` 包裹层：

.. code-block:: python

   # rlinf.data.embodied_io_struct.Trajectory
   Trajectory(
       max_episode_length=int,             # 标量
       model_weights_id=str,               # 由 model 版本派生的 uuid
       actions=torch.Tensor,               # [T, B, action_dim]（也可能含 chunk 维度）
       intervene_flags=torch.Tensor,       # [T, B, ...] bool，逐步干预标志
       rewards=torch.Tensor,               # [T, B, 1]
       terminations=torch.Tensor,          # [T, B, 1] bool
       truncations=torch.Tensor,           # [T, B, 1] bool
       dones=torch.Tensor,                 # [T, B, 1] bool
       loss_mask=Optional[torch.Tensor],   # 设置时形状为 [T, B, num_action_chunks]
       forward_inputs=dict,                # 与生产端相关的 forward kwargs
       curr_obs={                          # 第 t 步的 observation dict
           "states":      ...,             # 机器人状态，例如 [T, B, 19]
           "main_images": ...,              # 主摄像头，例如 [T, B, H, W, C] uint8
           # 其它 observation key（wrist_images、extra_view_images ……）
       },
       next_obs={                          # 第 t+1 步的 observation dict
           "states":      ...,
           "main_images": ...,
       },
   )

.. note::

   ``intervene_flags`` 在 rollout 与持久化 demo trajectory 中含义不同：

   - rollout 阶段（``EmbodiedRolloutResult``）字段为
     **逐步布尔标志**\ ，来源于 ``info["intervene_flag"]``
     （例如 SpaceMouse / GELLO 干预路径）。
   - 在 ``examples/embodiment/collect_real_data.py`` 将成功 trajectory
     写入 replay buffer 时，会用 ``torch.ones_like(...)`` 把每一步全部
     **覆盖为 True**\ ，使该 trajectory 在 RLPD / SAC-demo 训练中被
     视为完全专家数据。这是按算子语义设计的：成功的真机 demo 无论
     哪些 substep 被操作员介入，都视为专家 teleop。

   若你的训练需要原始逐步标志，请从 ``collect_real_data`` 后处理之前
   的 rollout-side ``Trajectory`` 读取，而非持久化的 demo buffer。

同时采集 Replay Buffer 与 LeRobot 数据
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``examples/embodiment/collect_real_data.py`` 现在支持在同一次真机采集中，
同时写出 replay buffer 和 ``CollectEpisode`` 导出的 episode 数据。只要启用
``env.eval.data_collection.enabled=True``，成功轨迹就会同时保存到：

- ``logs/{timestamp}/demos/``：``TrajectoryReplayBuffer`` 轨迹，用于 RLPD
- ``logs/{timestamp}/collected_data/``：``pickle`` 或 LeRobot 格式的 episode 数据

若希望在收集真机 replay buffer 的同时额外导出 LeRobot 数据集，可保留如下配置：

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

使用步骤
~~~~~~~~

1. 在控制节点激活环境：

   .. code-block:: bash

      source <path_to_your_venv>/bin/activate

2. 编辑配置文件 ``examples/embodiment/config/realworld_collect_data.yaml``，
   将 ``ROBOT_IP`` 和 ``TARGET_EE_POSE`` 替换为真实机器人 IP 与目标位姿。

   .. _zh-realworld-collect-yaml-shape:

   ``cluster.node_groups`` 是 YAML 列表，每个条目的 ``hardware.configs``
   同样是列表（每台机器人一个条目）。结构要与 shipped
   ``examples/embodiment/config/realworld_collect_data.yaml`` 一致：

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
                - robot_ip: "192.168.1.100"   # 替换为实际 IP
                  node_rank: 0

      env:
        eval:
          use_spacemouse: True
          override_cfg:
            target_ee_pose: [0.5, 0.0, 0.3, 0.0, 3.14, 0.0]
            success_hold_steps: 3

      runner:
        num_data_episodes: 50

3. 启动采集（支持通过第一个参数传入自定义配置名）：

   .. code-block:: bash

      bash examples/embodiment/collect_data.sh
      # 或使用自定义配置：
      bash examples/embodiment/collect_data.sh my_custom_config

4. 用 SpaceMouse（或 GELLO）操作机器人完成任务。成功次数达到
   ``num_data_episodes`` 后，脚本自动保存并退出，日志和数据存放在
   ``logs/{timestamp}/`` 目录下。

   如需使用 GELLO 替代 SpaceMouse，可使用专用配置：

   .. code-block:: bash

      bash examples/embodiment/collect_data.sh realworld_collect_data_gello

   GELLO 的安装与设置请参考 :doc:`../../examples/embodied/franka`。

----

最佳实践
--------

**Episode 采集（CollectEpisode）**

- 图像数据体积大，若磁盘有限，可配合 ``only_success=True`` 过滤失败 episode。
- 分布式训练时，每个 Worker 设置不同 ``rank``，避免文件名冲突。

**真机 Replay Buffer 采集**

- 优先保证轨迹质量：成功率低时减小 ``success_hold_steps``，降低判定门槛，
  或在 ``target_ee_pose`` 设置更宽松的容差。
- 收集完成后可用 ``TrajectoryReplayBuffer.load_checkpoint()`` 检查数据条数，
  确认达到预期数量后再启动训练。
- 如需追加数据，只需重新运行脚本并指向同一 ``demos`` 目录，
  buffer 的 ``auto_save=True`` 会增量写入而不覆盖已有轨迹。

可视化工具
----------

采集结束后，可以直接基于 ``logs/{timestamp}/`` 下的产物查看两种数据格式。

**Replay buffer 轨迹**

使用已有的 replay buffer 可视化脚本检查 ``logs/{timestamp}/demos/``：

.. code-block:: bash

   python toolkits/replay_buffer/visualize.py \
       --replay_dir logs/{timestamp}/demos

无显示环境可使用：

.. code-block:: bash

   python toolkits/replay_buffer/visualize_headless.py \
       --replay_dir logs/{timestamp}/demos \
       --output viz.png

**LeRobot 数据集**

使用 ``toolkits/lerobot/visualize_lerobot_dataset.py`` 将 LeRobot 数据集
展开为按 episode 分目录的 ``.jpg`` 图像和 ``.txt`` step 元数据。可视化
工具会直接读取 ``--dataset-path`` 下的 ``meta/info.json``，因此该参数应
指向 **某个具体的** LeRobot root（``save_dir/rank_*/id_*``），而不是父
目录 ``collected_data/``：

.. code-block:: bash

   python toolkits/lerobot/visualize_lerobot_dataset.py \
       --dataset-path logs/{timestamp}/collected_data/rank_0/id_0 \
       --output-dir logs/{timestamp}/collected_data_visualized/rank_0_id_0

如需查看所有采集 root，可对每个 ``rank_*/id_*`` 目录重复执行（例如用
shell 循环）。Replay-buffer 转换器则不同，它接受父目录
``collected_data/`` 作为输入，会自动遍历嵌套的
``rank_*/id_*/data/**/*.parquet`` root。

该工具会读取指定 root 下的 ``meta/info.json`` 和各个
``episode_*.parquet`` 文件，输出类似
``episode_000000/step_000003_image.jpg`` 与
``episode_000000/step_000003.txt`` 的结构，便于快速人工检查。
