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

import copy
import io
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.data.lerobot_replay_buffer import (
    convert_lerobot_dataset_to_replay_buffer,
    convert_lerobot_frames_to_trajectories,
    lerobot_episode_to_trajectory,
    load_lerobot_parquet_frames,
    write_lerobot_frames_to_replay_buffer,
)
from rlinf.data.replay_buffer import TrajectoryReplayBuffer


def _frame(
    frame_index: int,
    *,
    episode_index: int = 0,
    done: bool = False,
    intervene: bool = True,
    action_key: str = "actions",
) -> dict:
    return {
        "episode_index": episode_index,
        "frame_index": frame_index,
        "state": np.array([frame_index, frame_index + 0.5], dtype=np.float32),
        action_key: np.array([frame_index + 10.0], dtype=np.float32),
        "done": np.array([done], dtype=bool),
        "is_success": np.array([done], dtype=bool),
        "intervene_flag": np.array([intervene], dtype=bool),
    }


def _frame_with_next(
    frame_index: int,
    *,
    episode_index: int = 0,
    reward: float = 0.0,
    done: bool = False,
    terminated: bool | None = None,
    truncated: bool = False,
    intervene: bool = True,
    action_key: str = "actions",
) -> dict:
    frame = _frame(
        frame_index,
        episode_index=episode_index,
        done=done,
        intervene=intervene,
        action_key=action_key,
    )
    frame["next_state"] = np.array(
        [frame_index + 1.0, frame_index + 1.5], dtype=np.float32
    )
    frame["rewards"] = np.array([reward], dtype=np.float32)
    frame["truncated"] = np.array([truncated], dtype=bool)
    if terminated is not None:
        frame["terminated"] = np.array([terminated], dtype=bool)
    return frame


def _load_test_replay_buffer(save_path) -> TrajectoryReplayBuffer:
    buffer = TrajectoryReplayBuffer(
        seed=1234,
        enable_cache=True,
        cache_size=2,
        sample_window_size=2,
        auto_save=False,
    )
    buffer.load_checkpoint(str(save_path))
    return buffer


def _run_lerobot_converter(dataset_path, save_path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "rlinf.data.lerobot_replay_buffer",
            "--dataset-path",
            str(dataset_path),
            "--save-path",
            str(save_path),
        ],
        check=True,
    )


def test_lerobot_episode_rejects_terminal_action_without_next_observation():
    frames = [
        _frame(0, intervene=False),
        _frame(1, intervene=True),
        _frame(2, done=True, intervene=True),
    ]

    with pytest.raises(ValueError, match="next observation"):
        lerobot_episode_to_trajectory(frames)


def test_lerobot_episode_uses_explicit_next_observation_for_terminal_action():
    frames = [
        _frame_with_next(0, reward=0.5, intervene=False),
        _frame_with_next(1, reward=1.0, done=True, terminated=True, intervene=True),
    ]

    trajectory = lerobot_episode_to_trajectory(frames)

    assert trajectory.actions.shape == (2, 1, 1)
    torch.testing.assert_close(
        trajectory.actions[:, 0, 0],
        torch.tensor([10.0, 11.0]),
    )
    torch.testing.assert_close(
        trajectory.curr_obs["states"][:, 0],
        torch.tensor([[0.0, 0.5], [1.0, 1.5]]),
    )
    torch.testing.assert_close(
        trajectory.next_obs["states"][:, 0],
        torch.tensor([[1.0, 1.5], [2.0, 2.5]]),
    )
    torch.testing.assert_close(
        trajectory.dones[:, 0, 0],
        torch.tensor([False, True]),
    )
    torch.testing.assert_close(
        trajectory.rewards[:, 0, 0],
        torch.tensor([0.5, 1.0]),
    )
    torch.testing.assert_close(
        trajectory.terminations[:, 0, 0],
        torch.tensor([False, True]),
    )
    torch.testing.assert_close(
        trajectory.intervene_flags[:, 0, 0],
        torch.tensor([False, True]),
    )


def test_lerobot_episode_infers_termination_from_done_and_false_truncation():
    frames = [_frame_with_next(0, reward=1.0, done=True)]

    trajectory = lerobot_episode_to_trajectory(frames)

    torch.testing.assert_close(trajectory.dones[:, 0, 0], torch.tensor([True]))
    torch.testing.assert_close(trajectory.terminations[:, 0, 0], torch.tensor([True]))
    torch.testing.assert_close(trajectory.truncations[:, 0, 0], torch.tensor([False]))


def test_lerobot_episode_uses_explicit_final_observation_for_terminal_action():
    frames = [
        _frame(0),
        _frame(1, done=True),
        _frame(2),
    ]
    frames[-1].pop("actions")

    trajectory = lerobot_episode_to_trajectory(frames)

    torch.testing.assert_close(
        trajectory.actions[:, 0, 0],
        torch.tensor([10.0, 11.0]),
    )
    torch.testing.assert_close(
        trajectory.dones[:, 0, 0],
        torch.tensor([False, True]),
    )
    torch.testing.assert_close(
        trajectory.rewards[:, 0, 0],
        torch.tensor([0.0, 1.0]),
    )
    torch.testing.assert_close(
        trajectory.next_obs["states"][:, 0],
        torch.tensor([[1.0, 1.5], [2.0, 2.5]]),
    )


def test_lerobot_episode_rejects_noncontiguous_frame_indices():
    frames = [
        _frame(0, done=False),
        _frame(2, done=True),
    ]
    frames[-1].pop("actions")

    with pytest.raises(ValueError, match="contiguous"):
        lerobot_episode_to_trajectory(frames)


def test_lerobot_episode_rejects_explicit_non_terminal_final_observation():
    frames = [
        _frame(0),
        _frame(1, done=False),
    ]
    frames[-1].pop("actions")
    frames[-1]["terminated"] = np.array([False], dtype=bool)
    frames[-1]["truncated"] = np.array([False], dtype=bool)

    with pytest.raises(ValueError, match="final action frame"):
        lerobot_episode_to_trajectory(frames)


def test_lerobot_legacy_terminal_metadata_uses_target_frame():
    frames = [
        _frame(0, done=False),
        _frame(1, done=False),
    ]
    frames[-1].pop("actions")
    frames[-1].pop("done")
    frames[-1]["terminated"] = np.array([True], dtype=bool)
    frames[-1]["truncated"] = np.array([False], dtype=bool)

    trajectory = lerobot_episode_to_trajectory(frames)

    torch.testing.assert_close(trajectory.dones[:, 0, 0], torch.tensor([True]))
    torch.testing.assert_close(trajectory.terminations[:, 0, 0], torch.tensor([True]))
    torch.testing.assert_close(trajectory.truncations[:, 0, 0], torch.tensor([False]))


@pytest.mark.parametrize("field_name", ["terminated", "truncated"])
def test_lerobot_episode_rejects_partial_false_final_split_metadata_without_done(
    field_name,
):
    frames = [
        _frame(0),
        _frame(1),
    ]
    frames[0].pop("done")
    frames[-1].pop("actions")
    frames[-1].pop("done")
    frames[-1][field_name] = np.array([False], dtype=bool)

    with pytest.raises(ValueError, match="terminal metadata|final action frame"):
        lerobot_episode_to_trajectory(frames)


def test_lerobot_legacy_final_observation_reward_takes_precedence():
    frames = [
        _frame(0, done=False),
        _frame(1, done=True),
    ]
    frames[0]["rewards"] = np.array([0.0], dtype=np.float32)
    frames[-1].pop("actions")
    frames[-1]["rewards"] = np.array([7.0], dtype=np.float32)

    trajectory = lerobot_episode_to_trajectory(frames)

    torch.testing.assert_close(trajectory.rewards[:, 0, 0], torch.tensor([7.0]))


def test_lerobot_terminal_metadata_must_be_scalar():
    frames = [_frame_with_next(0, reward=0.5, done=False)]
    frames[0]["done"] = np.array([False, True], dtype=bool)

    with pytest.raises(ValueError, match="done.*scalar"):
        lerobot_episode_to_trajectory(frames)


def test_lerobot_episode_rejects_vector_rewards():
    frames = [_frame_with_next(0, done=True, terminated=True)]
    frames[0]["rewards"] = np.array([0.5, 1.0], dtype=np.float32)

    with pytest.raises(ValueError, match="reward.*scalar"):
        lerobot_episode_to_trajectory(frames)


def test_lerobot_episode_rejects_vector_success_flag_for_reward_fallback():
    frames = [_frame_with_next(0, done=True, terminated=True)]
    frames[0].pop("rewards")
    frames[0]["is_success"] = np.array([True, False], dtype=bool)

    with pytest.raises(ValueError, match="is_success.*scalar"):
        lerobot_episode_to_trajectory(frames)


def test_lerobot_episode_preserves_truncation_separate_from_termination():
    frames = [
        _frame_with_next(
            0,
            done=True,
            terminated=False,
            truncated=True,
            reward=0.25,
        ),
    ]

    trajectory = lerobot_episode_to_trajectory(frames)

    torch.testing.assert_close(trajectory.dones[:, 0, 0], torch.tensor([True]))
    torch.testing.assert_close(trajectory.truncations[:, 0, 0], torch.tensor([True]))
    torch.testing.assert_close(trajectory.terminations[:, 0, 0], torch.tensor([False]))
    torch.testing.assert_close(trajectory.rewards[:, 0, 0], torch.tensor([0.25]))


def test_lerobot_episode_stacks_indexed_multiview_image_keys():
    frames = [_frame_with_next(0, done=True, terminated=True)]
    frames[0]["extra_view_image-0"] = np.zeros((2, 3, 3), dtype=np.uint8)
    frames[0]["extra_view_image-1"] = np.ones((2, 3, 3), dtype=np.uint8)
    frames[0]["next_extra_view_image-0"] = np.full((2, 3, 3), 2, dtype=np.uint8)
    frames[0]["next_extra_view_image-1"] = np.full((2, 3, 3), 3, dtype=np.uint8)

    trajectory = lerobot_episode_to_trajectory(frames)

    assert trajectory.curr_obs["extra_view_images"].shape == (1, 1, 2, 2, 3, 3)
    assert trajectory.next_obs["extra_view_images"].shape == (1, 1, 2, 2, 3, 3)
    assert torch.equal(
        trajectory.curr_obs["extra_view_images"][0, 0, :, 0, 0, 0],
        torch.tensor([0, 1], dtype=torch.uint8),
    )
    assert torch.equal(
        trajectory.next_obs["extra_view_images"][0, 0, :, 0, 0, 0],
        torch.tensor([2, 3], dtype=torch.uint8),
    )


def test_lerobot_episode_ignores_all_null_optional_image_columns():
    frames = [_frame_with_next(0, done=True, terminated=True)]
    frames[0]["extra_view_image"] = None
    frames[0]["next_extra_view_image"] = None

    trajectory = lerobot_episode_to_trajectory(frames)

    assert "extra_view_images" not in trajectory.curr_obs
    assert "extra_view_images" not in trajectory.next_obs


def test_lerobot_episode_ignores_nested_null_optional_image_metadata():
    frames = [_frame_with_next(0, done=True, terminated=True)]
    frames[0]["extra_view_image"] = {"bytes": None, "path": None}
    frames[0]["next_extra_view_image"] = {"array": None, "path": None}

    trajectory = lerobot_episode_to_trajectory(frames)

    assert "extra_view_images" not in trajectory.curr_obs
    assert "extra_view_images" not in trajectory.next_obs


def test_lerobot_episode_rejects_partially_null_optional_image_columns():
    frames = [
        _frame_with_next(0),
        _frame_with_next(1, done=True, terminated=True),
    ]
    frames[0]["extra_view_image"] = np.zeros((2, 3, 3), dtype=np.uint8)
    frames[0]["next_extra_view_image"] = np.zeros((2, 3, 3), dtype=np.uint8)
    frames[1]["extra_view_image"] = None
    frames[1]["next_extra_view_image"] = None

    with pytest.raises(ValueError, match="extra_view_images.*frame 1"):
        lerobot_episode_to_trajectory(frames)


def test_lerobot_episode_prefers_indexed_images_over_null_aggregate_column():
    frames = [_frame_with_next(0, done=True, terminated=True)]
    frames[0]["extra_view_images"] = None
    frames[0]["next_extra_view_images"] = None
    frames[0]["extra_view_image-0"] = np.zeros((2, 3, 3), dtype=np.uint8)
    frames[0]["next_extra_view_image-0"] = np.ones((2, 3, 3), dtype=np.uint8)

    trajectory = lerobot_episode_to_trajectory(frames)

    assert trajectory.curr_obs["extra_view_images"].shape == (1, 1, 1, 2, 3, 3)
    assert trajectory.next_obs["extra_view_images"][0, 0, 0, 0, 0, 0].item() == 1


def test_lerobot_episode_rejects_inconsistent_multiview_indices_across_frames():
    frames = [
        _frame_with_next(0),
        _frame_with_next(1, done=True, terminated=True),
    ]
    frames[0]["extra_view_image-0"] = np.zeros((2, 3, 3), dtype=np.uint8)
    frames[0]["extra_view_image-1"] = np.ones((2, 3, 3), dtype=np.uint8)
    frames[1]["extra_view_image-1"] = np.full((2, 3, 3), 2, dtype=np.uint8)
    frames[1]["extra_view_image-2"] = np.full((2, 3, 3), 3, dtype=np.uint8)

    with pytest.raises(ValueError, match="indexed.*extra_view_images"):
        lerobot_episode_to_trajectory(frames)


def test_lerobot_episode_rejects_multiview_schema_mode_changes():
    frames = [
        _frame_with_next(0),
        _frame_with_next(1, done=True, terminated=True),
    ]
    frames[0]["extra_view_images"] = np.zeros((1, 2, 3, 3), dtype=np.uint8)
    frames[0]["next_extra_view_images"] = np.zeros((1, 2, 3, 3), dtype=np.uint8)
    frames[1]["extra_view_image-0"] = np.ones((2, 3, 3), dtype=np.uint8)
    frames[1]["next_extra_view_image-0"] = np.ones((2, 3, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="extra_view_images.*direct.*indexed"):
        lerobot_episode_to_trajectory(frames)


def test_convert_lerobot_frames_to_trajectories_groups_episodes_and_accepts_action_alias():
    frames = [
        _frame_with_next(1, episode_index=1, action_key="action"),
        _frame_with_next(0, episode_index=0, action_key="action"),
        _frame_with_next(
            1, episode_index=0, done=True, terminated=True, action_key="action"
        ),
        _frame_with_next(0, episode_index=1, action_key="action"),
        _frame_with_next(
            2, episode_index=1, done=True, terminated=True, action_key="action"
        ),
    ]

    trajectories = convert_lerobot_frames_to_trajectories(frames)

    assert len(trajectories) == 2
    assert trajectories[0].actions.shape == (2, 1, 1)
    assert trajectories[1].actions.shape == (3, 1, 1)
    torch.testing.assert_close(
        trajectories[1].curr_obs["states"][:, 0, 0],
        torch.tensor([0.0, 1.0, 2.0]),
    )


def test_lerobot_episode_to_trajectory_accepts_observation_state_alias():
    frames = [_frame_with_next(0, done=True, terminated=True)]
    for frame in frames:
        frame["observation.state"] = frame.pop("state")
        frame["next_observation.state"] = frame.pop("next_state")

    trajectory = lerobot_episode_to_trajectory(frames)

    torch.testing.assert_close(
        trajectory.curr_obs["states"][:, 0],
        torch.tensor([[0.0, 0.5]]),
    )


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    image_module = pytest.importorskip("PIL.Image")
    image = image_module.new("RGB", (2, 3), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_lerobot_episode_to_trajectory_decodes_image_metadata_by_default():
    frames = [_frame_with_next(0, done=True, terminated=True)]
    for frame in frames:
        frame["image"] = {"bytes": _png_bytes((1, 2, 3)), "path": "image.png"}
        frame["next_image"] = {
            "bytes": _png_bytes((4, 5, 6)),
            "path": "next_image.png",
        }

    trajectory = lerobot_episode_to_trajectory(frames)

    assert trajectory.curr_obs["main_images"].shape == (1, 1, 3, 2, 3)
    assert trajectory.next_obs["main_images"].shape == (1, 1, 3, 2, 3)
    assert trajectory.curr_obs["main_images"][0, 0, 0, 0].tolist() == [1, 2, 3]
    assert trajectory.next_obs["main_images"][0, 0, 0, 0].tolist() == [4, 5, 6]


def test_lerobot_episode_to_trajectory_decodes_relative_image_paths(tmp_path):
    (tmp_path / "image.png").write_bytes(_png_bytes((7, 8, 9)))
    (tmp_path / "next_image.png").write_bytes(_png_bytes((10, 11, 12)))
    frames = [_frame_with_next(0, done=True, terminated=True)]
    for frame in frames:
        frame["_lerobot_dataset_root"] = str(tmp_path)
        frame["image"] = {"path": "image.png"}
        frame["next_image"] = {"path": "next_image.png"}

    trajectory = lerobot_episode_to_trajectory(frames)

    assert trajectory.curr_obs["main_images"][0, 0, 0, 0].tolist() == [7, 8, 9]
    assert trajectory.next_obs["main_images"][0, 0, 0, 0].tolist() == [
        10,
        11,
        12,
    ]


def test_lerobot_image_path_cannot_escape_dataset_root(tmp_path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (tmp_path / "outside.png").write_bytes(b"not decoded")

    frames = [_frame_with_next(0, done=True, terminated=True)]
    for frame in frames:
        frame["_lerobot_dataset_root"] = str(dataset_root)
        frame["image"] = {"path": "../outside.png"}
        frame["next_image"] = np.zeros((2, 3, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="escapes.*dataset root"):
        lerobot_episode_to_trajectory(frames)


def test_lerobot_absolute_image_path_cannot_escape_dataset_root(tmp_path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    outside_image = tmp_path / "outside.png"
    outside_image.write_bytes(b"not decoded")

    frames = [_frame_with_next(0, done=True, terminated=True)]
    for frame in frames:
        frame["_lerobot_dataset_root"] = str(dataset_root)
        frame["image"] = {"path": str(outside_image)}
        frame["next_image"] = np.zeros((2, 3, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="escapes.*dataset root"):
        lerobot_episode_to_trajectory(frames)


def test_lerobot_episode_to_trajectory_decodes_top_level_relative_image_paths(
    tmp_path,
):
    (tmp_path / "image.png").write_bytes(_png_bytes((13, 14, 15)))
    (tmp_path / "next_image.png").write_bytes(_png_bytes((16, 17, 18)))
    frames = [_frame_with_next(0, done=True, terminated=True)]
    for frame in frames:
        frame["_lerobot_dataset_root"] = str(tmp_path)
        frame["image"] = "image.png"
        frame["next_image"] = "next_image.png"

    trajectory = lerobot_episode_to_trajectory(frames)

    assert trajectory.curr_obs["main_images"][0, 0, 0, 0].tolist() == [13, 14, 15]
    assert trajectory.next_obs["main_images"][0, 0, 0, 0].tolist() == [
        16,
        17,
        18,
    ]


def test_lerobot_episode_rejects_non_image_shaped_arrays_for_image_fields():
    frames = [_frame_with_next(0, done=True, terminated=True)]
    frames[0]["image"] = np.array([1, 2, 3], dtype=np.uint8)
    frames[0]["next_image"] = np.zeros((2, 3, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="image.*shape"):
        lerobot_episode_to_trajectory(frames)


def test_lerobot_episode_to_trajectory_decodes_indexed_image_metadata():
    frames = [_frame_with_next(0, done=True, terminated=True)]
    frames[0]["extra_view_image-0"] = {"bytes": _png_bytes((1, 1, 1))}
    frames[0]["extra_view_image-1"] = {"bytes": _png_bytes((2, 2, 2))}
    frames[0]["next_extra_view_image-0"] = {"bytes": _png_bytes((3, 3, 3))}
    frames[0]["next_extra_view_image-1"] = {"bytes": _png_bytes((4, 4, 4))}

    trajectory = lerobot_episode_to_trajectory(frames)

    assert trajectory.curr_obs["extra_view_images"].shape == (1, 1, 2, 3, 2, 3)
    assert trajectory.curr_obs["extra_view_images"][0, 0, :, 0, 0, 0].tolist() == [
        1,
        2,
    ]
    assert trajectory.next_obs["extra_view_images"][0, 0, :, 0, 0, 0].tolist() == [
        3,
        4,
    ]


def test_lerobot_episode_to_trajectory_allows_encoded_image_metadata_in_state_only_mode():
    frames = [_frame_with_next(0, done=True, terminated=True)]
    for frame in frames:
        frame["image"] = {"path": "episode/image.png"}
        frame["next_image"] = {"path": "episode/next_image.png"}

    trajectory = lerobot_episode_to_trajectory(frames, state_only=True)

    assert "states" in trajectory.curr_obs
    assert "main_images" not in trajectory.curr_obs


def test_convert_lerobot_frames_rejects_mixed_observation_schema_across_episodes():
    frames = [
        _frame_with_next(0, episode_index=0, done=True, terminated=True),
        _frame_with_next(0, episode_index=1, done=True, terminated=True),
    ]
    frames[0]["image"] = np.zeros((2, 3, 3), dtype=np.uint8)
    frames[0]["next_image"] = np.zeros((2, 3, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="observation schema.*main_images"):
        convert_lerobot_frames_to_trajectories(frames)


def test_lerobot_episode_uses_legacy_final_frame_terminal_metadata():
    frames = [_frame(0), _frame(1), _frame(2, done=True)]
    frames[-1].pop("actions")
    frames[-1]["rewards"] = np.array([0.75], dtype=np.float32)

    trajectory = lerobot_episode_to_trajectory(frames)

    torch.testing.assert_close(
        trajectory.dones[:, 0, 0],
        torch.tensor([False, True]),
    )
    torch.testing.assert_close(
        trajectory.rewards[:, 0, 0],
        torch.tensor([0.0, 0.75]),
    )


def test_lerobot_episode_infers_legacy_final_termination_from_done_only():
    frames = [_frame(0), _frame(1, done=True)]
    frames[-1].pop("actions")
    frames[-1]["truncated"] = np.array([False], dtype=bool)

    trajectory = lerobot_episode_to_trajectory(frames)

    torch.testing.assert_close(
        trajectory.dones[:, 0, 0],
        torch.tensor([True]),
    )
    torch.testing.assert_close(
        trajectory.terminations[:, 0, 0],
        torch.tensor([True]),
    )
    torch.testing.assert_close(
        trajectory.truncations[:, 0, 0],
        torch.tensor([False]),
    )


def test_lerobot_episode_rejects_legacy_final_frame_conflicting_terminal_flags():
    frames = [_frame(0), _frame(1, done=True)]
    frames[-1].pop("actions")
    frames[-1]["terminated"] = np.array([False], dtype=bool)
    frames[-1]["truncated"] = np.array([False], dtype=bool)

    with pytest.raises(ValueError, match="done.*terminated"):
        lerobot_episode_to_trajectory(frames)


def test_lerobot_episode_rejects_conflicting_done_and_termination_flags():
    frames = [_frame_with_next(0, done=False, terminated=True)]

    with pytest.raises(ValueError, match="done.*terminated"):
        lerobot_episode_to_trajectory(frames)


def test_lerobot_episode_rejects_explicit_next_final_frame_marked_non_terminal():
    frames = [_frame_with_next(0, done=False, terminated=False, truncated=False)]

    with pytest.raises(ValueError, match="final.*non-terminal"):
        lerobot_episode_to_trajectory(frames)


def test_lerobot_episode_rejects_explicit_next_middle_terminal_frame():
    frames = [
        _frame_with_next(0, done=True, terminated=True),
        _frame_with_next(1, done=True, terminated=True),
    ]

    with pytest.raises(ValueError, match="terminal.*before final"):
        lerobot_episode_to_trajectory(frames)


def test_lerobot_episode_rejects_mismatched_state_and_next_state_shapes():
    frames = [_frame_with_next(0, done=True, terminated=True)]
    frames[0]["next_state"] = np.array([1.0, 1.5, 2.0], dtype=np.float32)

    with pytest.raises(ValueError, match="mismatched states"):
        lerobot_episode_to_trajectory(frames)


def test_lerobot_episode_normalizes_singleton_state_vectors():
    frames = [_frame_with_next(0, done=True, terminated=True)]
    frames[0]["state"] = np.array([[0.0, 0.5]], dtype=np.float32)
    frames[0]["next_state"] = np.array([[1.0, 1.5]], dtype=np.float32)

    trajectory = lerobot_episode_to_trajectory(frames)

    assert trajectory.curr_obs["states"].shape == (1, 1, 2)
    assert trajectory.next_obs["states"].shape == (1, 1, 2)
    torch.testing.assert_close(
        trajectory.curr_obs["states"][:, 0], torch.tensor([[0.0, 0.5]])
    )


def test_convert_lerobot_frames_to_trajectories_accepts_single_explicit_next_frame():
    frames = [_frame_with_next(0, done=True, terminated=True)]

    trajectories = convert_lerobot_frames_to_trajectories(frames)

    assert len(trajectories) == 1
    torch.testing.assert_close(
        trajectories[0].dones[:, 0, 0],
        torch.tensor([True]),
    )


def test_lerobot_episode_normalizes_matrix_action_to_replay_action_shape():
    frames = [_frame_with_next(0, done=True, terminated=True)]
    frames[0]["actions"] = np.array([[10.0, 11.0]], dtype=np.float32)

    trajectory = lerobot_episode_to_trajectory(frames)

    assert trajectory.actions.shape == (1, 1, 2)
    assert trajectory.intervene_flags.shape == (1, 1, 2)
    torch.testing.assert_close(
        trajectory.actions[0, 0],
        torch.tensor([10.0, 11.0]),
    )


def test_lerobot_episode_rejects_multi_element_intervene_flag():
    frames = [_frame_with_next(0, done=True, terminated=True)]
    frames[0]["intervene_flag"] = np.array([False, True], dtype=bool)

    with pytest.raises(ValueError, match="intervene_flag.*scalar"):
        lerobot_episode_to_trajectory(frames)


def test_lerobot_episode_rejects_missing_next_image_for_explicit_next_state():
    frames = [_frame_with_next(0, done=True, terminated=True)]
    frames[0]["image"] = np.zeros((2, 3, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="next_obs.*main_images"):
        lerobot_episode_to_trajectory(frames)


def test_write_lerobot_frames_to_replay_buffer_checkpoint_roundtrip(tmp_path):
    frames = [
        _frame_with_next(0),
        _frame_with_next(1),
        _frame_with_next(2, done=True, terminated=True),
    ]
    save_path = tmp_path / "replay_buffer"

    write_lerobot_frames_to_replay_buffer(frames, str(save_path))

    buffer = TrajectoryReplayBuffer(
        seed=1234,
        enable_cache=True,
        cache_size=2,
        sample_window_size=2,
        auto_save=False,
    )
    buffer.load_checkpoint(str(save_path))

    assert buffer.get_stats()["num_trajectories"] == 1
    assert buffer.get_stats()["total_samples"] == 3
    batch = buffer.sample(2)
    assert "actions" in batch
    assert "intervene_flags" in batch
    assert batch["actions"].shape == (2, 1)
    assert batch["curr_obs"]["states"].shape == (2, 2)


def test_load_lerobot_parquet_frames_reads_local_dataset_in_episode_order(tmp_path):
    pandas = pytest.importorskip("pandas")
    data_dir = tmp_path / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    pandas.DataFrame(
        [
            _frame(1, episode_index=0),
            _frame(0, episode_index=0),
            _frame(2, episode_index=0, done=True),
        ]
    ).to_parquet(data_dir / "episode_000000.parquet")

    frames = load_lerobot_parquet_frames(str(tmp_path))

    assert [int(frame["frame_index"]) for frame in frames] == [0, 1, 2]


def test_load_lerobot_parquet_frames_requires_episode_and_frame_indices(tmp_path):
    pandas = pytest.importorskip("pandas")
    data_dir = tmp_path / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    frame = _frame(0)
    frame.pop("episode_index")
    pandas.DataFrame([frame]).to_parquet(data_dir / "episode_000000.parquet")

    with pytest.raises(ValueError, match="episode_index"):
        load_lerobot_parquet_frames(str(tmp_path))


def test_convert_lerobot_rejects_duplicate_frame_indices(tmp_path):
    pandas = pytest.importorskip("pandas")
    data_dir = tmp_path / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    pandas.DataFrame(
        [
            _frame_with_next(0, episode_index=0),
            _frame_with_next(0, episode_index=0),
        ]
    ).to_parquet(data_dir / "episode_000000.parquet")

    with pytest.raises(ValueError, match="contiguous"):
        convert_lerobot_dataset_to_replay_buffer(
            str(tmp_path), str(tmp_path / "buffer.pt")
        )


def test_load_lerobot_parquet_frames_state_only_skips_image_columns(tmp_path):
    pandas = pytest.importorskip("pandas")
    data_dir = tmp_path / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    frame = _frame_with_next(0, done=True, terminated=True)
    frame["image"] = {"path": "episode/image.png"}
    pandas.DataFrame([frame]).to_parquet(data_dir / "episode_000000.parquet")

    frames = load_lerobot_parquet_frames(str(tmp_path), state_only=True)

    assert "image" not in frames[0]
    assert "state" in frames[0]
    assert "next_state" in frames[0]


def test_load_lerobot_parquet_frames_reads_collector_parent_without_merging_ids(
    tmp_path,
):
    pandas = pytest.importorskip("pandas")
    collected_data = tmp_path / "collected_data"
    for dataset_id in ("id_0", "id_1"):
        data_dir = collected_data / "rank_0" / dataset_id / "data" / "chunk-000"
        data_dir.mkdir(parents=True)
        pandas.DataFrame(
            [
                _frame_with_next(0, episode_index=0),
                _frame_with_next(1, episode_index=0, done=True, terminated=True),
            ]
        ).to_parquet(data_dir / "episode_000000.parquet")

    frames = load_lerobot_parquet_frames(str(collected_data))
    trajectories = convert_lerobot_frames_to_trajectories(frames)

    assert len(trajectories) == 2


def test_lerobot_replay_buffer_cli_converts_collector_parent_layout(tmp_path):
    pandas = pytest.importorskip("pandas")
    data_dir = tmp_path / "collected_data" / "rank_0" / "id_0" / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    pandas.DataFrame(
        [
            _frame_with_next(0, episode_index=0),
            _frame_with_next(1, episode_index=0, done=True, terminated=True),
        ]
    ).to_parquet(data_dir / "episode_000000.parquet")
    save_path = tmp_path / "replay_buffer"

    _run_lerobot_converter(tmp_path / "collected_data", save_path)

    buffer = _load_test_replay_buffer(save_path)

    assert buffer.get_stats()["num_trajectories"] == 1
    assert buffer.get_stats()["total_samples"] == 2


def test_lerobot_replay_buffer_cli_converts_relative_image_paths(tmp_path):
    pandas = pytest.importorskip("pandas")
    data_root = tmp_path / "dataset"
    data_dir = data_root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    (data_root / "image.png").write_bytes(_png_bytes((21, 22, 23)))
    (data_root / "next_image.png").write_bytes(_png_bytes((24, 25, 26)))
    frame = _frame_with_next(0, episode_index=0, done=True, terminated=True)
    frame["image"] = "image.png"
    frame["next_image"] = "next_image.png"
    pandas.DataFrame([frame]).to_parquet(data_dir / "episode_000000.parquet")
    save_path = tmp_path / "replay_buffer"

    _run_lerobot_converter(data_root, save_path)

    buffer = _load_test_replay_buffer(save_path)
    trajectory = buffer._load_trajectory(0, "lerobot")

    assert trajectory.curr_obs["main_images"][0, 0, 0, 0].tolist() == [21, 22, 23]
    assert trajectory.next_obs["main_images"][0, 0, 0, 0].tolist() == [24, 25, 26]


def test_collect_episode_lerobot_export_includes_transition_next_obs_and_flags():
    from rlinf.envs.wrappers.collect_episode import CollectEpisode

    wrapper = CollectEpisode.__new__(CollectEpisode)
    wrapper.num_envs = 1
    buf = {
        "observations": [
            {"state": np.array([0.0, 0.5], dtype=np.float32)},
            {"state": np.array([1.0, 1.5], dtype=np.float32)},
            {"state": np.array([2.0, 2.5], dtype=np.float32)},
        ],
        "actions": [
            np.array([10.0], dtype=np.float32),
            np.array([11.0], dtype=np.float32),
        ],
        "rewards": [0.0, 0.5, 1.0],
        "terminated": [False, False, True],
        "truncated": [False, False, False],
        "infos": [
            {},
            {"intervene_flag": np.array([False], dtype=bool)},
            {"intervene_flag": np.array([True], dtype=bool)},
        ],
    }

    frames = wrapper._buffer_to_lerobot_ep(buf, env_idx=0, is_success=True)
    trajectory = lerobot_episode_to_trajectory(frames)

    assert len(frames) == 2
    torch.testing.assert_close(
        trajectory.actions[:, 0, 0],
        torch.tensor([10.0, 11.0]),
    )
    torch.testing.assert_close(
        trajectory.next_obs["states"][:, 0],
        torch.tensor([[1.0, 1.5], [2.0, 2.5]]),
    )
    torch.testing.assert_close(
        trajectory.rewards[:, 0, 0],
        torch.tensor([0.5, 1.0]),
    )
    torch.testing.assert_close(
        trajectory.dones[:, 0, 0],
        torch.tensor([False, True]),
    )
    torch.testing.assert_close(
        trajectory.terminations[:, 0, 0],
        torch.tensor([False, True]),
    )


def test_collect_episode_final_intervene_action_keeps_action_vector():
    from rlinf.envs.wrappers.collect_episode import CollectEpisode

    wrapper = CollectEpisode.__new__(CollectEpisode)
    wrapper.num_envs = 1
    wrapper._global_step = 0
    wrapper._pending_obs = [None]
    wrapper._pending_info = [None]
    wrapper._episode_success = [False]
    wrapper._buffers = [wrapper._new_buffer()]
    wrapper._record_reset_obs({"state": np.array([[0.0, 0.5]], dtype=np.float32)})

    wrapper._record_step(
        action=np.array([[10.0, 20.0]], dtype=np.float32),
        obs={"state": np.array([[99.0, 99.5]], dtype=np.float32)},
        reward=np.array([1.0], dtype=np.float32),
        terminated=np.array([True]),
        truncated=np.array([False]),
        info={
            "final_observation": {"state": np.array([[1.0, 1.5]], dtype=np.float32)},
            "final_info": {
                "intervene_flag": np.array([True], dtype=bool),
                "intervene_action": np.array([30.0, 40.0], dtype=np.float32),
            },
        },
    )

    frames = wrapper._buffer_to_lerobot_ep(
        wrapper._buffers[0], env_idx=0, is_success=True
    )

    np.testing.assert_allclose(frames[0]["actions"], np.array([30.0, 40.0]))


def test_collect_episode_final_intervene_action_selects_terminal_chunk_action():
    from rlinf.envs.wrappers.collect_episode import CollectEpisode

    wrapper = CollectEpisode.__new__(CollectEpisode)
    wrapper.num_envs = 1
    wrapper._global_step = 0
    wrapper._pending_obs = [None]
    wrapper._pending_info = [None]
    wrapper._episode_success = [False]
    wrapper._buffers = [wrapper._new_buffer()]
    wrapper._record_reset_obs({"state": np.array([[0.0, 0.5]], dtype=np.float32)})

    wrapper._record_step(
        action=np.array([[10.0, 20.0]], dtype=np.float32),
        obs={"state": np.array([[99.0, 99.5]], dtype=np.float32)},
        reward=np.array([1.0], dtype=np.float32),
        terminated=np.array([True]),
        truncated=np.array([False]),
        info={
            "final_observation": {"state": np.array([[1.0, 1.5]], dtype=np.float32)},
            "final_info": {
                "intervene_flag": np.array([[False, True]], dtype=bool),
                "intervene_action": np.array([[30.0, 40.0, 50.0, 60.0]]),
                "terminal_chunk_index": np.array([1]),
            },
        },
    )

    frames = wrapper._buffer_to_lerobot_ep(
        wrapper._buffers[0], env_idx=0, is_success=True
    )

    np.testing.assert_allclose(frames[0]["actions"], np.array([50.0, 60.0]))


def test_collect_episode_final_intervene_action_uses_terminal_chunk_index():
    from rlinf.envs.wrappers.collect_episode import CollectEpisode

    info = {
        "intervene_flag": np.array([True, False], dtype=bool),
        "intervene_action": np.array([[30.0, 40.0], [50.0, 60.0]], dtype=np.float32),
        "terminal_chunk_index": np.array([0]),
    }

    CollectEpisode._normalize_final_intervention_info(info)

    np.testing.assert_allclose(info["intervene_action"], np.array([30.0, 40.0]))
    np.testing.assert_array_equal(info["intervene_flag"], np.array([True]))


def _fake_realworld_handle_auto_reset(env, dones, final_obs, infos):
    reset_env_indices = np.arange(0, env.num_envs)[dones]
    final_info = copy.deepcopy(infos)
    obs, reset_infos = env.reset(
        env_idx=reset_env_indices,
        reset_state_ids=(
            env.reset_state_ids[reset_env_indices]
            if env.use_fixed_reset_state_ids
            else None
        ),
    )
    reset_infos["final_observation"] = copy.deepcopy(final_obs)
    reset_infos["final_info"] = final_info
    reset_infos["_final_info"] = dones
    reset_infos["_final_observation"] = dones
    reset_infos["_elapsed_steps"] = dones
    return obs, reset_infos


def _bind_fake_realworld_chunk_methods(env):
    from rlinf.envs.realworld_chunk_utils import run_realworld_chunk_step

    env.chunk_step = run_realworld_chunk_step.__get__(env, type(env))
    env._handle_auto_reset = _fake_realworld_handle_auto_reset.__get__(env, type(env))


def test_collect_episode_chunk_export_uses_true_middle_terminal_step():
    from rlinf.envs.wrappers.collect_episode import CollectEpisode

    class FakeChunkEnv:
        def __init__(self):
            self.step_index = 0
            self.num_envs = 1

        def step(self, actions, auto_reset=False):
            index = self.step_index
            self.step_index += 1
            obs = {
                "state": torch.tensor([[index + 1.0, index + 1.5]], dtype=torch.float32)
            }
            reward = torch.tensor([float(index + 1)], dtype=torch.float32)
            terminated = torch.tensor([index == 1], dtype=torch.bool)
            truncated = torch.tensor([False], dtype=torch.bool)
            info = {
                "intervene_action": torch.tensor(
                    [[30.0, 40.0]] if index == 1 else [[0.0, 0.0]],
                    dtype=torch.float32,
                ),
                "intervene_flag": torch.tensor([index == 1], dtype=torch.bool),
            }
            return obs, reward, terminated, truncated, info

        def reset(self, env_idx=None, reset_state_ids=None):
            return {"state": torch.tensor([[100.0, 100.5]], dtype=torch.float32)}, {}

    env = FakeChunkEnv()
    env.auto_reset = True
    env.ignore_terminations = False
    env.use_fixed_reset_state_ids = False
    env.reset_state_ids = np.array([0])
    _bind_fake_realworld_chunk_methods(env)

    wrapper = CollectEpisode.__new__(CollectEpisode)
    wrapper.env = env
    wrapper.num_envs = 1
    wrapper.only_success = False
    wrapper._global_step = 0
    wrapper._pending_obs = [None]
    wrapper._pending_info = [None]
    wrapper._episode_success = [False]
    wrapper._episode_ids = [0]
    wrapper._buffers = [wrapper._new_buffer()]
    wrapper._record_reset_obs({"state": np.array([[0.0, 0.5]], dtype=np.float32)})
    exported_frames = []

    def flush_episode(env_idx, is_success):
        exported_frames.extend(
            wrapper._buffer_to_lerobot_ep(
                wrapper._buffers[env_idx], env_idx=env_idx, is_success=is_success
            )
        )

    wrapper._flush_episode = flush_episode

    wrapper.chunk_step(np.zeros((1, 3, 2), dtype=np.float32))

    assert len(exported_frames) == 2
    np.testing.assert_allclose(exported_frames[-1]["actions"], np.array([30.0, 40.0]))
    np.testing.assert_allclose(
        exported_frames[-1]["next_state"],
        np.array([2.0, 2.5], dtype=np.float32),
    )
    np.testing.assert_allclose(exported_frames[-1]["rewards"], np.array([2.0]))
    assert exported_frames[-1]["terminated"].item()


def test_collect_episode_rejects_invalid_padded_done_step():
    from rlinf.envs.wrappers.collect_episode import CollectEpisode

    class FakeChunkEnv:
        num_envs = 1

        def chunk_step(self, chunk_actions):
            return (
                [{"state": torch.tensor([[1.0]], dtype=torch.float32)}],
                torch.zeros(1, 1),
                torch.tensor([[True]], dtype=torch.bool),
                torch.tensor([[False]], dtype=torch.bool),
                [{"_valid_step": torch.tensor([False], dtype=torch.bool)}],
            )

    wrapper = CollectEpisode.__new__(CollectEpisode)
    wrapper.env = FakeChunkEnv()
    wrapper.num_envs = 1
    wrapper.only_success = False
    wrapper._global_step = 0
    wrapper._pending_obs = [None]
    wrapper._pending_info = [None]
    wrapper._episode_success = [False]
    wrapper._episode_ids = [0]
    wrapper._buffers = [wrapper._new_buffer()]

    with pytest.raises(ValueError, match="invalid.*done"):
        wrapper.chunk_step(np.zeros((1, 1, 2), dtype=np.float32))


def test_realworld_chunk_step_stops_after_auto_reset_terminal_step():
    class FakeChunkEnv:
        num_envs = 1
        auto_reset = True
        ignore_terminations = False
        use_fixed_reset_state_ids = False
        reset_state_ids = np.array([0])
        call_count = 0

        def step(self, actions, auto_reset=False):
            self.call_count += 1
            index = self.call_count - 1
            obs = {
                "state": torch.tensor([[index + 1.0, index + 1.5]], dtype=torch.float32)
            }
            reward = torch.tensor([float(index + 1)], dtype=torch.float32)
            terminations = torch.tensor([index == 1], dtype=torch.bool)
            truncations = torch.tensor([False], dtype=torch.bool)
            infos = {
                "intervene_action": torch.tensor(
                    [[10.0 + index, 20.0 + index]], dtype=torch.float32
                ),
                "intervene_flag": torch.tensor([index == 1], dtype=torch.bool),
                "episode": {"return": torch.tensor([float(index + 1)])},
            }
            return obs, reward, terminations, truncations, infos

        def reset(self, env_idx=None, reset_state_ids=None):
            return {"state": torch.tensor([[100.0, 100.5]], dtype=torch.float32)}, {}

    env = FakeChunkEnv()
    _bind_fake_realworld_chunk_methods(env)

    _, rewards, terminations, truncations, infos_list = env.chunk_step(
        torch.zeros(1, 3, 2)
    )

    assert env.call_count == 2
    torch.testing.assert_close(rewards, torch.tensor([[1.0, 2.0, 0.0]]))
    torch.testing.assert_close(
        terminations, torch.tensor([[False, True, False]], dtype=torch.bool)
    )
    assert not truncations.any()
    assert len(infos_list) == 3
    assert infos_list[-1]["chunk_intervene_flag"].tolist() == [[False, True, False]]
    torch.testing.assert_close(
        infos_list[-1]["chunk_intervene_action"],
        torch.tensor([[10.0, 20.0, 11.0, 21.0, 0.0, 0.0]]),
    )


def test_realworld_chunk_step_preserves_active_env_tail_after_partial_done():
    class FakeChunkEnv:
        num_envs = 2
        auto_reset = True
        ignore_terminations = False
        use_fixed_reset_state_ids = False
        reset_state_ids = np.array([0, 1])

        def __init__(self):
            self.call_count = 0
            self.actions = []

        def step(self, actions, auto_reset=False):
            self.actions.append(torch.as_tensor(actions).clone())
            index = self.call_count
            self.call_count += 1
            obs = {
                "state": torch.tensor(
                    [[index + 1.0], [index + 10.0]], dtype=torch.float32
                )
            }
            reward = torch.tensor([float(index + 1), float((index + 1) * 10)])
            terminations = torch.tensor([index == 1, False], dtype=torch.bool)
            truncations = torch.tensor([False, False], dtype=torch.bool)
            infos = {}
            return obs, reward, terminations, truncations, infos

        def reset(self, env_idx=None, reset_state_ids=None):
            return {"state": torch.tensor([[100.0], [200.0]], dtype=torch.float32)}, {}

    env = FakeChunkEnv()
    _bind_fake_realworld_chunk_methods(env)

    _, rewards, terminations, _, infos_list = env.chunk_step(torch.ones(2, 3, 2))

    assert env.call_count == 3
    torch.testing.assert_close(
        rewards,
        torch.tensor([[1.0, 2.0, 0.0], [10.0, 20.0, 30.0]]),
    )
    torch.testing.assert_close(
        terminations,
        torch.tensor([[False, True, False], [False, False, False]], dtype=torch.bool),
    )
    torch.testing.assert_close(
        infos_list[-1]["chunk_valid_step"],
        torch.tensor([[True, True, False], [True, True, True]], dtype=torch.bool),
    )
    torch.testing.assert_close(env.actions[-1][0], torch.zeros(2))
    torch.testing.assert_close(env.actions[-1][1], torch.ones(2))


def test_realworld_chunk_step_unit_test_does_not_import_realworld_package(
    monkeypatch,
):
    import builtins

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "rlinf.envs.realworld" or name.startswith("rlinf.envs.realworld."):
            raise AssertionError("unit tests must not import the realworld package")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    test_realworld_chunk_step_stops_after_auto_reset_terminal_step()


def test_realworld_chunk_step_helper_import_does_not_import_realworld_package(
    monkeypatch,
):
    import builtins
    import importlib

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "rlinf.envs.realworld" or name.startswith("rlinf.envs.realworld."):
            raise AssertionError("helper import must not import the realworld package")
        return real_import(name, globals, locals, fromlist, level)

    sys.modules.pop("rlinf.envs.realworld_chunk_utils", None)
    importlib.invalidate_caches()
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    module = importlib.import_module("rlinf.envs.realworld_chunk_utils")

    assert callable(module.run_realworld_chunk_step)


def test_realworld_chunk_step_keeps_mid_chunk_truncation_before_invalid_tail():
    class FakeChunkEnv:
        num_envs = 1
        auto_reset = True
        ignore_terminations = True
        use_fixed_reset_state_ids = False
        reset_state_ids = np.array([0])
        call_count = 0

        def step(self, actions, auto_reset=False):
            self.call_count += 1
            index = self.call_count - 1
            obs = {"state": torch.tensor([[float(index)]], dtype=torch.float32)}
            reward = torch.tensor([float(index + 1)], dtype=torch.float32)
            terminations = torch.tensor([False], dtype=torch.bool)
            truncations = torch.tensor([index == 1], dtype=torch.bool)
            infos = {"episode": {"return": torch.tensor([float(index + 1)])}}
            return obs, reward, terminations, truncations, infos

        def reset(self, env_idx=None, reset_state_ids=None):
            return {"state": torch.tensor([[100.0]], dtype=torch.float32)}, {}

    env = FakeChunkEnv()
    _bind_fake_realworld_chunk_methods(env)

    _, _, terminations, truncations, infos_list = env.chunk_step(torch.zeros(1, 3, 2))

    assert env.call_count == 2
    torch.testing.assert_close(
        terminations, torch.tensor([[False, False, False]], dtype=torch.bool)
    )
    torch.testing.assert_close(
        truncations, torch.tensor([[False, True, False]], dtype=torch.bool)
    )
    assert infos_list[1]["_valid_step"].item()
    assert not infos_list[2]["_valid_step"].item()


def test_realworld_chunk_step_breaks_after_mid_chunk_done_without_auto_reset():
    """When auto_reset=False, run_realworld_chunk_step must stop stepping
    after termination so that post-terminal frames are not recorded as
    valid transitions."""

    class FakeChunkEnv:
        num_envs = 1
        auto_reset = False
        ignore_terminations = False
        use_fixed_reset_state_ids = False
        reset_state_ids = np.array([0])
        call_count = 0

        def step(self, actions, auto_reset=False):
            self.call_count += 1
            index = self.call_count - 1
            obs = {"state": torch.tensor([[float(index)]], dtype=torch.float32)}
            reward = torch.tensor([float(index + 1)], dtype=torch.float32)
            # Terminate at substep 0; any further step() calls would record
            # garbage post-terminal frames.
            terminations = torch.tensor([index == 0], dtype=torch.bool)
            truncations = torch.tensor([False], dtype=torch.bool)
            return obs, reward, terminations, truncations, {}

        def reset(self, env_idx=None, reset_state_ids=None):
            raise AssertionError("reset must not run when auto_reset=False")

    env = FakeChunkEnv()
    _bind_fake_realworld_chunk_methods(env)

    _, rewards, terminations, truncations, infos_list = env.chunk_step(
        torch.zeros(1, 3, 2)
    )

    assert env.call_count == 1, "loop should break after the terminal substep"
    torch.testing.assert_close(rewards, torch.tensor([[1.0, 0.0, 0.0]]))
    torch.testing.assert_close(
        terminations, torch.tensor([[True, False, False]], dtype=torch.bool)
    )
    torch.testing.assert_close(
        truncations, torch.tensor([[False, False, False]], dtype=torch.bool)
    )
    # The terminal substep's valid_step is True; padded tail is False.
    assert infos_list[0]["_valid_step"].item()
    assert not infos_list[1]["_valid_step"].item()
    assert not infos_list[2]["_valid_step"].item()


def test_env_worker_ignore_terminations_collects_mid_chunk_truncation():
    from rlinf.workers.env.env_worker import EnvWorker

    class FakeRealWorldEnv:
        def chunk_step(self, chunk_actions):
            obs_list = [
                {"state": torch.tensor([[0.0]])},
                {"state": torch.tensor([[1.0]])},
                {"state": torch.tensor([[2.0]])},
            ]
            rewards = torch.zeros(1, 3)
            terminations = torch.zeros(1, 3, dtype=torch.bool)
            truncations = torch.tensor([[False, True, False]], dtype=torch.bool)
            infos_list = [
                {},
                {
                    "final_info": {
                        "episode": {
                            "return": torch.tensor([7.0]),
                            "length": torch.tensor([2]),
                        }
                    }
                },
                {},
            ]
            return obs_list, rewards, terminations, truncations, infos_list

    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "env": {
                "train": {
                    "env_type": "realworld",
                    "auto_reset": False,
                    "ignore_terminations": True,
                }
            },
            "actor": {
                "model": {
                    "model_type": "dummy",
                    "num_action_chunks": 3,
                    "action_dim": 2,
                }
            },
        }
    )
    worker.env_list = [FakeRealWorldEnv()]
    worker.use_external_reward_model = False
    worker._timer_metrics = {}

    _, env_info = worker.env_interact_step(torch.zeros(1, 3, 2), stage_id=0)

    torch.testing.assert_close(env_info["return"], torch.tensor([7.0]))
    torch.testing.assert_close(env_info["length"], torch.tensor([2]))


def test_env_worker_bootstrap_rewards_uses_true_terminal_substep():
    from rlinf.data.embodied_io_struct import EnvOutput
    from rlinf.workers.env.env_worker import EnvWorker

    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "env": {"train": {"auto_reset": True}},
            "algorithm": {"bootstrap_type": "standard", "gamma": 0.5},
        }
    )
    worker.env_reward_weight = 1.0
    worker.reward_weight = 1.0
    worker._timer_metrics = {}
    env_output = EnvOutput(
        obs={"state": torch.zeros(2, 2)},
        rewards=torch.zeros(2, 3),
        dones=torch.tensor(
            [[False, True, False], [False, False, False]], dtype=torch.bool
        ),
        terminations=torch.zeros(2, 3, dtype=torch.bool),
        truncations=torch.tensor(
            [[False, True, False], [False, False, False]], dtype=torch.bool
        ),
    )

    adjusted = worker.compute_bootstrap_rewards(
        env_output,
        bootstrap_values=torch.tensor([[4.0], [9.0]]),
        reward_model_output=None,
    )

    torch.testing.assert_close(
        adjusted, torch.tensor([[0.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
    )


def test_env_worker_bootstrap_rewards_rejects_rank_mismatch():
    from rlinf.data.embodied_io_struct import EnvOutput
    from rlinf.workers.env.env_worker import EnvWorker

    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "env": {"train": {"auto_reset": True}},
            "algorithm": {"bootstrap_type": "standard", "gamma": 0.5},
        }
    )
    worker.env_reward_weight = 1.0
    worker.reward_weight = 1.0
    worker._timer_metrics = {}
    env_output = EnvOutput(
        obs={"state": torch.zeros(2, 2)},
        rewards=torch.zeros(2, 3),
        dones=torch.tensor([True, False], dtype=torch.bool),
        terminations=torch.zeros(2, dtype=torch.bool),
        truncations=torch.tensor([True, False], dtype=torch.bool),
    )

    with pytest.raises(ValueError, match="rank"):
        worker.compute_bootstrap_rewards(
            env_output,
            bootstrap_values=torch.tensor([[4.0], [9.0]]),
            reward_model_output=None,
        )


def test_env_worker_compute_bootstrap_rewards_accepts_batched_reward_model_output_for_chunks():
    from rlinf.data.embodied_io_struct import EnvOutput
    from rlinf.workers.env.env_worker import EnvWorker

    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "env": {"train": {"auto_reset": True}},
            "algorithm": {"bootstrap_type": "standard", "gamma": 0.5},
        }
    )
    worker.env_reward_weight = 0.0
    worker.reward_weight = 1.0
    worker._timer_metrics = {}
    env_output = EnvOutput(
        obs={"state": torch.zeros(2, 2)},
        rewards=torch.zeros(2, 3),
        dones=torch.zeros(2, 3, dtype=torch.bool),
        terminations=torch.zeros(2, 3, dtype=torch.bool),
        truncations=torch.zeros(2, 3, dtype=torch.bool),
        env_infos={
            "chunk_valid_step": torch.tensor([[True, True, False], [True, True, True]])
        },
    )

    adjusted = worker.compute_bootstrap_rewards(
        env_output,
        bootstrap_values=None,
        reward_model_output=torch.tensor([[4.0], [9.0]]),
    )

    torch.testing.assert_close(
        adjusted, torch.tensor([[4.0, 4.0, 0.0], [9.0, 9.0, 9.0]])
    )


def test_env_worker_collects_chunk_final_episode_info_from_terminal_substeps():
    from rlinf.workers.env.env_worker import EnvWorker

    worker = object.__new__(EnvWorker)
    env_info = {}
    infos_list = [
        {
            "final_info": {
                "episode": {
                    "return": torch.tensor([1.0, 10.0]),
                    "length": torch.tensor([2, 20]),
                }
            }
        },
        {
            "final_info": {
                "episode": {
                    "return": torch.tensor([3.0, 30.0]),
                    "length": torch.tensor([4, 40]),
                }
            }
        },
    ]
    done_steps = torch.tensor([[True, False], [False, True]], dtype=torch.bool)

    worker._collect_chunk_final_episode_info(env_info, infos_list, done_steps)

    torch.testing.assert_close(env_info["return"], torch.tensor([1.0, 30.0]))
    torch.testing.assert_close(env_info["length"], torch.tensor([2, 40]))


def test_env_worker_reward_env_infos_merges_final_info_per_terminal_env():
    from rlinf.workers.env.env_worker import EnvWorker

    worker = object.__new__(EnvWorker)
    last_infos = {
        "final_info": {
            "episode": {
                "return": torch.tensor([0.0, 0.0]),
                "success": torch.tensor([False, False]),
            }
        }
    }
    infos_list = [
        {
            "final_info": {
                "episode": {
                    "return": torch.tensor([1.0, 10.0]),
                    "success": torch.tensor([True, False]),
                }
            }
        },
        {
            "final_info": {
                "episode": {
                    "return": torch.tensor([3.0, 30.0]),
                    "success": torch.tensor([False, True]),
                }
            }
        },
    ]
    done_steps = torch.tensor([[True, False], [False, True]], dtype=torch.bool)

    env_infos = worker._build_chunk_reward_env_infos(last_infos, infos_list, done_steps)

    torch.testing.assert_close(
        env_infos["final_info"]["episode"]["return"],
        torch.tensor([1.0, 30.0]),
    )
    torch.testing.assert_close(
        env_infos["final_info"]["episode"]["success"],
        torch.tensor([True, True]),
    )


def test_env_worker_reward_env_infos_masks_final_info_when_base_is_missing():
    from rlinf.workers.env.env_worker import EnvWorker

    worker = object.__new__(EnvWorker)
    infos_list = [
        {
            "final_info": {
                "episode": {
                    "return": torch.tensor([1.0, 10.0]),
                    "success": torch.tensor([True, True]),
                }
            }
        }
    ]
    done_steps = torch.tensor([[True], [False]], dtype=torch.bool)

    env_infos = worker._build_chunk_reward_env_infos({}, infos_list, done_steps)

    torch.testing.assert_close(
        env_infos["final_info"]["episode"]["return"],
        torch.tensor([1.0, 0.0]),
    )
    torch.testing.assert_close(
        env_infos["final_info"]["episode"]["success"],
        torch.tensor([True, False]),
    )


def test_env_worker_reward_env_infos_ignores_stale_base_and_masks_new_keys():
    from rlinf.workers.env.env_worker import EnvWorker

    worker = object.__new__(EnvWorker)
    last_infos = {
        "final_info": {
            "episode": {
                "return": torch.tensor([0.0, 99.0]),
            }
        }
    }
    infos_list = [
        {
            "final_info": {
                "episode": {
                    "return": torch.tensor([1.0, 10.0]),
                    "success": torch.tensor([True, True]),
                }
            }
        }
    ]
    done_steps = torch.tensor([[True], [False]], dtype=torch.bool)

    env_infos = worker._build_chunk_reward_env_infos(last_infos, infos_list, done_steps)

    torch.testing.assert_close(
        env_infos["final_info"]["episode"]["return"],
        torch.tensor([1.0, 0.0]),
    )
    torch.testing.assert_close(
        env_infos["final_info"]["episode"]["success"],
        torch.tensor([True, False]),
    )


def test_env_worker_reward_env_infos_rejects_mismatched_initial_final_info_shape():
    from rlinf.workers.env.env_worker import EnvWorker

    worker = object.__new__(EnvWorker)
    infos_list = [
        {
            "final_info": {
                "episode": {
                    "success": torch.tensor([True]),
                }
            }
        }
    ]
    done_steps = torch.tensor([[True], [False]], dtype=torch.bool)

    with pytest.raises(ValueError, match="final_info.*shape"):
        worker._build_chunk_reward_env_infos({}, infos_list, done_steps)


def test_env_worker_masks_invalid_tail_actions_from_realworld_chunk_valid_step():
    from rlinf.data.embodied_io_struct import EnvOutput, RolloutResult
    from rlinf.workers.env.env_worker import EnvWorker

    worker = object.__new__(EnvWorker)
    rollout_action = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    rollout_result = RolloutResult(
        actions=rollout_action.clone(),
        forward_inputs={
            "action": rollout_action.clone(),
            "model_action": rollout_action.clone(),
            "states": torch.ones(1, 2),
        },
    )
    env_output = EnvOutput(
        obs={"state": torch.zeros(1, 2)},
        dones=torch.tensor([[False, True, False]], dtype=torch.bool),
        env_infos={"chunk_valid_step": torch.tensor([[True, True, False]])},
    )

    masked_result = worker._mask_invalid_chunk_actions(rollout_result, env_output)

    expected_action = torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.0, 0.0]])
    torch.testing.assert_close(masked_result.actions, expected_action)
    torch.testing.assert_close(masked_result.forward_inputs["action"], expected_action)
    torch.testing.assert_close(
        masked_result.forward_inputs["model_action"], expected_action
    )
    torch.testing.assert_close(masked_result.forward_inputs["states"], torch.ones(1, 2))


def test_env_worker_masks_intervention_overwrite_from_invalid_tail():
    from rlinf.data.embodied_io_struct import (
        ChunkStepResult,
        EmbodiedRolloutResult,
        EnvOutput,
    )
    from rlinf.workers.env.env_worker import EnvWorker

    worker = object.__new__(EnvWorker)
    rollout_result = EmbodiedRolloutResult(max_episode_length=3)
    rollout_result.append_step_result(
        ChunkStepResult(
            actions=torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
            forward_inputs={
                "action": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
                "model_action": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
            },
        )
    )
    env_output = EnvOutput(
        obs={"state": torch.zeros(1, 2)},
        intervene_actions=torch.tensor([[10.0, 20.0, 30.0, 40.0, 99.0, 99.0]]),
        intervene_flags=torch.tensor([[False, True, True]], dtype=torch.bool),
        env_infos={"chunk_valid_step": torch.tensor([[True, True, False]])},
    )

    worker._update_last_actions_from_env_output(rollout_result, env_output)

    expected_action = torch.tensor([[1.0, 2.0, 30.0, 40.0, 0.0, 0.0]])
    torch.testing.assert_close(rollout_result.actions[-1], expected_action)
    torch.testing.assert_close(
        rollout_result.forward_inputs[-1]["action"], expected_action
    )
    assert "model_action" not in rollout_result.forward_inputs[-1]
    torch.testing.assert_close(
        rollout_result.intervene_flags[-1],
        torch.tensor([[False, False, True, True, False, False]]),
    )


def test_env_worker_masks_last_stored_actions_from_realworld_chunk_valid_step():
    from rlinf.data.embodied_io_struct import (
        ChunkStepResult,
        EmbodiedRolloutResult,
        EnvOutput,
    )
    from rlinf.workers.env.env_worker import EnvWorker

    worker = object.__new__(EnvWorker)
    rollout_action = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    rollout_result = EmbodiedRolloutResult(max_episode_length=3)
    rollout_result.append_step_result(
        ChunkStepResult(
            actions=rollout_action.clone(),
            forward_inputs={
                "action": rollout_action.clone(),
                "model_action": rollout_action.clone(),
            },
        )
    )
    env_output = EnvOutput(
        obs={"state": torch.zeros(1, 2)},
        dones=torch.tensor([[False, True, False]], dtype=torch.bool),
        env_infos={"chunk_valid_step": torch.tensor([[True, True, False]])},
    )

    worker._mask_last_invalid_chunk_actions(rollout_result, env_output)

    expected_action = torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.0, 0.0]])
    torch.testing.assert_close(rollout_result.actions[-1], expected_action)
    torch.testing.assert_close(
        rollout_result.forward_inputs[-1]["action"], expected_action
    )
    torch.testing.assert_close(
        rollout_result.forward_inputs[-1]["model_action"], expected_action
    )
    torch.testing.assert_close(
        rollout_result.intervene_flags[-1],
        torch.tensor([[False, False, False, False, False, False]]),
    )


def test_env_worker_realworld_invalid_chunk_tail_sets_loss_mask():
    from rlinf.data.embodied_io_struct import (
        ChunkStepResult,
        EmbodiedRolloutResult,
        EnvOutput,
        convert_trajectories_to_batch,
    )
    from rlinf.workers.env.env_worker import EnvWorker

    worker = object.__new__(EnvWorker)
    rollout_result = EmbodiedRolloutResult(max_episode_length=3)
    rollout_result.append_step_result(
        ChunkStepResult(
            actions=torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
            rewards=torch.tensor([[1.0, 2.0, 0.0]]),
            dones=torch.tensor([[False, True, False]], dtype=torch.bool),
        )
    )
    env_output = EnvOutput(
        obs={"state": torch.zeros(1, 2)},
        dones=torch.tensor([[False, True, False]], dtype=torch.bool),
        env_infos={"chunk_valid_step": torch.tensor([[True, True, False]])},
    )

    worker._mask_last_invalid_chunk_actions(rollout_result, env_output)

    batch = convert_trajectories_to_batch([rollout_result.to_trajectory()])

    torch.testing.assert_close(
        batch["loss_mask"], torch.tensor([[[True, True, False]]])
    )


def test_env_worker_reward_input_collapses_chunk_dones_with_any():
    from rlinf.data.embodied_io_struct import EnvOutput
    from rlinf.workers.env.env_worker import EnvWorker

    worker = object.__new__(EnvWorker)
    worker.reward_mode = "per_step"
    worker.env_infos_reward_keys = ("final_info",)
    worker._timer_metrics = {}
    captured = {}
    worker._select_reward_env_infos = lambda env_infos: env_infos
    worker.send_reward_input = lambda send_channel, reward_input: captured.update(
        reward_input
    )
    worker.recv_reward_results = lambda recv_channel: torch.zeros(2)
    env_output = EnvOutput(
        obs={"state": torch.zeros(2, 2)},
        dones=torch.tensor(
            [[False, True, False], [False, False, False]], dtype=torch.bool
        ),
    )

    worker.get_reward_model_output(
        env_output,
        send_channel=None,
        recv_channel=None,
    )

    torch.testing.assert_close(captured["dones"], torch.tensor([True, False]))


class _FakeDemoBuffer:
    def __init__(self, trajectories: list[SimpleNamespace]):
        self._trajectory_id_list = list(range(len(trajectories)))
        self._trajectory_index = {
            trajectory_id: {"model_weights_id": "test"}
            for trajectory_id in self._trajectory_id_list
        }
        self._trajectories = trajectories

    def __len__(self):
        return len(self._trajectory_id_list)

    def is_ready(self, min_size: int) -> bool:
        return len(self) >= min_size

    def _load_trajectory(self, trajectory_id: int, model_weights_id: str):
        return self._trajectories[trajectory_id]


def _demo_trajectory(curr_obs: dict, next_obs: dict, length: int = 1, **overrides):
    fields = {
        "curr_obs": curr_obs,
        "next_obs": next_obs,
        "actions": torch.zeros(length, 1, 1),
        "rewards": torch.zeros(length, 1, 1),
        "terminations": torch.zeros(length, 1, 1, dtype=torch.bool),
        "truncations": torch.zeros(length, 1, 1, dtype=torch.bool),
        "dones": torch.zeros(length, 1, 1, dtype=torch.bool),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _sac_policy_for_demo_validation(model_cfg: dict, *, world_size: int = 1):
    from rlinf.workers.actor import sac_demo_buffer_utils

    policy = SimpleNamespace()
    policy.cfg = OmegaConf.create({"actor": {"model": model_cfg}})
    policy._rank = 0
    policy._world_size = world_size
    policy._validate_loaded_demo_buffer = lambda load_path, min_demo_buffer_size: (
        sac_demo_buffer_utils.validate_loaded_demo_buffer(
            policy.demo_buffer,
            load_path=load_path,
            min_demo_buffer_size=min_demo_buffer_size,
            rank=policy._rank,
            world_size=policy._world_size,
            model_cfg=policy.cfg.actor.model,
            load_mode=policy.cfg.get("algorithm", {})
            .get("demo_buffer", {})
            .get("load_mode", "shard"),
        )
    )
    policy._demo_buffer_load_kwargs = lambda: (
        sac_demo_buffer_utils.demo_buffer_load_kwargs(
            policy.cfg.algorithm.demo_buffer,
            rank=policy._rank,
            world_size=policy._world_size,
        )
    )
    policy._validate_sac_model_type = lambda: __import__(
        "rlinf.config", fromlist=["validate_embodied_sac_model_type"]
    ).validate_embodied_sac_model_type(
        policy.cfg.actor.model, policy.cfg.get("algorithm", None)
    )
    return policy


def _assert_demo_validation_error(
    *,
    model_cfg: dict,
    curr_obs: dict,
    match: str,
    next_obs: dict | None = None,
    **trajectory_overrides,
):
    policy = _sac_policy_for_demo_validation(model_cfg)
    trajectory = _demo_trajectory(
        curr_obs,
        curr_obs if next_obs is None else next_obs,
        **trajectory_overrides,
    )
    policy.demo_buffer = _FakeDemoBuffer([trajectory])

    with pytest.raises(ValueError, match=match):
        policy._validate_loaded_demo_buffer("/tmp/demo", min_demo_buffer_size=1)


def test_sac_demo_required_obs_keys_allow_state_only_flow_policy():
    from rlinf.workers.actor import sac_demo_buffer_utils

    policy = _sac_policy_for_demo_validation(
        {
            "model_type": "flow_policy",
            "input_type": "state",
            "image_num": 1,
        }
    )

    assert sac_demo_buffer_utils.required_demo_obs_keys(policy.cfg.actor.model) == {
        "states"
    }


def test_sac_demo_validation_checks_all_loaded_trajectories():
    policy = _sac_policy_for_demo_validation(
        {
            "model_type": "cnn_policy",
            "image_num": 1,
        }
    )
    image_obs = {
        "states": torch.zeros(1, 1, 2),
        "main_images": torch.zeros(1, 1, 2, 2, 3, dtype=torch.uint8),
    }
    state_only_obs = {"states": torch.zeros(1, 1, 2)}
    policy.demo_buffer = _FakeDemoBuffer(
        [
            _demo_trajectory(image_obs, image_obs),
            _demo_trajectory(state_only_obs, state_only_obs),
        ]
    )

    with pytest.raises(ValueError, match="trajectory_id=1.*main_images"):
        policy._validate_loaded_demo_buffer("/tmp/demo", min_demo_buffer_size=1)


@pytest.mark.parametrize(
    ("model_cfg", "curr_obs", "match", "trajectory_overrides"),
    [
        (
            {"model_type": "cnn_policy", "image_num": 3},
            {
                "states": torch.zeros(1, 1, 2),
                "main_images": torch.zeros(1, 1, 2, 2, 3, dtype=torch.uint8),
                "extra_view_images": torch.zeros(1, 1, 1, 2, 2, 3, dtype=torch.uint8),
            },
            "extra_view_images.*image_num",
            {},
        ),
        (
            {"model_type": "cnn_policy", "image_num": 0},
            {
                "states": torch.zeros(1, 1, 2),
                "main_images": torch.zeros(1, 1, 2, 2, 3, dtype=torch.uint8),
            },
            "image_num",
            {},
        ),
        (
            {"model_type": "cnn_policy", "image_num": 1},
            {
                "states": torch.zeros(1, 1, 2),
                "main_images": torch.zeros(1, 1, 3, dtype=torch.uint8),
            },
            "main_images.*shape",
            {},
        ),
        (
            {"model_type": "cnn_policy", "image_num": 3},
            {
                "states": torch.zeros(1, 1, 2),
                "main_images": torch.zeros(1, 1, 2, 2, 3, dtype=torch.uint8),
                "extra_view_images": torch.zeros(1, 1, 2, dtype=torch.uint8),
            },
            "extra_view_images.*shape",
            {},
        ),
        (
            {"model_type": "mlp_policy", "action_dim": 2, "obs_dim": 2},
            {"states": torch.zeros(1, 1, 2)},
            "actions.*action_dim",
            {"actions": torch.zeros(1, 1, 3)},
        ),
        (
            {"model_type": "mlp_policy", "action_dim": 1, "obs_dim": 2},
            {"states": torch.zeros(1, 1, 3)},
            "states.*obs_dim",
            {},
        ),
        (
            {"model_type": "mlp_policy", "action_dim": 1, "obs_dim": 2},
            {"states": torch.zeros(1, 2, 2)},
            r"\[T, B\]",
            {"rewards": torch.zeros(1)},
        ),
    ],
)
def test_sac_demo_validation_rejects_invalid_schema(
    model_cfg,
    curr_obs,
    match,
    trajectory_overrides,
):
    _assert_demo_validation_error(
        model_cfg=model_cfg,
        curr_obs=curr_obs,
        match=match,
        **trajectory_overrides,
    )


def test_sac_worker_rejects_openvla_for_sac_training():
    policy = _sac_policy_for_demo_validation({"model_type": "openvla"})

    with pytest.raises(ValueError, match="does not support SAC"):
        policy._validate_sac_model_type()


def test_sac_worker_allows_openpi_dsrl_for_sac_training():
    policy = _sac_policy_for_demo_validation(
        {"model_type": "openpi", "openpi": {"use_dsrl": True}}
    )

    policy._validate_sac_model_type()


@pytest.mark.parametrize(
    ("model_cfg", "algorithm_cfg", "match"),
    [
        ({"model_type": "openvla"}, None, "does not support SAC"),
        (
            {"model_type": "openpi", "openpi": {"use_dsrl": True}},
            {"q_head_type": "crossq"},
            "CrossQ",
        ),
        (
            {
                "model_type": "mlp_policy",
                "add_q_head": True,
                "q_head_type": "default",
            },
            {"q_head_type": "crossq"},
            "q_head_type",
        ),
        (
            {
                "model_type": "mlp_policy",
                "add_q_head": True,
                "q_head_type": "croosq",
            },
            {"q_head_type": "croosq"},
            "q_head_type",
        ),
        ({"model_type": "openvla_oft"}, None, "does not support SAC"),
        ({"model_type": "openpi", "openpi": {"use_dsrl": False}}, None, "use_dsrl"),
    ],
)
def test_config_validation_rejects_invalid_embodied_sac_model(
    model_cfg,
    algorithm_cfg,
    match,
):
    from rlinf.config import validate_embodied_sac_model_type

    args = [OmegaConf.create(model_cfg)]
    if algorithm_cfg is not None:
        args.append(OmegaConf.create(algorithm_cfg))
    with pytest.raises(ValueError, match=match):
        validate_embodied_sac_model_type(*args)


def test_config_validation_allows_openpi_dsrl_for_embodied_sac():
    from rlinf.config import validate_embodied_sac_model_type

    validate_embodied_sac_model_type(
        OmegaConf.create({"model_type": "openpi", "openpi": {"use_dsrl": True}})
    )


def test_sac_worker_validation_uses_algorithm_q_head_type():
    policy = _sac_policy_for_demo_validation(
        {
            "model_type": "mlp_policy",
            "add_q_head": True,
            "q_head_type": "default",
        }
    )
    policy.cfg.algorithm = OmegaConf.create({"q_head_type": "crossq"})

    with pytest.raises(ValueError, match="q_head_type"):
        policy._validate_sac_model_type()


def test_config_validation_allows_openpi_without_dsrl_for_eval_only_embodied_sac():
    from rlinf.config import validate_embodied_sac_cfg

    cfg = OmegaConf.create(
        {
            "runner": {"only_eval": True},
            "actor": {"model": {"model_type": "openpi", "openpi": {"use_dsrl": False}}},
            "algorithm": {"loss_type": "embodied_sac", "adv_type": "embodied_sac"},
        }
    )

    validate_embodied_sac_cfg(cfg)


def test_validate_cfg_rejects_invalid_sac_model_before_cluster_init():
    from rlinf.config import validate_cfg

    cfg = OmegaConf.create(
        {
            "runner": {
                "task_type": "embodied",
                "logger": {"log_path": "/tmp", "experiment_name": "test"},
            },
            "cluster": {},
            "actor": {"model": {"model_type": "openvla"}},
            "algorithm": {"loss_type": "embodied_sac", "adv_type": "embodied_sac"},
        }
    )

    with patch("rlinf.config.Cluster", side_effect=AssertionError("Cluster called")):
        with pytest.raises(ValueError, match="does not support SAC"):
            validate_cfg(cfg)


def test_sac_demo_validation_schema_error_does_not_suggest_replicate():
    policy = _sac_policy_for_demo_validation(
        {
            "model_type": "cnn_policy",
            "image_num": 1,
        }
    )
    state_only_obs = {"states": torch.zeros(1, 1, 2)}
    policy.demo_buffer = _FakeDemoBuffer(
        [_demo_trajectory(state_only_obs, state_only_obs)]
    )

    with pytest.raises(ValueError) as exc_info:
        policy._validate_loaded_demo_buffer("/tmp/demo", min_demo_buffer_size=1)

    message = str(exc_info.value)
    assert "main_images" in message
    assert "available_obs_keys" in message
    assert "load_mode: replicate" not in message


def test_sac_demo_buffer_replicate_load_mode_disables_distributed_sharding():
    policy = _sac_policy_for_demo_validation(
        {
            "model_type": "mlp_policy",
        },
        world_size=4,
    )
    policy.cfg.algorithm = OmegaConf.create({"demo_buffer": {"load_mode": "replicate"}})

    assert policy._demo_buffer_load_kwargs() == {"is_distributed": False}


def test_sac_demo_buffer_rejects_invalid_load_mode():
    from rlinf.config import validate_demo_buffer_load_mode

    with pytest.raises(ValueError, match="load_mode"):
        validate_demo_buffer_load_mode(OmegaConf.create({"load_mode": "bad"}))


def test_sac_demo_validation_rejects_empty_loaded_shard():
    policy = _sac_policy_for_demo_validation({"model_type": "mlp_policy"})
    policy.demo_buffer = _FakeDemoBuffer([])

    with pytest.raises(ValueError, match="empty.*load_mode: replicate"):
        policy._validate_loaded_demo_buffer("/tmp/demo", min_demo_buffer_size=0)


def test_sac_demo_validation_rejects_loaded_shard_below_min_size():
    policy = _sac_policy_for_demo_validation({"model_type": "mlp_policy"})
    trajectory = _demo_trajectory(
        {"states": torch.zeros(1, 1, 2)},
        {"states": torch.zeros(1, 1, 2)},
    )
    policy.demo_buffer = _FakeDemoBuffer([trajectory])

    with pytest.raises(ValueError, match="smaller than min_buffer_size"):
        policy._validate_loaded_demo_buffer("/tmp/demo", min_demo_buffer_size=2)


def test_sac_demo_validation_rejects_missing_required_trajectory_fields():
    policy = _sac_policy_for_demo_validation({"model_type": "mlp_policy"})
    policy.demo_buffer = _FakeDemoBuffer(
        [
            SimpleNamespace(
                curr_obs={"states": torch.zeros(1, 1, 2)},
                next_obs={"states": torch.zeros(1, 1, 2)},
                rewards=torch.zeros(1, 1, 1),
                terminations=torch.zeros(1, 1, 1, dtype=torch.bool),
                truncations=torch.zeros(1, 1, 1, dtype=torch.bool),
                dones=torch.zeros(1, 1, 1, dtype=torch.bool),
            )
        ]
    )

    with pytest.raises(ValueError, match="actions"):
        policy._validate_loaded_demo_buffer("/tmp/demo", min_demo_buffer_size=1)


def test_sac_demo_validator_accepts_epoch_padded_dones():
    """dones/terminations/truncations may carry one extra entry per rollout
    epoch; the validator must accept any divisor-of-T padding."""
    policy = _sac_policy_for_demo_validation({"model_type": "mlp_policy"})
    # actions shape: [T=4, B=1, action_dim=2]; dones shape: [T+epoch=8, B=1, 1]
    # (epoch_count=4, epoch_len=1: matches _flatten_trajectory's reshape rule).
    policy.demo_buffer = _FakeDemoBuffer(
        [
            SimpleNamespace(
                curr_obs={"states": torch.zeros(4, 1, 2)},
                next_obs={"states": torch.zeros(4, 1, 2)},
                actions=torch.zeros(4, 1, 2),
                rewards=torch.zeros(4, 1, 1),
                terminations=torch.zeros(8, 1, 1, dtype=torch.bool),
                truncations=torch.zeros(8, 1, 1, dtype=torch.bool),
                dones=torch.zeros(8, 1, 1, dtype=torch.bool),
            )
        ]
    )

    policy._validate_loaded_demo_buffer("/tmp/demo", min_demo_buffer_size=1)


def test_sac_demo_validator_rejects_non_divisor_epoch_padding():
    """dones first-dim that is neither T nor a divisor-multiple of T must fail."""
    policy = _sac_policy_for_demo_validation({"model_type": "mlp_policy"})
    policy.demo_buffer = _FakeDemoBuffer(
        [
            SimpleNamespace(
                curr_obs={"states": torch.zeros(4, 1, 2)},
                next_obs={"states": torch.zeros(4, 1, 2)},
                actions=torch.zeros(4, 1, 2),
                rewards=torch.zeros(4, 1, 1),
                terminations=torch.zeros(9, 1, 1, dtype=torch.bool),
                truncations=torch.zeros(9, 1, 1, dtype=torch.bool),
                dones=torch.zeros(9, 1, 1, dtype=torch.bool),
            )
        ]
    )

    with pytest.raises(ValueError, match="inconsistent"):
        policy._validate_loaded_demo_buffer("/tmp/demo", min_demo_buffer_size=1)


def test_sac_apply_chunk_mask_matches_mean_when_weights_none():
    from rlinf.workers.actor.sac_demo_buffer_utils import (
        apply_chunk_mask,
        chunk_sample_weights,
    )

    per_sample = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert chunk_sample_weights({}, torch.float32) is None
    torch.testing.assert_close(
        apply_chunk_mask(per_sample, None), per_sample.mean()
    )


def test_sac_apply_chunk_mask_excludes_zero_weighted_samples():
    """An all-zero mask gives 0 loss with no NaN, and partial mask correctly
    averages over only the valid samples and feature axis."""
    from rlinf.workers.actor.sac_demo_buffer_utils import apply_chunk_mask

    per_sample = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    weights_all_zero = torch.zeros(2, 1)
    out = apply_chunk_mask(per_sample, weights_all_zero)
    assert torch.isfinite(out).all() and out.item() == 0.0

    weights_first_only = torch.tensor([[1.0], [0.0]])
    out = apply_chunk_mask(per_sample, weights_first_only)
    torch.testing.assert_close(out, torch.tensor(2.0))


def test_lerobot_trajectory_emits_all_true_loss_mask():
    """LeRobot demos must carry a loss_mask so concat_batch with a chunked
    replay batch does not silently drop the SAC chunk-mask key."""
    from rlinf.utils.nested_dict_process import concat_batch

    frames = [
        _frame_with_next(0),
        _frame_with_next(1, done=True),
    ]
    trajectory = lerobot_episode_to_trajectory(frames)

    assert trajectory.loss_mask is not None
    assert trajectory.loss_mask.dtype == torch.bool
    assert trajectory.loss_mask.shape == trajectory.dones.shape
    assert trajectory.loss_mask.all()

    # Smoke-check the surviving-key behaviour: concat with another dict that
    # also has loss_mask keeps it (i.e. mixing demo+replay batches retains
    # the mask used by SAC loss reduction).
    demo_batch = {
        "loss_mask": trajectory.loss_mask.reshape(-1, *trajectory.loss_mask.shape[2:]),
    }
    replay_batch = {
        "loss_mask": torch.ones(3, 1, dtype=torch.bool),
    }
    out = concat_batch(replay_batch, demo_batch)
    assert "loss_mask" in out
    assert out["loss_mask"].shape == (5, 1)


def test_sac_chunk_sample_weights_requires_all_chunk_substeps_valid():
    from rlinf.workers.actor.sac_demo_buffer_utils import chunk_sample_weights

    batch = {
        "loss_mask": torch.tensor(
            [[True, True, True], [True, True, False]], dtype=torch.bool
        )
    }
    weights = chunk_sample_weights(batch, torch.float32)
    torch.testing.assert_close(weights, torch.tensor([[1.0], [0.0]]))
