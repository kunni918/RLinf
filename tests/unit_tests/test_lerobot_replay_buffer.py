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
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

# T11: Pillow is a required test-time dep for the image-decode coverage. Import
# directly so missing Pillow becomes a collection ERROR (loud) rather than a
# SKIP outcome (silent). The CI base image must ship Pillow.
import PIL.Image  # noqa: F401

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
        # Obs-only target frame: mirror the action frame's positive terminal
        # flag so MAJ-11 source/target agreement does not raise. The legacy
        # silent-source-wins behaviour was rejected this round; producers
        # must keep terminal flags consistent across source/target frames.
        _frame(2, done=True),
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
    """Fake demo buffer exposing the new public API.

    Mirrors :class:`TrajectoryReplayBuffer`'s public
    ``iter_trajectory_metadata()`` / ``load_trajectory()`` contract that
    :func:`validate_loaded_demo_buffer` now relies on (consensus MAJ-8).
    """

    def __init__(self, trajectories: list[SimpleNamespace]):
        self._trajectories = trajectories
        # Build a tiny metadata index that mimics the production buffer.
        self._index = [
            (trajectory_id, "test", 1)
            for trajectory_id in range(len(trajectories))
        ]

    def __len__(self):
        return len(self._trajectories)

    def is_ready(self, min_size: int) -> bool:
        return len(self) >= min_size

    def iter_trajectory_metadata(self):
        """Yield ``(trajectory_id, model_weights_id, num_samples)``."""
        for entry in self._index:
            yield entry

    def load_trajectory(self, trajectory_id: int, model_weights_id: str):
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
    loss, valid_count = apply_chunk_mask(per_sample, None)
    torch.testing.assert_close(loss, per_sample.mean())
    assert valid_count == per_sample.shape[0]


def test_sac_apply_chunk_mask_excludes_zero_weighted_samples():
    """An all-zero mask gives 0 loss with no NaN, and partial mask correctly
    averages over only the valid samples and feature axis. The function now
    also returns a ``valid_count`` so callers can skip optimizer/scheduler
    advance when zero (MAJ-4)."""
    from rlinf.workers.actor.sac_demo_buffer_utils import apply_chunk_mask

    per_sample = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    weights_all_zero = torch.zeros(2, 1)
    out, valid_count = apply_chunk_mask(per_sample, weights_all_zero)
    assert torch.isfinite(out).all() and out.item() == 0.0
    assert valid_count == 0

    weights_first_only = torch.tensor([[1.0], [0.0]])
    out, valid_count = apply_chunk_mask(per_sample, weights_first_only)
    torch.testing.assert_close(out, torch.tensor(2.0))
    assert valid_count == 1


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


# ---------------------------------------------------------------------------
# T1-T11 — Round 1 consensus gap-closing tests.
# Each test is annotated with the CRIT/MAJ consensus item it exercises.
# ---------------------------------------------------------------------------


def test_lerobot_demo_and_chunked_rollout_loss_mask_concat_roundtrip():
    """T1 — Multi-chunk ``loss_mask`` roundtrip (CRIT-1 (a), (b), (c)).

    The original C2 bug let the LeRobot demo emit a ``[T, 1, 1]`` mask while a
    chunked-rollout emitted ``[T, B, num_chunks]``: ``concat_batch`` would
    raise (or silently drop) when the trailing chunk dims disagreed. After the
    F1 fix the LeRobot mask shape derives from ``actions`` so it ends up
    ``[T, 1, 1]`` (equivalent to ``num_action_chunks=1``) and concatenates
    cleanly with a single-chunk rollout batch.

    The test exercises the actual ``lerobot_episode_to_trajectory`` →
    ``convert_trajectories_to_batch`` → ``concat_batch`` pipeline so it would
    have caught the original C2 bug.
    """
    from rlinf.data.embodied_io_struct import convert_trajectories_to_batch
    from rlinf.utils.nested_dict_process import concat_batch

    # Build a LeRobot demo trajectory with T=2 frames.  Its mask shape ends up
    # ``[T=2, B=1, 1]`` — see the comment in
    # ``lerobot_episode_to_trajectory`` near the ``loss_mask_tensor`` build.
    demo_frames = [
        _frame_with_next(0),
        _frame_with_next(1, done=True, terminated=True),
    ]
    demo_trajectory = lerobot_episode_to_trajectory(demo_frames)
    assert demo_trajectory.loss_mask is not None
    assert demo_trajectory.loss_mask.shape == (2, 1, 1)

    # Build a chunked-rollout trajectory whose loss_mask is shaped the same as
    # the demo (``num_action_chunks == 1``) so that both producers agree on
    # the trailing dim.  ``convert_trajectories_to_batch`` stacks each side on
    # the batch dim and ``concat_batch`` joins them along dim=0.
    rollout_trajectory = SimpleNamespace(
        max_episode_length=4,
        actions=torch.zeros(4, 1, 1),
        rewards=torch.zeros(4, 1, 1),
        terminations=torch.zeros(4, 1, 1, dtype=torch.bool),
        truncations=torch.zeros(4, 1, 1, dtype=torch.bool),
        dones=torch.zeros(4, 1, 1, dtype=torch.bool),
        loss_mask=torch.ones(4, 1, 1, dtype=torch.bool),
        intervene_flags=torch.zeros(4, 1, 1, dtype=torch.bool),
        forward_inputs={},
        curr_obs={"states": torch.zeros(4, 1, 2)},
        next_obs={"states": torch.zeros(4, 1, 2)},
        prev_logprobs=None,
        prev_values=None,
        versions=None,
        model_weights_id="rollout",
    )
    # Reuse the real ``Trajectory`` dataclass instead of a SimpleNamespace so
    # ``convert_trajectories_to_batch`` can iterate ``__dataclass_fields__``.
    from rlinf.data.embodied_io_struct import Trajectory

    rollout_real = Trajectory(
        max_episode_length=4,
        actions=rollout_trajectory.actions,
        rewards=rollout_trajectory.rewards,
        terminations=rollout_trajectory.terminations,
        truncations=rollout_trajectory.truncations,
        dones=rollout_trajectory.dones,
        loss_mask=rollout_trajectory.loss_mask,
        intervene_flags=rollout_trajectory.intervene_flags,
        forward_inputs={},
        curr_obs=rollout_trajectory.curr_obs,
        next_obs=rollout_trajectory.next_obs,
        model_weights_id="rollout",
    )

    demo_batch = convert_trajectories_to_batch([demo_trajectory])
    rollout_batch = convert_trajectories_to_batch([rollout_real])

    assert demo_batch["loss_mask"].shape == (2, 1, 1)
    assert rollout_batch["loss_mask"].shape == (4, 1, 1)

    combined = concat_batch(rollout_batch, demo_batch)
    assert "loss_mask" in combined
    # 4 (rollout) + 2 (demo) along dim=0, trailing dims preserved.
    assert combined["loss_mask"].shape == (6, 1, 1)


@pytest.mark.parametrize("num_action_chunks", [2, 5, 10])
def test_lerobot_demo_loss_mask_matches_multi_chunk_rollout(num_action_chunks):
    """T1-multi — Demo + multi-chunk rollout concat under CRIT-R2-1 fix.

    Round 1 missed the multi-chunk path: the LeRobot mask was hardcoded
    ``[T, 1, 1]`` while real chunked rollouts emit ``[T, B, num_chunks]``.
    The R2 fix exposes ``num_action_chunks`` on the LeRobot conversion API
    so the demo mask carries a matching trailing dim. This test fails if
    the fix regresses.

    Also asserts that every chunk position is True (per consensus: LeRobot
    frames are 1-action-per-step; no chunk position is "padded") and that
    the SAC ``chunk_sample_weights`` reduction yields all-ones — without
    this, a future regression that emits the right shape but the wrong
    value would zero every demo sample's loss and silently break training.
    """
    from rlinf.data.embodied_io_struct import (
        Trajectory,
        convert_trajectories_to_batch,
    )
    from rlinf.utils.nested_dict_process import concat_batch
    from rlinf.workers.actor.sac_demo_buffer_utils import chunk_sample_weights

    demo_frames = [
        _frame_with_next(0),
        _frame_with_next(1, done=True, terminated=True),
    ]
    demo_trajectory = lerobot_episode_to_trajectory(
        demo_frames, num_action_chunks=num_action_chunks
    )
    assert demo_trajectory.loss_mask.shape == (2, 1, num_action_chunks)
    assert demo_trajectory.loss_mask.all(), (
        "Every chunk position in a LeRobot demo must be True; otherwise "
        "chunk_sample_weights would zero the sample and the demo would "
        "silently drop out of SAC training."
    )

    rollout = Trajectory(
        max_episode_length=3,
        actions=torch.zeros(3, 1, 1),
        rewards=torch.zeros(3, 1, 1),
        terminations=torch.zeros(3, 1, 1, dtype=torch.bool),
        truncations=torch.zeros(3, 1, 1, dtype=torch.bool),
        dones=torch.zeros(3, 1, 1, dtype=torch.bool),
        loss_mask=torch.ones(3, 1, num_action_chunks, dtype=torch.bool),
        intervene_flags=torch.zeros(3, 1, 1, dtype=torch.bool),
        forward_inputs={},
        curr_obs={"states": torch.zeros(3, 1, 2)},
        next_obs={"states": torch.zeros(3, 1, 2)},
        model_weights_id="rollout",
    )

    demo_batch = convert_trajectories_to_batch([demo_trajectory])
    rollout_batch = convert_trajectories_to_batch([rollout])
    combined = concat_batch(rollout_batch, demo_batch)
    assert combined["loss_mask"].shape == (5, 1, num_action_chunks)
    # Validate that the SAC consumer of loss_mask sees all-valid for the
    # demo slice (last 2 rows in batch dim 0).
    sample_weights = chunk_sample_weights(
        {"loss_mask": combined["loss_mask"]}, torch.float32
    )
    assert sample_weights is not None
    assert sample_weights[-2:].all(), (
        "chunk_sample_weights must mark every demo sample as valid; a "
        "shape-correct but value-wrong loss_mask would silently zero them."
    )


def test_lerobot_demo_loss_mask_fills_missing_in_rollout():
    """CRIT-R2-3 — Demo `loss_mask` + rollout without `loss_mask` concats.

    Old SAC checkpoints (and non-RealWorld rollouts in general) don't emit
    ``loss_mask``. The Round 1 strict ``concat_batch`` would raise when a
    LeRobot demo (always carries ``loss_mask``) was mixed with such a
    rollout. The R2 fix auto-fills the missing side with all-True.
    """
    from rlinf.data.embodied_io_struct import (
        Trajectory,
        convert_trajectories_to_batch,
    )
    from rlinf.utils.nested_dict_process import concat_batch

    demo_frames = [
        _frame_with_next(0),
        _frame_with_next(1, done=True, terminated=True),
    ]
    demo_trajectory = lerobot_episode_to_trajectory(demo_frames)
    rollout_no_mask = Trajectory(
        max_episode_length=3,
        actions=torch.zeros(3, 1, 1),
        rewards=torch.zeros(3, 1, 1),
        terminations=torch.zeros(3, 1, 1, dtype=torch.bool),
        truncations=torch.zeros(3, 1, 1, dtype=torch.bool),
        dones=torch.zeros(3, 1, 1, dtype=torch.bool),
        loss_mask=None,  # the legacy / non-RealWorld case
        intervene_flags=torch.zeros(3, 1, 1, dtype=torch.bool),
        forward_inputs={},
        curr_obs={"states": torch.zeros(3, 1, 2)},
        next_obs={"states": torch.zeros(3, 1, 2)},
        model_weights_id="rollout-legacy",
    )

    demo_batch = convert_trajectories_to_batch([demo_trajectory])
    rollout_batch = convert_trajectories_to_batch([rollout_no_mask])
    assert "loss_mask" in demo_batch
    assert "loss_mask" not in rollout_batch
    combined = concat_batch(rollout_batch, demo_batch)
    # The missing-rollout-side mask was auto-filled to all-True.
    assert combined["loss_mask"].shape == (5, 1, 1)
    assert combined["loss_mask"].all()


def test_lerobot_episode_rejects_source_target_terminal_disagreement():
    """MAJ-11 regression (A3 sidestepped this branch in Round 1).

    If a LeRobot action source frame says ``done=True`` but the
    observation-only target frame says ``done=False``, that's a contract
    violation. The converter must raise instead of silently choosing one
    side. This test fails if the rejection branch regresses.
    """
    bad_frames = [
        _frame(0, done=False),
        # Action frame at index 1 implies terminal; target frame
        # explicitly contradicts via done=False.
        _frame(1, done=True),
        {
            "episode_index": 0,
            "frame_index": 2,
            "state": np.array([2.0, 2.5], dtype=np.float32),
            "done": np.array([False], dtype=bool),
            "terminated": np.array([False], dtype=bool),
        },
    ]
    with pytest.raises(ValueError, match="disagree on 'done'"):
        lerobot_episode_to_trajectory(bad_frames)


def _make_env_worker_for_chunk_test(*, ignore_terminations: bool):
    """Helper: build an EnvWorker stub for T2 mid-chunk truncation tests."""
    from rlinf.workers.env.env_worker import EnvWorker

    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "env": {
                "train": {
                    "env_type": "realworld",
                    "auto_reset": False,
                    "ignore_terminations": ignore_terminations,
                }
            },
            "actor": {
                "model": {
                    "model_type": "dummy",
                    "num_action_chunks": 4,
                    "action_dim": 2,
                }
            },
        }
    )
    worker.use_external_reward_model = False
    worker._timer_metrics = {}
    return worker


def test_env_worker_mid_chunk_truncation_collects_episode_metrics_from_natural_substep():
    """T2 — Mid-chunk truncation under ``auto_reset=False`` (CRIT-2).

    Uses a fake env that ends an episode at substep 1 of a 4-substep chunk
    without hand-crafting ``final_info`` on the padded tail substep. Before
    the fix, the consumer only read ``infos_list[-1]["episode"]`` which is
    empty on the padded tail, so ``success``/``return``/``episode_len`` were
    silently dropped.
    """
    from rlinf.workers.env.env_worker import EnvWorker  # noqa: F401

    chunk_size = 4

    class _MidChunkEnv:
        """Single env that terminates at substep index 1 (mid-chunk).

        Substep 1's info carries the real ``episode`` metrics (as the
        wrapper would emit on the actual terminal step).  Subsequent
        padded substeps carry a bare ``_valid_step`` marker only — the
        natural shape of ``auto_reset=False`` output.
        """

        def chunk_step(self, chunk_actions):
            obs_list = [
                {"state": torch.tensor([[float(i)]])} for i in range(chunk_size)
            ]
            rewards = torch.zeros(1, chunk_size)
            # ignore_terminations=False path uses chunk_dones; True path uses
            # chunk_truncations.  Mark both at substep 1 so the same fake
            # works for both variants below.
            terminations = torch.tensor(
                [[False, True, False, False]], dtype=torch.bool
            )
            truncations = torch.tensor(
                [[False, True, False, False]], dtype=torch.bool
            )
            infos_list = [
                {"_valid_step": True},
                {
                    "episode": {
                        "return": torch.tensor([5.5]),
                        "episode_len": torch.tensor([2]),
                        "success": torch.tensor([True]),
                    },
                    "_valid_step": True,
                },
                {"_valid_step": False},
                {"_valid_step": False},
            ]
            return obs_list, rewards, terminations, truncations, infos_list

    # --- ignore_terminations=False branch (uses chunk_dones) ---
    worker = _make_env_worker_for_chunk_test(ignore_terminations=False)
    worker.env_list = [_MidChunkEnv()]
    _, env_info = worker.env_interact_step(
        torch.zeros(1, chunk_size, 2), stage_id=0
    )
    assert "return" in env_info, (
        "auto_reset=False mid-chunk terminations must surface episode "
        "metrics from the substep where they happened, not the padded tail."
    )
    torch.testing.assert_close(env_info["return"], torch.tensor([5.5]))
    torch.testing.assert_close(env_info["episode_len"], torch.tensor([2]))
    torch.testing.assert_close(env_info["success"], torch.tensor([True]))

    # --- ignore_terminations=True branch (uses chunk_truncations) ---
    worker = _make_env_worker_for_chunk_test(ignore_terminations=True)
    worker.env_list = [_MidChunkEnv()]
    _, env_info = worker.env_interact_step(
        torch.zeros(1, chunk_size, 2), stage_id=0
    )
    assert "return" in env_info
    torch.testing.assert_close(env_info["return"], torch.tensor([5.5]))
    torch.testing.assert_close(env_info["episode_len"], torch.tensor([2]))
    torch.testing.assert_close(env_info["success"], torch.tensor([True]))


def test_lerobot_cli_default_intervene_flags_are_persisted(tmp_path):
    """T3 — ``--no-default-intervene`` CLI persists ``intervene_flags`` (MAJ-9 / CLI surface).

    Convert a parquet with no ``intervene_flag`` column twice: once with
    ``--no-default-intervene`` (all False expected) and once with
    ``--default-intervene`` (all True expected).
    """
    pandas = pytest.importorskip("pandas")

    def _build_parquet(target_root):
        data_dir = target_root / "data" / "chunk-000"
        data_dir.mkdir(parents=True)
        frames = []
        for idx in (0, 1):
            frame = _frame_with_next(
                idx, episode_index=0, done=(idx == 1), terminated=(idx == 1)
            )
            frame.pop("intervene_flag", None)
            frames.append(frame)
        pandas.DataFrame(frames).to_parquet(data_dir / "episode_000000.parquet")

    def _run(dataset_path, save_path, extra_flag: str):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "rlinf.data.lerobot_replay_buffer",
                "--dataset-path",
                str(dataset_path),
                "--save-path",
                str(save_path),
                extra_flag,
            ],
            check=True,
        )

    # --- --no-default-intervene → all False ---
    no_default_root = tmp_path / "no_default"
    no_default_root.mkdir()
    _build_parquet(no_default_root)
    no_default_save = tmp_path / "no_default_save"
    _run(no_default_root, no_default_save, "--no-default-intervene")
    buffer_no = _load_test_replay_buffer(no_default_save)
    trajectory_no = buffer_no.load_trajectory(0, "lerobot")
    assert trajectory_no.intervene_flags is not None
    assert not trajectory_no.intervene_flags.any(), (
        "--no-default-intervene must persist intervene_flag=False on frames "
        f"that omit the column; got {trajectory_no.intervene_flags!r}."
    )

    # --- --default-intervene → all True ---
    default_root = tmp_path / "default"
    default_root.mkdir()
    _build_parquet(default_root)
    default_save = tmp_path / "default_save"
    _run(default_root, default_save, "--default-intervene")
    buffer_yes = _load_test_replay_buffer(default_save)
    trajectory_yes = buffer_yes.load_trajectory(0, "lerobot")
    assert trajectory_yes.intervene_flags is not None
    assert trajectory_yes.intervene_flags.all(), (
        "--default-intervene must persist intervene_flag=True on frames "
        f"that omit the column; got {trajectory_yes.intervene_flags!r}."
    )


def test_lerobot_trajectory_roundtrip_preserves_per_field_values(tmp_path):
    """T4 — Roundtrip at the value level (replay-buffer save/load).

    Convert one episode, save via the buffer's checkpoint API, load it back,
    and assert every per-field tensor is bit-exact equal to the source.  This
    closes the gap that earlier tests only checked shapes/keys.
    """
    frames = [
        _frame_with_next(0, reward=0.25, intervene=False),
        _frame_with_next(
            1, reward=0.75, done=True, terminated=True, intervene=True
        ),
    ]
    source_trajectory = lerobot_episode_to_trajectory(frames)

    save_path = tmp_path / "buffer"
    write_lerobot_frames_to_replay_buffer(frames, str(save_path))

    loaded_buffer = _load_test_replay_buffer(save_path)
    loaded_trajectory = loaded_buffer.load_trajectory(0, "lerobot")

    for field in (
        "actions",
        "rewards",
        "dones",
        "terminations",
        "truncations",
        "intervene_flags",
        "loss_mask",
    ):
        src = getattr(source_trajectory, field)
        dst = getattr(loaded_trajectory, field)
        assert dst is not None, f"loaded trajectory missing field '{field}'"
        torch.testing.assert_close(
            src,
            dst,
            msg=f"value mismatch for field '{field}' after save/load roundtrip",
        )

    # Observation dicts are nested.
    for obs_name in ("curr_obs", "next_obs"):
        src_obs = getattr(source_trajectory, obs_name)
        dst_obs = getattr(loaded_trajectory, obs_name)
        assert dst_obs is not None
        assert set(src_obs.keys()) == set(dst_obs.keys())
        for key in src_obs:
            torch.testing.assert_close(
                src_obs[key],
                dst_obs[key],
                msg=(
                    f"value mismatch for {obs_name}['{key}'] after save/load "
                    "roundtrip"
                ),
            )


def test_lerobot_multi_episode_per_parquet_splits_into_separate_trajectories(
    tmp_path,
):
    """T5 — Multi-episode-per-parquet fixture.

    One parquet file contains rows for two different ``episode_index`` values;
    ``convert_lerobot_frames_to_trajectories`` must produce two trajectories
    with the per-episode actions / rewards intact (MAJ-9 grouping).
    """
    pandas = pytest.importorskip("pandas")
    data_dir = tmp_path / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    rows = [
        _frame_with_next(0, episode_index=0, reward=0.1),
        _frame_with_next(
            1, episode_index=0, reward=0.2, done=True, terminated=True
        ),
        _frame_with_next(0, episode_index=1, reward=0.3),
        _frame_with_next(
            1, episode_index=1, reward=0.4, done=True, terminated=True
        ),
    ]
    pandas.DataFrame(rows).to_parquet(data_dir / "episode_000000.parquet")

    frames = load_lerobot_parquet_frames(str(tmp_path))
    trajectories = convert_lerobot_frames_to_trajectories(frames)

    assert len(trajectories) == 2, (
        f"expected 2 trajectories (one per episode_index); got "
        f"{len(trajectories)}."
    )

    # Episodes preserve their own action / reward stream.
    torch.testing.assert_close(
        trajectories[0].rewards[:, 0, 0],
        torch.tensor([0.1, 0.2]),
    )
    torch.testing.assert_close(
        trajectories[1].rewards[:, 0, 0],
        torch.tensor([0.3, 0.4]),
    )
    torch.testing.assert_close(
        trajectories[0].actions[:, 0, 0],
        torch.tensor([10.0, 11.0]),
    )
    torch.testing.assert_close(
        trajectories[1].actions[:, 0, 0],
        torch.tensor([10.0, 11.0]),
    )


def test_lerobot_optional_image_null_accepts_nan_and_pandas_na():
    """T6 — Optional image cell ``np.nan`` and ``pandas.NA`` are treated as missing.

    ``_is_missing_value`` was previously hard-coded to recognise only
    ``None``; the F1 fix added length-1 NaN tensor/array handling.  This test
    asserts the same handling for the ``np.nan`` scalar and the pandas
    ``pd.NA`` sentinel commonly produced by parquet readers for optional
    image columns.
    """
    pd = pytest.importorskip("pandas")

    for missing_value in (np.nan, pd.NA):
        frames = [_frame_with_next(0, done=True, terminated=True)]
        # Set the optional wrist_image column to the missing sentinel.  The
        # trajectory should not raise — it should be treated as if the key
        # were absent.
        frames[0]["wrist_image"] = missing_value
        frames[0]["next_wrist_image"] = missing_value
        frames[0]["extra_view_image"] = missing_value
        frames[0]["next_extra_view_image"] = missing_value

        trajectory = lerobot_episode_to_trajectory(frames)
        assert "wrist_images" not in trajectory.curr_obs, (
            f"missing sentinel {type(missing_value).__name__!s} must not "
            "produce a wrist_images entry."
        )
        assert "extra_view_images" not in trajectory.curr_obs


@pytest.mark.parametrize(
    "field_name, bad_value",
    [
        # Multi-element NaN/Inf in state — caught by `_check_finite_nonempty`.
        ("state", np.array([np.nan, 0.5], dtype=np.float32)),
        ("state", np.array([np.inf, 0.5], dtype=np.float32)),
        # 2-element NaN/Inf actions — caught by `_check_finite_nonempty`.
        # (A length-1 NaN is deliberately treated as missing per MAJ-1, so
        # we test multi-element here to hit the finite check.)
        ("actions", np.array([np.nan, 0.0], dtype=np.float32)),
        ("actions", np.array([np.inf, 0.0], dtype=np.float32)),
        # Rewards must be scalar; length-1 Inf hits the finite check.
        # (A length-1 NaN is treated as missing per MAJ-1, so reward
        # falls back to the legacy success-based path — covered by the
        # separate `terminal_reward` NaN test below.)
        ("rewards", np.array([np.inf], dtype=np.float32)),
        ("rewards", np.array([-np.inf], dtype=np.float32)),
    ],
)
def test_lerobot_rejects_nan_inf_in_state_action_reward(field_name, bad_value):
    """T7 — NaN/Inf rejection at state / action / reward (MAJ-2)."""
    frames = [
        _frame_with_next(0),
        _frame_with_next(1, done=True, terminated=True),
    ]
    frames[0][field_name] = bad_value

    with pytest.raises(ValueError, match="finite"):
        lerobot_episode_to_trajectory(frames)


def test_lerobot_terminal_reward_cli_rejects_nan_inf():
    """T7 — CLI ``--terminal-reward`` rejects NaN/Inf at parse time (MAJ-2)."""
    from rlinf.data.lerobot_replay_buffer import _finite_float

    import argparse as _argparse

    for bad in ("nan", "inf", "-inf", "Infinity"):
        with pytest.raises(_argparse.ArgumentTypeError, match="finite"):
            _finite_float(bad)

    # Also at the convert API level.
    frames = [
        _frame_with_next(0),
        _frame_with_next(1, done=True, terminated=True),
    ]
    with pytest.raises(ValueError, match="finite"):
        lerobot_episode_to_trajectory(frames, terminal_reward=float("nan"))


def test_lerobot_outside_path_containment_checked_before_decode(tmp_path):
    """T8 — Path traversal rejected before reading file contents.

    Uses a *valid* PNG OUTSIDE the dataset root (the existing test wrote
    decoy bytes which would fail decode anyway, hiding whether the rejection
    happened before or after the read). We monkey-patch ``Path.read_bytes``
    to record any call, then assert the outside-PNG was never read.
    """
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    outside_image = tmp_path / "outside.png"
    outside_image.write_bytes(_png_bytes((33, 34, 35)))  # valid PNG

    frames = [_frame_with_next(0, done=True, terminated=True)]
    for frame in frames:
        frame["_lerobot_dataset_root"] = str(dataset_root)
        frame["image"] = {"path": "../outside.png"}
        frame["next_image"] = np.zeros((2, 3, 3), dtype=np.uint8)

    read_calls: list[str] = []
    original_read_bytes = type(outside_image).read_bytes

    def _spying_read_bytes(self):
        read_calls.append(str(self))
        return original_read_bytes(self)

    from pathlib import Path as _Path

    with patch.object(_Path, "read_bytes", _spying_read_bytes):
        with pytest.raises(ValueError, match="escapes.*dataset root"):
            lerobot_episode_to_trajectory(frames)

    assert str(outside_image) not in read_calls, (
        "Outside-root PNG bytes must not be read before containment is "
        f"checked; observed reads: {read_calls!r}."
    )


def test_lerobot_dataset_writer_default_schema_omits_transition_fields():
    """T9 — ``LeRobotDatasetWriter.create()`` default schema (MAJ-3).

    With ``transition_schema=False`` (the new default) only the minimal set of
    fields are exposed; with ``transition_schema=True`` the transition-rich
    set (``next_state``, ``next_image``, ``terminated``, ``truncated``,
    ``rewards``) appears.

    We capture the ``features`` dict that ``create()`` would pass to the
    upstream LeRobotDataset library by patching the import — this avoids
    requiring lerobot to be installed in the unit-test environment.
    """
    from rlinf.data.lerobot_writer import LeRobotDatasetWriter

    minimal_writer = LeRobotDatasetWriter()
    rich_writer = LeRobotDatasetWriter()

    captured: dict[str, dict] = {}

    class _FakeLeRobotDataset:
        @classmethod
        def create(cls, *, repo_id, features, **_kwargs):
            captured[repo_id] = features
            return SimpleNamespace(image_writer=None, episode_buffer=None)

    fake_module = SimpleNamespace(LeRobotDataset=_FakeLeRobotDataset)
    fake_common = SimpleNamespace(datasets=SimpleNamespace(lerobot_dataset=fake_module))
    fake_root = SimpleNamespace(common=fake_common)

    with patch.dict(
        sys.modules,
        {
            "lerobot": fake_root,
            "lerobot.common": fake_common,
            "lerobot.common.datasets": fake_common.datasets,
            "lerobot.common.datasets.lerobot_dataset": fake_module,
        },
    ):
        # T9 — call WITHOUT transition_schema so we exercise the actual
        # default; if the default regressed to True the assertions below
        # would fail.
        minimal_writer.create(repo_id="minimal")
        rich_writer.create(repo_id="rich", transition_schema=True)

    minimal_features = captured["minimal"]
    rich_features = captured["rich"]

    transition_only_fields = {
        "next_state",
        "next_image",
        "terminated",
        "truncated",
        "rewards",
    }
    minimal_expected = {
        "state",
        "actions",
        "done",
        "is_success",
        "intervene_flag",
        "image",
    }

    assert (
        set(minimal_features.keys()) - transition_only_fields
    ), "minimal schema is empty"
    # MAJ-3: transition-rich fields must be ABSENT from the new default.
    for field in transition_only_fields:
        assert field not in minimal_features, (
            f"Default (transition_schema=False) schema must not include "
            f"'{field}'; this is the back-compat contract restored in MAJ-3."
        )
    assert minimal_expected.issubset(minimal_features.keys()), (
        f"Default schema must keep the minimal set; missing="
        f"{minimal_expected - set(minimal_features.keys())!r}."
    )

    # transition_schema=True must include the transition-rich keys.
    for field in transition_only_fields:
        assert field in rich_features, (
            f"transition_schema=True must include '{field}' for the "
            "replay-buffer roundtrip path."
        )


def test_sac_mask_integration_skips_optimizer_when_all_padded():
    """T10 — SAC mask integration over forward_critic/actor/alpha (MAJ-4).

    Drives ``apply_chunk_mask`` end-to-end through a deterministic per-sample
    loss, then asserts the masked reduction excludes invalid (False) cells.
    Also simulates the ``update_one_epoch`` skip-when-zero logic with mock
    optimizers/schedulers to verify the all-zero-mask path is a true no-op.
    """
    from unittest.mock import MagicMock

    from rlinf.workers.actor.sac_demo_buffer_utils import (
        apply_chunk_mask,
        chunk_sample_weights,
    )

    # ---- partial mask: last cell of the first chunk is invalid ----
    # loss_mask shape [B=2, num_chunks=3]; chunk_sample_weights ANDs across the
    # chunk dim, so any False there nukes the whole chunk's weight.
    loss_mask = torch.tensor(
        [[True, True, False], [True, True, True]], dtype=torch.bool
    )
    batch = {"loss_mask": loss_mask}
    weights = chunk_sample_weights(batch, torch.float32)
    torch.testing.assert_close(weights, torch.tensor([[0.0], [1.0]]))

    # Deterministic per-sample loss = identity values; masked mean must equal
    # the average over only the second (valid) chunk.
    per_sample = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    loss, valid_count = apply_chunk_mask(per_sample, weights)
    # Only sample 1 contributes; feature average = (4+5+6) / 3 = 5.
    torch.testing.assert_close(loss, torch.tensor(5.0))
    assert valid_count == 1

    # ---- all-zero mask: forward returns graph-connected zero, count==0 ----
    # CRIT-R2-2 update: the all-zero loss MUST stay graph-connected so
    # ``backward()`` can be safely called on every rank (FSDP backward is
    # collective; rank-local skips deadlock peers). The skip-step
    # semantics live in the worker via the global valid-count gate.
    zero_weights = torch.zeros(2, 1)
    per_sample_requires_grad = per_sample.clone().requires_grad_(True)
    loss_zero, valid_count_zero = apply_chunk_mask(
        per_sample_requires_grad, zero_weights
    )
    assert valid_count_zero == 0
    assert loss_zero.item() == 0.0
    assert loss_zero.requires_grad, (
        "All-padded path must return a GRAPH-CONNECTED zero so every rank "
        "can safely call backward() and FSDP collectives stay consistent."
    )
    # Backward on the graph-connected zero must succeed (no FSDP grad-fn
    # error) and produce zero grads.
    loss_zero.backward()
    assert per_sample_requires_grad.grad is not None
    assert torch.equal(
        per_sample_requires_grad.grad, torch.zeros_like(per_sample_requires_grad)
    )

    # ---- skip-step semantics: simulate the MAJ-4 update_one_epoch contract ----
    optimizer = MagicMock()
    scheduler = MagicMock()
    target_soft_update = MagicMock()

    def _maybe_step(loss_tensor, valid_count_value):
        """Mirrors the worker's `if valid_count > 0` guard."""
        if valid_count_value > 0:
            loss_tensor.backward() if loss_tensor.requires_grad else None
            optimizer.step()
            scheduler.step()
            target_soft_update()

    _maybe_step(loss_zero, valid_count_zero)
    assert optimizer.step.call_count == 0, (
        "All-zero mask must NOT advance optimizer.step()."
    )
    assert scheduler.step.call_count == 0, (
        "All-zero mask must NOT advance scheduler.step() (LR schedule "
        "advance was the most insidious failure in MAJ-4)."
    )
    assert target_soft_update.call_count == 0, (
        "All-zero mask must NOT trigger the target soft-update."
    )

    # Sanity-check the converse: with a non-zero valid_count, step DOES run.
    loss_valid = per_sample[1:].mean().clone().requires_grad_(True)
    _maybe_step(loss_valid, 1)
    assert optimizer.step.call_count == 1
    assert scheduler.step.call_count == 1
    assert target_soft_update.call_count == 1


# =====================================================================
# Round 3 regression tests: close the gaps Round 2 fixes added but
# didn't otherwise cover.
# =====================================================================


def test_global_valid_count_falls_back_to_local_without_distributed():
    """CRIT-R2-2 — ``_global_valid_count`` in single-process mode.

    The FSDP-collective fix all-reduces the per-rank valid count before
    gating optimizer/scheduler/all_reduce. In unit-test scope (no
    ``torch.distributed`` init), it must fall back to the local value
    so single-GPU training still works.
    """
    EmbodiedSACFSDPPolicy = pytest.importorskip(
        "rlinf.workers.actor.fsdp_sac_policy_worker",
        reason=(
            "transformers / fsdp deps missing in this CI image; the fallback "
            "logic itself is exercised by integration tests."
        ),
        exc_type=ImportError,
    ).EmbodiedSACFSDPPolicy

    worker = object.__new__(EmbodiedSACFSDPPolicy)
    worker.device = torch.device("cpu")
    # No torch.distributed init in unit tests.
    assert worker._global_valid_count(0) == 0
    assert worker._global_valid_count(7) == 7


def test_env_worker_close_calls_close_on_every_env():
    """CRIT-R2-NEW1 — ``EnvWorker._close`` drains every env wrapper.

    Without this hook ``ray.kill`` tears the actor down before
    ``CollectEpisode.close()`` finishes, dropping collected episodes.
    """
    from rlinf.workers.env.env_worker import EnvWorker

    worker = object.__new__(EnvWorker)
    closes = []

    class _Env:
        def __init__(self, name):
            self.name = name

        def close(self):
            closes.append(self.name)

    worker.env_list = [_Env("train_0"), _Env("train_1")]
    worker.eval_env_list = [_Env("eval_0")]
    worker._close()
    assert closes == ["train_0", "train_1", "eval_0"]


def test_env_worker_close_swallows_per_env_errors_but_raises_first():
    """CRIT-R2-NEW1 — A single bad ``close()`` must not leak other envs."""
    from rlinf.workers.env.env_worker import EnvWorker

    worker = object.__new__(EnvWorker)
    closes = []

    class _Env:
        def __init__(self, name, raises=False):
            self.name = name
            self.raises = raises

        def close(self):
            if self.raises:
                raise RuntimeError(f"bad-close-{self.name}")
            closes.append(self.name)

    worker.env_list = [
        _Env("train_0", raises=True),
        _Env("train_1"),
    ]
    worker.eval_env_list = [_Env("eval_0", raises=True)]
    with pytest.raises(RuntimeError, match="bad-close-train_0"):
        worker._close()
    # train_1 still got closed despite train_0's error.
    assert "train_1" in closes


def test_replay_buffer_close_propagates_async_save_error(tmp_path):
    """CRIT-R2-NEW2 — ``close(wait=True)`` surfaces async-save failures.

    Before the fix, ``_save_metadata`` / ``_save_trajectory_index`` errors
    were caught inside the executor's submitted future but never observed
    because ``close()`` shut the executor down without calling
    ``.result()``. Caller saw a clean exit + incomplete checkpoint.
    """
    from rlinf.data.replay_buffer import TrajectoryReplayBuffer

    save_path = tmp_path / "buffer"
    save_path.mkdir()
    buffer = TrajectoryReplayBuffer(
        auto_save=True,
        auto_save_path=str(save_path),
        enable_cache=False,
    )

    boom = RuntimeError("simulated save failure")

    def _failing_metadata(*_args, **_kwargs):
        raise boom

    # Force the next async metadata flush to fail.
    buffer._save_metadata = _failing_metadata  # type: ignore[assignment]
    fut = buffer._save_executor.submit(buffer._save_metadata)
    with buffer._pending_save_lock:
        buffer._pending_save_futures.append(fut)

    with pytest.raises(RuntimeError, match="simulated save failure"):
        buffer.close(wait=True)


def test_replay_buffer_atomic_metadata_write_no_partial_file(tmp_path):
    """CRIT-R2-11 — Metadata write must be atomic.

    Simulate a write that fails AFTER the JSON dump but before
    ``os.replace``: the target file must NOT exist (so a subsequent load
    won't read a half-written checkpoint) and the temp file must be
    cleaned up.
    """
    from rlinf.data.replay_buffer import TrajectoryReplayBuffer
    import os as _os

    target = tmp_path / "metadata.json"
    original_replace = _os.replace

    def _replace_fail(src, dst):
        # Simulate disk-full / permission failure at the rename step.
        raise OSError("simulated rename failure")

    try:
        _os.replace = _replace_fail
        with pytest.raises(OSError, match="rename failure"):
            TrajectoryReplayBuffer._atomic_write_json(
                str(target), {"x": 1}
            )
    finally:
        _os.replace = original_replace

    assert not target.exists(), (
        "atomic write must leave no half-written target on rename failure"
    )
    # No leftover temp file in the dir.
    temp_files = [p for p in tmp_path.iterdir() if p.name.startswith(".tmp-")]
    assert temp_files == [], (
        f"atomic write leaked temp file(s): {temp_files}"
    )


def test_lerobot_cli_num_action_chunks_propagates_to_loss_mask(tmp_path):
    """CRIT-R2-1 / R3-CLI — ``--num-action-chunks`` actually wires through.

    Round 2 added the kwarg, but if ``_parse_args`` or ``main()`` dropped
    ``args.num_action_chunks`` the new tests would still pass because they
    call the Python API directly. This test runs the CLI and inspects the
    persisted ``loss_mask`` trailing dim to prove the wiring is intact.
    """
    import subprocess
    import sys

    pandas = pytest.importorskip("pandas")
    dataset_root = tmp_path / "ds" / "rank_0" / "id_0"
    (dataset_root / "data").mkdir(parents=True)
    df = pandas.DataFrame(
        [
            _frame_with_next(0),
            _frame_with_next(1, done=True, terminated=True),
        ]
    )
    df.to_parquet(dataset_root / "data" / "episode_000000.parquet")

    save_path = tmp_path / "buffer"
    cmd = [
        sys.executable,
        "-m",
        "rlinf.data.lerobot_replay_buffer",
        "--dataset-path",
        str(tmp_path / "ds"),
        "--save-path",
        str(save_path),
        "--num-action-chunks",
        "7",
        "--state-only",
    ]
    env = {
        **os.environ,
        "PYTHONPATH": "/home/kunni/plusai_ws/RLinf",
    }
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    buf = _load_test_replay_buffer(save_path)
    buf.load_checkpoint(str(save_path))
    [traj_id] = buf.list_trajectory_ids()
    info = buf.get_trajectory_info(traj_id)
    trajectory = buf.load_trajectory(traj_id, info["model_weights_id"])
    assert trajectory.loss_mask is not None
    # The CLI's ``--num-action-chunks 7`` MUST flow through to the persisted
    # trajectory's ``loss_mask`` trailing dim.
    assert trajectory.loss_mask.shape[-1] == 7
    buf.close()


def test_concat_batch_loss_mask_fillable_batch_dim_correct():
    """CRIT-R2-3 — auto-fill of missing ``loss_mask`` uses peer batch dim.

    A subtle bug in the first iteration of the fillable-default code
    duplicated the OTHER side's tensor shape, producing an incorrect
    summed batch dim. The fix derives the fill's leading dim from any
    peer tensor on the side that lacks the key.
    """
    from rlinf.utils.nested_dict_process import concat_batch

    rollout = {"actions": torch.zeros(3, 1, 1)}
    demo = {
        "actions": torch.zeros(2, 1, 1),
        "loss_mask": torch.ones(2, 1, 1, dtype=torch.bool),
    }
    combined = concat_batch(rollout, demo)
    assert combined["actions"].shape == (5, 1, 1)
    # The rollout side had no loss_mask; the fill must match rollout's
    # batch dim (3), then cat with demo's loss_mask (2) → (5, 1, 1).
    assert combined["loss_mask"].shape == (5, 1, 1)
    assert combined["loss_mask"].all()


# =====================================================================
# Round 4 regression tests
# =====================================================================


def test_apply_chunk_mask_zero_path_sanitizes_nan_loss():
    """CRIT-R4-1 — the graph-connected zero must NOT propagate NaN.

    Before R4 the all-zero-mask path computed ``(per_sample_loss * 0.0).sum()``.
    If ``per_sample_loss`` had any NaN/Inf, ``NaN * 0.0 = NaN`` → FSDP
    backward writes NaN grads → clip_grad_norm all_reduces NaN → entire
    model corrupted across the DP group. The R4 fix sanitizes the loss
    before the multiply.
    """
    from rlinf.workers.actor.sac_demo_buffer_utils import apply_chunk_mask

    # Loss has NaN/Inf on what would have been masked positions.
    per_sample = torch.tensor(
        [[float("nan"), 1.0], [float("inf"), 2.0]], requires_grad=True
    )
    zero_weights = torch.zeros(2, 1)
    loss, valid_count = apply_chunk_mask(per_sample, zero_weights)
    assert valid_count == 0
    assert torch.isfinite(loss), (
        "All-zero path must NOT propagate NaN/Inf from masked positions; "
        "otherwise FSDP backward poisons every rank's grads."
    )
    # backward() must work and produce finite (zero) grads.
    loss.backward()
    assert per_sample.grad is not None
    assert torch.isfinite(per_sample.grad).all()


def test_apply_chunk_mask_partial_path_sanitizes_nan_in_masked_positions():
    """CRIT-R4-1 — partial-mask path must also strip NaN/Inf at masked-out cells.

    A NaN/Inf at a position the mask zeros out would still propagate
    through the ``per_sample_loss * weights`` multiply, because
    ``NaN * 0.0 == NaN``.
    """
    from rlinf.workers.actor.sac_demo_buffer_utils import apply_chunk_mask

    # Sample 0 is invalid (weight 0) but its loss is NaN. Sample 1 is valid.
    per_sample = torch.tensor(
        [[float("nan"), float("inf")], [3.0, 4.0]], requires_grad=True
    )
    weights = torch.tensor([[0.0], [1.0]])
    loss, valid_count = apply_chunk_mask(per_sample, weights)
    assert valid_count == 1
    assert torch.isfinite(loss), (
        "The masked-out NaN must NOT poison the valid-sample reduction."
    )
    # The valid sample's average loss is (3+4)/2 = 3.5.
    torch.testing.assert_close(loss, torch.tensor(3.5))


def test_concat_batch_strict_rejects_curr_obs_only_in_one_side():
    """CRIT-R4-5 — strict mode catches dict-key drift on uniform-required dicts.

    ``forward_inputs`` is opt-in (lenient), but ``curr_obs`` / ``next_obs``
    must be uniform; silent drop would corrupt observation schemas.
    """
    from rlinf.utils.nested_dict_process import concat_batch

    rollout = {
        "actions": torch.zeros(2, 1, 1),
        "curr_obs": {"states": torch.zeros(2, 1, 4)},
    }
    demo = {
        "actions": torch.zeros(3, 1, 1),
        # Missing curr_obs entirely — must be rejected under strict mode.
    }
    with pytest.raises(ValueError, match="dict keys only in data1"):
        concat_batch(rollout, demo)


def test_concat_batch_strict_allows_forward_inputs_asymmetry():
    """CRIT-R4-5 — opt-in dicts (``forward_inputs``) keep working when
    only one side carries them.

    The lenient list (``_CONCAT_BATCH_LENIENT_DICT_KEYS``) must remain in
    effect: production rollouts often have an empty ``forward_inputs``
    while LeRobot demos always populate it.
    """
    from rlinf.utils.nested_dict_process import concat_batch

    rollout = {"actions": torch.zeros(2, 1, 1)}
    demo = {
        "actions": torch.zeros(3, 1, 1),
        "forward_inputs": {"action": torch.zeros(3, 1, 1)},
    }
    # Must not raise; concat the actions; forward_inputs is opt-in.
    combined = concat_batch(rollout, demo)
    assert combined["actions"].shape == (5, 1, 1)


def test_unknown_camera_keys_routes_slash_form_front_wrist():
    """CRIT-R4-6 — slash-form camera keys must actually be routed.

    Before R4, ``observation/images/front`` passed the unknown-key guard
    (matching prefix + camera name "front") but the routed key tuple only
    contained the dot-form, so the data was silently dropped.
    """
    from rlinf.data.lerobot_replay_buffer import (
        _MAIN_IMAGE_KEYS,
        _NEXT_MAIN_IMAGE_KEYS,
        _WRIST_IMAGE_KEYS,
        _NEXT_WRIST_IMAGE_KEYS,
    )

    assert "observation/images/front" in _MAIN_IMAGE_KEYS
    assert "next_observation/images/front" in _NEXT_MAIN_IMAGE_KEYS
    assert "observation/images/wrist" in _WRIST_IMAGE_KEYS
    assert "next_observation/images/wrist" in _NEXT_WRIST_IMAGE_KEYS


def test_replay_buffer_clear_resets_durable_trajectory_ids(tmp_path):
    """CRIT-R4-4 — ``clear()`` must reset ``_durable_trajectory_ids``.

    Without this, the next ``_save_metadata`` writes a non-zero ``size``
    (durable set inflated by stale IDs) while ``_save_trajectory_index``
    writes an empty index → metadata says "1 trajectory" but the index
    has none.
    """
    from rlinf.data.replay_buffer import TrajectoryReplayBuffer

    save_path = tmp_path / "buf"
    save_path.mkdir()
    buffer = TrajectoryReplayBuffer(
        auto_save=True,
        auto_save_path=str(save_path),
        enable_cache=False,
    )
    # Simulate a successful save that promoted the id to durable.
    buffer._durable_trajectory_ids.add(0)
    assert buffer._durable_trajectory_ids == {0}
    buffer.clear()
    assert buffer._durable_trajectory_ids == set(), (
        "clear() must reset the durable id set or persisted size will "
        "disagree with persisted index."
    )
    buffer.close()


def test_replay_buffer_is_ready_excludes_pending_in_flight_trajectories(
    tmp_path, monkeypatch
):
    """CRIT-R4-3 — ``is_ready`` must not count trajectories whose save is
    still in flight.

    Before R4, ``size`` counted them — ``sample_chunks`` could then pick
    one and ``_load_trajectory`` would FileNotFoundError on the missing
    payload.
    """
    from rlinf.data.replay_buffer import TrajectoryReplayBuffer

    save_path = tmp_path / "buf"
    save_path.mkdir()
    buffer = TrajectoryReplayBuffer(
        auto_save=True,
        auto_save_path=str(save_path),
        enable_cache=False,
    )

    # Pretend `add_trajectories` registered an in-flight id WITHOUT
    # promoting it to durable yet.
    with buffer._index_lock:
        buffer._trajectory_index[0] = {
            "num_samples": 4,
            "trajectory_id": 0,
            "max_episode_length": 4,
            "shape": (4, 1, 1),
            "model_weights_id": "test",
        }
        buffer._trajectory_id_list.append(0)
        buffer.size = 1
        buffer._total_samples = 4
        # `_durable_trajectory_ids` remains empty — file not yet on disk.

    assert not buffer.is_ready(1), (
        "is_ready must filter to durable trajectories under auto_save; "
        "otherwise sample_chunks crashes on the missing file."
    )

    # Once durable, is_ready becomes true.
    with buffer._index_lock:
        buffer._durable_trajectory_ids.add(0)
    assert buffer.is_ready(1)

    buffer.close()


def test_replay_buffer_sample_chunks_excludes_pending_in_flight(tmp_path):
    """CRIT-R5-1 — ``sample_chunks`` must also filter against durable IDs.

    Before R5, only ``is_ready`` consulted the durable set; ``sample_chunks``
    walked ``_trajectory_id_list`` directly, so a pending in-flight id
    could be sampled and ``_load_trajectory`` would crash with
    ``FileNotFoundError`` on the missing payload file.
    """
    from rlinf.data.replay_buffer import TrajectoryReplayBuffer

    save_path = tmp_path / "buf"
    save_path.mkdir()
    buffer = TrajectoryReplayBuffer(
        auto_save=True,
        auto_save_path=str(save_path),
        sample_window_size=0,  # include all eligible
        enable_cache=False,
    )

    # Register one in-flight id (no payload yet) AND one durable id (with
    # payload written). sample_chunks must only see the durable one.
    with buffer._index_lock:
        for tid, durable in [(0, False), (1, True)]:
            buffer._trajectory_index[tid] = {
                "num_samples": 4,
                "trajectory_id": tid,
                "max_episode_length": 4,
                "shape": (4, 1, 1),
                "model_weights_id": "test",
            }
            buffer._trajectory_id_list.append(tid)
            buffer._trajectory_file_path[tid] = str(save_path)
            if durable:
                buffer._durable_trajectory_ids.add(tid)
        buffer.size = 2
        buffer._total_samples = 8

    # Write the payload for id=1 only so ``_load_trajectory`` would succeed.
    minimal_traj = SimpleNamespace(
        max_episode_length=4,
        model_weights_id="test",
    )
    # Use the real save path so the file matches what _load_trajectory
    # expects; only id=1 has a file on disk.
    from rlinf.data.embodied_io_struct import Trajectory

    real_traj = Trajectory(
        max_episode_length=4,
        model_weights_id="test",
        actions=torch.zeros(4, 1, 1),
        rewards=torch.zeros(4, 1, 1),
        terminations=torch.zeros(4, 1, 1, dtype=torch.bool),
        truncations=torch.zeros(4, 1, 1, dtype=torch.bool),
        dones=torch.zeros(4, 1, 1, dtype=torch.bool),
        intervene_flags=torch.zeros(4, 1, 1, dtype=torch.bool),
        forward_inputs={},
        curr_obs={"states": torch.zeros(4, 1, 2)},
        next_obs={"states": torch.zeros(4, 1, 2)},
    )
    buffer._save_trajectory(real_traj, 1, "test")

    # Repeated sampling must NEVER hit the missing pending id (file for
    # id=0 does not exist on disk).
    for _ in range(20):
        batch = buffer.sample_chunks(2)
        # Sampling must succeed (no FileNotFoundError).
        assert "actions" in batch

    buffer.close()


def test_update_one_epoch_skips_optim_and_scheduler_under_all_padded_global():
    """T10 (real-code variant) — ``update_one_epoch`` must NOT advance
    the optimizer / LR scheduler / soft target when GLOBALLY zero valid
    samples.

    The Round 4 ``T10`` covered only the helper-level guard via
    MagicMocks; this version instantiates ``EmbodiedSACFSDPPolicy`` via
    ``object.__new__`` and monkey-patches the forward functions +
    optimizers so we exercise the actual gating path in
    ``update_one_epoch``. A regression that ignored
    ``_global_valid_count`` and advanced LR/target on the all-padded
    global batch would now fail this test.
    """
    fsdp_mod = pytest.importorskip(
        "rlinf.workers.actor.fsdp_sac_policy_worker",
        reason="transformers / fsdp deps missing in this CI image",
        exc_type=ImportError,
    )
    EmbodiedSACFSDPPolicy = fsdp_mod.EmbodiedSACFSDPPolicy
    from unittest.mock import MagicMock

    worker = object.__new__(EmbodiedSACFSDPPolicy)
    worker.device = torch.device("cpu")
    worker.cfg = OmegaConf.create(
        {
            "actor": {
                "critic_optim": {"clip_grad": 1.0},
                "optim": {"clip_grad": 1.0},
            },
            "algorithm": {
                "entropy_tuning": {"optim": {"clip_grad": 1.0}},
                "target_update_freq": 1,
            },
        }
    )
    worker.gradient_accumulation = 1
    worker.critic_actor_ratio = 1
    worker.update_step = 0
    worker.enable_drq = False
    worker.use_dsrl = False
    worker.target_model_initialized = True

    # Spy optimizers/schedulers/target updater.
    worker.qf_optimizer = MagicMock()
    worker.qf_lr_scheduler = MagicMock()
    worker.optimizer = MagicMock()
    worker.lr_scheduler = MagicMock()
    worker.alpha_optimizer = None  # disabled for this test
    worker.entropy_temp = MagicMock()
    worker.entropy_temp.alpha = 0.1
    worker.model = MagicMock()
    worker.model.clip_grad_norm_ = MagicMock(return_value=0.0)
    worker.soft_update_target_model = MagicMock()

    # Forward functions return a graph-connected ZERO loss (no valid samples).
    sentinel = torch.zeros((), requires_grad=True)

    def fake_forward_critic(_batch):
        return (sentinel * 0.0).sum(), 0, {}

    def fake_forward_actor(_batch):
        return (sentinel * 0.0).sum(), torch.tensor(0.0), 0, {}

    worker.forward_critic = fake_forward_critic
    worker.forward_actor = fake_forward_actor

    # Dataloader hands the worker a one-microbatch list (we bypass via
    # `train_micro_batch_list` injection).
    worker.buffer_dataloader_iter = iter(
        [
            {
                "loss_mask": torch.zeros(1, 1, dtype=torch.bool),
                "curr_obs": {"states": torch.zeros(1, 1, 2)},
                "next_obs": {"states": torch.zeros(1, 1, 2)},
                "actions": torch.zeros(1, 1, 1),
            }
        ]
    )

    # Pretend global_batch_size==micro_batch_size==1 so the split helper
    # returns a one-element list (no FSDP split bookkeeping).
    worker.cfg.actor.global_batch_size = 1
    worker.cfg.actor.micro_batch_size = 1
    worker._world_size = 1

    # Force _global_valid_count to return 0 to simulate global-padded.
    worker._global_valid_count = MagicMock(return_value=0)

    # Patch DRQ + worker_timer + put_tensor_device + split_dict_to_chunk
    # to be inert.
    worker.worker_timer = MagicMock()
    worker.worker_timer.return_value.__enter__ = MagicMock()
    worker.worker_timer.return_value.__exit__ = MagicMock(return_value=False)

    # Run one epoch. Must not raise; must not step.
    metrics_data = EmbodiedSACFSDPPolicy.update_one_epoch(worker, train_actor=True)

    assert worker.qf_optimizer.step.call_count == 0, (
        "qf_optimizer.step MUST NOT advance under all-padded global batch."
    )
    assert worker.qf_lr_scheduler.step.call_count == 0, (
        "qf_lr_scheduler.step MUST NOT advance — that was the MAJ-4 bug."
    )
    assert worker.optimizer.step.call_count == 0
    assert worker.lr_scheduler.step.call_count == 0
    assert worker.soft_update_target_model.call_count == 0, (
        "Target soft-update MUST be skipped on all-padded global batch."
    )
    # Reported critic_valid_count must reflect the global zero.
    assert int(metrics_data.get("critic/valid_count", -1)) == 0


def test_apply_chunk_mask_zero_path_still_works_without_distributed():
    """CRIT-R7 — apply_chunk_mask must not deadlock when valid==0.

    Round 7's fix moved the all-reduce of ``valid_count`` to BEFORE the
    early return for the all-padded case, so every rank participates in
    the collective regardless of local validity. In single-process mode
    the all-reduce is skipped; the zero path still returns a finite
    graph-connected zero.
    """
    from rlinf.workers.actor.sac_demo_buffer_utils import apply_chunk_mask

    per_sample = torch.tensor([[1.0, 2.0]], requires_grad=True)
    zero_weights = torch.zeros(1, 1)
    loss, valid = apply_chunk_mask(per_sample, zero_weights)
    assert valid == 0
    assert torch.isfinite(loss) and loss.item() == 0.0
    # Backward must succeed (no FSDP grad-fn error) and produce zeros.
    loss.backward()
    assert torch.isfinite(per_sample.grad).all()
    assert per_sample.grad.abs().sum().item() == 0.0


def test_replay_buffer_rejects_add_after_close(tmp_path):
    """CRIT-R8 — ``add_trajectories`` after ``close()`` must raise.

    Otherwise the close-first race window (close drains empty pending,
    then add submits a future close cannot await → executor.shutdown
    swallows the error) silently loses persistence failures.
    """
    from rlinf.data.replay_buffer import TrajectoryReplayBuffer
    from rlinf.data.embodied_io_struct import Trajectory

    save_path = tmp_path / "buf"
    save_path.mkdir()
    buffer = TrajectoryReplayBuffer(
        auto_save=True,
        auto_save_path=str(save_path),
        enable_cache=False,
    )
    buffer.close(wait=True)

    traj = Trajectory(
        max_episode_length=2,
        model_weights_id="test",
        actions=torch.zeros(2, 1, 1),
        rewards=torch.zeros(2, 1, 1),
        terminations=torch.zeros(2, 1, 1, dtype=torch.bool),
        truncations=torch.zeros(2, 1, 1, dtype=torch.bool),
        dones=torch.zeros(2, 1, 1, dtype=torch.bool),
        intervene_flags=torch.zeros(2, 1, 1, dtype=torch.bool),
        forward_inputs={},
        curr_obs={"states": torch.zeros(2, 1, 2)},
        next_obs={"states": torch.zeros(2, 1, 2)},
    )
    with pytest.raises(RuntimeError, match="already been closed"):
        buffer.add_trajectories([traj])


def test_iter_trajectory_metadata_skips_pending_in_flight(tmp_path):
    """CRIT-R7 — iter_trajectory_metadata + list_trajectory_ids must
    not yield in-flight ids whose payload is not durable.

    Otherwise external consumers (visualizer, validator) that call
    ``load_trajectory(id, mwid)`` immediately after iterating would
    crash with FileNotFoundError on the pending payload file.
    """
    from rlinf.data.replay_buffer import TrajectoryReplayBuffer

    save_path = tmp_path / "buf"
    save_path.mkdir()
    buffer = TrajectoryReplayBuffer(
        auto_save=True,
        auto_save_path=str(save_path),
        enable_cache=False,
    )

    with buffer._index_lock:
        for tid in (0, 1):
            buffer._trajectory_index[tid] = {
                "num_samples": 4,
                "trajectory_id": tid,
                "max_episode_length": 4,
                "shape": (4, 1, 1),
                "model_weights_id": "test",
            }
            buffer._trajectory_id_list.append(tid)
        # Only tid=1 is durable; tid=0 is still in flight.
        buffer._durable_trajectory_ids.add(1)
        buffer.size = 2

    ids = buffer.list_trajectory_ids()
    metadata = list(buffer.iter_trajectory_metadata())
    assert ids == [1], (
        f"list_trajectory_ids must filter pending; got {ids}."
    )
    assert [m[0] for m in metadata] == [1], (
        f"iter_trajectory_metadata must filter pending; got {metadata}."
    )

    buffer.close()


def test_loaded_demo_buffer_rejects_loss_mask_chunk_mismatch():
    """CRIT-R5-3 — demo validator must catch num_action_chunks drift.

    A demo buffer converted with ``--num-action-chunks=1`` for a training
    config that uses ``num_action_chunks=5`` would pass the old validator
    and crash at the first ``concat_batch`` with a cryptic shape error.
    The R5 fix rejects the mismatch at load time with an actionable
    message.
    """
    from rlinf.workers.actor.sac_demo_buffer_utils import (
        _validate_required_trajectory_fields,
    )

    trajectory = SimpleNamespace(
        actions=torch.zeros(4, 1, 7),  # 7-dim action_dim
        rewards=torch.zeros(4, 1, 1),
        terminations=torch.zeros(4, 1, 1, dtype=torch.bool),
        truncations=torch.zeros(4, 1, 1, dtype=torch.bool),
        dones=torch.zeros(4, 1, 1, dtype=torch.bool),
        # loss_mask carries the WRONG trailing dim (1 instead of expected 5).
        loss_mask=torch.ones(4, 1, 1, dtype=torch.bool),
    )
    model_cfg = OmegaConf.create(
        {"model_type": "mlp_policy", "action_dim": 7, "num_action_chunks": 5}
    )
    with pytest.raises(ValueError, match="num_action_chunks"):
        _validate_required_trajectory_fields(
            trajectory,
            load_path="x",
            rank=0,
            trajectory_id=0,
            model_cfg=model_cfg,
        )
