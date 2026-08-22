# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROBOT_DESCRIPTION_ROOT = _REPO_ROOT / "assets" / "robot_description"


def get_robot_mjcf_path(robot_type: str) -> Path:
    if robot_type == "unitree_g1":
        return (
            _ROBOT_DESCRIPTION_ROOT
            / "newton-assets_unitree_g1_308a72cd"
            / "unitree_g1"
            / "mjcf"
            / "g1_29dof_rev_1_0.xml"
        )
    if robot_type == "unitree_h2":
        return _ROBOT_DESCRIPTION_ROOT / "mjcf" / "h2.xml"
    raise ValueError(f"Unknown robot type: {robot_type}")


def get_robot_urdf_path(robot_name: str) -> Path:
    if robot_name == "g1":
        return (
            _ROBOT_DESCRIPTION_ROOT
            / "newton-assets_unitree_g1_308a72cd"
            / "unitree_g1"
            / "urdf"
            / "g1_29dof_rev_1_0.urdf"
        )
    if robot_name == "h2":
        return _ROBOT_DESCRIPTION_ROOT / "urdf" / "h2" / "h2.urdf"
    raise ValueError(f"Unknown robot name: {robot_name}")
