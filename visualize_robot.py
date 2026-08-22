#!/usr/bin/env python3
"""Visualize Unitree G1/H2 retarget CSV files with viser."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    import trimesh
    import viser
    import yourdfpy
except ImportError as exc:
    raise SystemExit(
        "Missing visualization dependency. Install viser, yourdfpy and trimesh in the active environment."
    ) from exc

from soma_retargeter.utils.robot_description import get_robot_urdf_path


ROOT_COLUMNS = [
    "root_translateX",
    "root_translateY",
    "root_translateZ",
    "root_rotateX",
    "root_rotateY",
    "root_rotateZ",
]


class RobotAnimationApp:
    def __init__(self, urdf_path: Path, port: int = 8080):
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        self.urdf_path = urdf_path
        self.port = port
        self.urdf = yourdfpy.URDF.load(str(urdf_path), load_meshes=True, build_scene_graph=True)
        self.server = viser.ViserServer(port=port, host="0.0.0.0")
        self.server.scene.set_up_direction("+z")
        self.server.scene.add_grid("/grid", width=2.0, height=2.0)
        self.server.initial_camera.position = (1.0, -1.5, 1.0)
        self.server.initial_camera.look_at = (0.0, 0.0, 0.5)

        self.link_frames = {}
        self.joint_names = []
        self.joint_limits = {}
        self.cfg = np.zeros(0, dtype=float)
        self.root_transform = np.eye(4, dtype=float)
        self.csv_column_indices: dict[str, int] = {}
        self.motions: list[list[str]] = []
        self.playing = False
        self.current_frame = 0.0
        self.playback_speed = 1.0

        self._load_scene()
        self._load_joints()
        self._setup_ui()

    def _load_scene(self) -> None:
        mesh_count = 0
        urdf_dir = self.urdf_path.parent

        for link in self.urdf.robot.links:
            self.link_frames[link.name] = self.server.scene.add_frame(
                f"/robot/{link.name}",
                wxyz=(1.0, 0.0, 0.0, 0.0),
                position=(0.0, 0.0, 0.0),
                axes_length=0.04,
                axes_radius=0.002,
            )

            for visual_idx, visual in enumerate(link.visuals or []):
                mesh = visual.geometry.mesh
                if mesh is None or not mesh.filename:
                    continue

                mesh_path = self._resolve_mesh_path(mesh.filename, urdf_dir)
                if mesh_path is None:
                    print(f"Warning: mesh not found: {mesh.filename}")
                    continue

                loaded_mesh = trimesh.load(str(mesh_path), force="mesh")
                vertices = np.asarray(loaded_mesh.vertices, dtype=float)
                faces = np.asarray(loaded_mesh.faces, dtype=np.uint32)
                if len(vertices) == 0 or len(faces) == 0:
                    continue

                origin = visual.origin
                self.server.scene.add_mesh_simple(
                    f"/robot/{link.name}/mesh_{visual_idx}",
                    vertices=vertices,
                    faces=faces,
                    color=(150, 150, 200),
                    position=tuple(float(v) for v in origin[:3, 3]),
                    wxyz=self._matrix_to_wxyz(origin),
                )
                mesh_count += 1

        print(f"Loaded URDF {self.urdf_path}")
        print(f"Loaded {len(self.link_frames)} link frames and {mesh_count} meshes")

    @staticmethod
    def _resolve_mesh_path(mesh_filename: str, urdf_dir: Path) -> Path | None:
        if mesh_filename.startswith("file://"):
            mesh_filename = mesh_filename[len("file://") :]

        candidates: list[Path] = []
        if mesh_filename.startswith("package://"):
            package_rel = Path(mesh_filename[len("package://") :])
            candidates.extend(parent / package_rel for parent in [urdf_dir, *urdf_dir.parents])
        else:
            rel = Path(mesh_filename)
            candidates.extend([urdf_dir / rel, urdf_dir.parent / rel])

        return next((path for path in candidates if path.exists()), None)

    def _load_joints(self) -> None:
        for joint in self.urdf.robot.joints:
            if joint.type not in {"revolute", "continuous", "prismatic"}:
                continue
            self.joint_names.append(joint.name)
            if joint.limit is None:
                self.joint_limits[joint.name] = (-np.pi, np.pi)
            else:
                self.joint_limits[joint.name] = (float(joint.limit.lower), float(joint.limit.upper))

        self.cfg = np.zeros(len(self.joint_names), dtype=float)
        print(f"Loaded {len(self.joint_names)} actuated joints")

    def _setup_ui(self) -> None:
        with self.server.gui.add_folder("Playback Control"):
            self.frame_slider = self.server.gui.add_slider("Frame", min=0.0, max=0.0, step=1.0, initial_value=0.0)
            self.play_button = self.server.gui.add_checkbox("Play", initial_value=False)
            self.speed_slider = self.server.gui.add_slider("Speed", min=0.1, max=3.0, step=0.1, initial_value=1.0)
            self.status_text = self.server.gui.add_text("Status", initial_value="No motion loaded", disabled=True)

        with self.server.gui.add_folder("Joint State", expand_by_default=False):
            self.joint_sliders = {}
            for name in self.joint_names:
                lower, upper = self.joint_limits[name]
                slider = self.server.gui.add_slider(name, min=lower, max=upper, step=0.001, initial_value=0.0)
                self.joint_sliders[name] = slider
                slider.on_update(lambda _, n=name: self._on_joint_update(n))

        self.play_button.on_update(lambda _: self._on_play_toggle())
        self.speed_slider.on_update(lambda _: self._on_speed_update())
        self.frame_slider.on_update(lambda _: self._on_frame_update())

    def _on_joint_update(self, joint_name: str) -> None:
        self.cfg[self.joint_names.index(joint_name)] = float(self.joint_sliders[joint_name].value)
        self._update_robot()

    def _on_play_toggle(self) -> None:
        self.playing = bool(self.play_button.value)

    def _on_speed_update(self) -> None:
        self.playback_speed = float(self.speed_slider.value)

    def _on_frame_update(self) -> None:
        if self.motions:
            self._show_frame(int(self.frame_slider.value))

    @staticmethod
    def _matrix_to_wxyz(matrix: np.ndarray) -> tuple[float, float, float, float]:
        quat_xyzw = R.from_matrix(matrix[:3, :3]).as_quat()
        return (float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]))

    def _set_root_transform_from_row(self, row: list[str]) -> None:
        tx, ty, tz = (float(row[self.csv_column_indices[name]]) * 0.01 for name in ROOT_COLUMNS[:3])
        rx, ry, rz = (float(row[self.csv_column_indices[name]]) for name in ROOT_COLUMNS[3:])
        self.root_transform = np.eye(4, dtype=float)
        self.root_transform[:3, :3] = R.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
        self.root_transform[:3, 3] = [tx, ty, tz]

    def _joint_csv_column(self, joint_name: str) -> int | None:
        candidates = [
            f"{joint_name}_dof",
            joint_name.replace("_joint", "") + "_dof",
            joint_name.replace("_joint", "_joint_dof"),
        ]
        return next((self.csv_column_indices[name] for name in candidates if name in self.csv_column_indices), None)

    def _update_robot(self) -> None:
        self.urdf.update_cfg(self.cfg)
        for link_name, frame in self.link_frames.items():
            transform = self.urdf.get_transform(link_name)
            if transform is None:
                continue
            world_transform = self.root_transform @ transform
            frame.position = tuple(float(v) for v in world_transform[:3, 3])
            frame.wxyz = self._matrix_to_wxyz(world_transform)

    def _show_frame(self, frame_idx: int) -> None:
        frame_idx = max(0, min(frame_idx, len(self.motions) - 1))
        self.current_frame = float(frame_idx)
        row = self.motions[frame_idx]
        cfg = np.zeros(len(self.joint_names), dtype=float)
        self._set_root_transform_from_row(row)

        for idx, joint_name in enumerate(self.joint_names):
            column = self._joint_csv_column(joint_name)
            if column is not None:
                cfg[idx] = np.radians(float(row[column]))

        self.cfg = cfg
        self._update_robot()
        for idx, name in enumerate(self.joint_names):
            self.joint_sliders[name].value = float(cfg[idx])
        self.status_text.value = f"Frame {frame_idx}/{len(self.motions) - 1}"
        self.frame_slider.value = float(frame_idx)

    def load_motion(self, csv_file: Path) -> bool:
        with csv_file.open(newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        if not rows:
            print(f"CSV file is empty: {csv_file}")
            return False

        bad_row = next((idx + 2 for idx, row in enumerate(rows) if len(row) != len(header)), None)
        if bad_row is not None:
            print(f"CSV row {bad_row} has a different column count than the header")
            return False

        missing_root = [name for name in ROOT_COLUMNS if name not in header]
        if missing_root:
            print(f"Missing root pose columns: {missing_root}")
            return False

        self.csv_column_indices = {name: idx for idx, name in enumerate(header)}
        missing_joints = [name for name in self.joint_names if self._joint_csv_column(name) is None]
        if missing_joints:
            print(f"Missing CSV columns for {len(missing_joints)} URDF joints: {missing_joints}")
            return False

        self.motions = rows
        self.frame_slider.max = float(len(rows) - 1)
        self.frame_slider.value = 0.0
        self.status_text.value = f"Loaded {len(rows)} frames"
        self._show_frame(0)

        print(f"Loaded motion {csv_file}")
        print(f"Mapped {len(self.joint_names)} joints from {len(header)} CSV columns")
        return True

    def run(self) -> None:
        print(f"Open http://localhost:{self.port}")
        try:
            while True:
                if self.playing and self.motions:
                    self.current_frame = (self.current_frame + self.playback_speed) % len(self.motions)
                    self._show_frame(int(self.current_frame))
                time.sleep(0.03)
        except KeyboardInterrupt:
            print("Exiting")


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize G1/H2 retarget CSV with viser")
    parser.add_argument("robot", choices=["g1", "h2"], help="Robot model to visualize")
    parser.add_argument("csv", type=Path, help="CSV file or folder to replay")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    urdf_path = get_robot_urdf_path(args.robot)
    csv_path = args.csv
    if not csv_path.exists():
        print(f"CSV path not found: {csv_path}")
        return 1

    csv_file = csv_path if csv_path.is_file() else next(iter(sorted(csv_path.glob("*.csv"))), None)
    if csv_file is None:
        print(f"No CSV files found in {csv_path}")
        return 1

    app = RobotAnimationApp(urdf_path, args.port)
    if not app.load_motion(csv_file):
        return 1

    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
