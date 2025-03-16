import os
from typing import List

import mediapy as media
import numpy as np
from moojoco.environment.base import MuJoCoEnvironmentConfiguration


def post_render(
        render_output: List[np.ndarray],
        environment_configuration: MuJoCoEnvironmentConfiguration
        ) -> np.ndarray:
    if render_output is None:
        # Temporary workaround until https://github.com/google-deepmind/mujoco/issues/1379 is fixed
        return None

    num_cameras = len(environment_configuration.camera_ids)
    num_envs = len(render_output) // num_cameras

    if num_cameras > 1:
        # Horizontally stack frames of the same environment
        frames_per_env = np.array_split(render_output, num_envs)
        render_output = [np.concatenate(env_frames, axis=1) for env_frames in frames_per_env]

    # Vertically stack frames of different environments
    render_output = np.concatenate(render_output, axis=0)

    return render_output[:, :, ::-1]  # RGB to BGR


def show_video(
        images: List[np.ndarray | None],
        sim_time: float,
        path: str | None = None
        ) -> str | None:
    if path:
        media.write_video(path=path, images=images)
        os.system("xdg-open {}".format(path))
    return media.show_video(images=images, fps=len(images)//sim_time)