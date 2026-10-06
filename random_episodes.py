from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch


ENV_ID = "PointMaze_UMaze-v3"
IMAGE_SIZE = 224


def create_environment(steps_per_episode: int):
    gym.register_envs(gymnasium_robotics)

    return gym.make(
        ENV_ID,
        render_mode="rgb_array",
        width=IMAGE_SIZE,
        height=IMAGE_SIZE,
        continuing_task=True,
        reset_target=True,
        max_episode_steps=steps_per_episode,
    )


def hide_goal_marker(env) -> None:
    """Make the red target invisible without changing the dynamics."""
    point_maze = env.unwrapped
    point_maze.point_env.model.site_rgba[
        point_maze.target_site_id, 3
    ] = 0.0


def validate_frame(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)

    expected_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
    if frame.shape != expected_shape:
        raise RuntimeError(
            f"Expected frame shape {expected_shape}, got {frame.shape}"
        )

    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    return np.ascontiguousarray(frame)


def collect_episode(
    env,
    episode_index: int,
    steps_per_episode: int,
    seed: int,
    show_goal: bool,
) -> dict:
    observation, _ = env.reset(seed=seed)

    breakpoint()
    if not show_goal:
        hide_goal_marker(env)

    # T + 1 values: the initial observation and one after every action.
    frames = [validate_frame(env.render())]
    states = [
        np.asarray(observation["observation"], dtype=np.float32).copy()
    ]

    # T values: one action causes each state transition.
    actions = []

    rng = np.random.default_rng(seed)

    for step_index in range(steps_per_episode):
        action = rng.uniform(
            low=env.action_space.low,
            high=env.action_space.high,
        ).astype(np.float32)

        observation, _, terminated, truncated, _ = env.step(action)

        actions.append(action)
        states.append(
            np.asarray(
                observation["observation"],
                dtype=np.float32,
            ).copy()
        )
        frames.append(validate_frame(env.render()))

        ended_early = (
            terminated or truncated
        ) and step_index < steps_per_episode - 1

        if ended_early:
            raise RuntimeError(
                f"Episode {episode_index} ended after "
                f"{step_index + 1} steps instead of "
                f"{steps_per_episode}."
            )

    episode = {
        "frames": torch.from_numpy(np.stack(frames)),
        "actions": torch.from_numpy(np.stack(actions)),
        "states": torch.from_numpy(np.stack(states)),
        "desired_goal": torch.as_tensor(
            observation["desired_goal"],
            dtype=torch.float32,
        ),
        "episode_index": episode_index,
        "seed": seed,
    }

    validate_episode(episode, steps_per_episode)
    return episode


def validate_episode(episode: dict, steps_per_episode: int) -> None:
    expected_frame_shape = (
        steps_per_episode + 1,
        IMAGE_SIZE,
        IMAGE_SIZE,
        3,
    )

    assert episode["frames"].shape == expected_frame_shape
    assert episode["frames"].dtype == torch.uint8

    assert episode["actions"].shape == (steps_per_episode, 2)
    assert episode["actions"].dtype == torch.float32

    assert episode["states"].shape == (steps_per_episode + 1, 4)
    assert episode["states"].dtype == torch.float32


def save_episode(episode: dict, output_path: Path) -> None:
    # Save to a temporary file first so an interrupted write does not
    # leave an apparently valid episode file.
    temporary_path = output_path.with_suffix(".tmp")
    torch.save(episode, temporary_path)
    temporary_path.replace(output_path)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/pointmaze/raw"),
    )
    parser.add_argument("--num-episodes", type=int, default=2000)
    parser.add_argument("--steps-per-episode", type=int, default=100)
    parser.add_argument("--training-frame-skip", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument(
        "--show-goal",
        action="store_true",
        help="Keep the red target visible in the rendered frames.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip episode files that already exist.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.steps_per_episode % args.training_frame_skip != 0:
        raise ValueError(
            "steps-per-episode must be divisible by training-frame-skip"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    existing_files = list(args.output_dir.glob("episode_*.pt"))
    if existing_files and not args.resume:
        raise FileExistsError(
            f"{args.output_dir} already contains episode files. "
            "Use --resume to continue without overwriting them."
        )

    env = create_environment(args.steps_per_episode)

    # Obtain the actual simulated time advanced by env.step().
    # control_dt = float(env.unwrapped.point_env.dt)

    metadata = {
        "environment": ENV_ID,
        "num_episodes": args.num_episodes,
        "steps_per_episode": args.steps_per_episode,
        "training_frame_skip": args.training_frame_skip,
        "image_shape": [IMAGE_SIZE, IMAGE_SIZE, 3],
        "action_dimension": 2,
        "state_dimension": 4,
        "base_seed": args.base_seed,
        "goal_visible": args.show_goal,
        # "control_dt_seconds": control_dt,
        # "control_frequency_hz": 1.0 / control_dt,
    }

    manifest_path = args.output_dir / "metadata.json"

    if args.resume and manifest_path.exists():
        previous_metadata = json.loads(manifest_path.read_text())

        if previous_metadata != metadata:
            raise RuntimeError(
                "The existing dataset was generated with different "
                "settings. Use another output directory."
            )
    else:
        manifest_path.write_text(json.dumps(metadata, indent=2))

    # print(f"Environment control dt: {control_dt:.4f} seconds")
    # print(f"Control frequency: {1.0 / control_dt:.2f} Hz")

    try:
        for episode_index in range(args.num_episodes):
            output_path = (
                args.output_dir / f"episode_{episode_index:04d}.pt"
            )

            if args.resume and output_path.exists():
                continue

            seed = args.base_seed + episode_index

            episode = collect_episode(
                env=env,
                episode_index=episode_index,
                steps_per_episode=args.steps_per_episode,
                seed=seed,
                show_goal=args.show_goal,
            )

            save_episode(episode, output_path)

            if (episode_index + 1) % 10 == 0:
                print(
                    f"Saved {episode_index + 1}/"
                    f"{args.num_episodes} episodes"
                )
    finally:
        env.close()


if __name__ == "__main__":
    main()