import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_path", type=Path)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--save", type=Path, default=None)
    args = parser.parse_args()

    episode = torch.load(
        args.episode_path,
        map_location="cpu",
        weights_only=True,
    )

    frames = episode["frames"]

    if not 0 <= args.frame_index < len(frames):
        raise IndexError(
            f"Frame index must be between 0 and {len(frames) - 1}, "
            f"but received {args.frame_index}."
        )

    frame = frames[args.frame_index].numpy()

    print(f"Episode: {args.episode_path}")
    print(f"All frames: {frames.shape}, {frames.dtype}")
    print(f"Selected frame: {frame.shape}, {frame.dtype}")
    print(f"State: {episode['states'][args.frame_index]}")

    plt.figure(figsize=(6, 6))
    plt.imshow(frame)
    plt.title(
        f"Episode {episode['episode_index']} — "
        f"frame {args.frame_index}"
    )
    plt.axis("off")
    plt.tight_layout()

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save, bbox_inches="tight", pad_inches=0)
        print(f"Saved image to {args.save}")

    plt.show()


if __name__ == "__main__":
    main()