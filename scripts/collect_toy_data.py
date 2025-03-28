import os
import random

import cv2
import einops
import numpy as np
import torch
from tqdm import tqdm


def main(args):
    # Canvas size
    os.makedirs(args.save_dir, exist_ok=True)
    for episode_id in tqdm(range(args.num_episodes), desc="Sampling episodes"):
        width, height = 256, 256

        # Create a black background (this will accumulate the line trace)
        background = np.full((height, width, 3), 255, dtype=np.uint8)

        # Starting position (center of the canvas)
        x, y = width // 2, height // 2
        old_x, old_y = width // 2, height // 2

        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            os.path.join(args.save_dir, f"episode_{episode_id}.mp4"),
            fourcc,
            15,
            (width, height),
        )

        # Possible moves: (dx, dy)
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        actions = [0, 1, 2, 3]
        total_steps = 250
        steps_completed = 0

        frames = []
        actions_record = []

        while steps_completed < total_steps:
            # Pick a random direction
            act = random.choice(actions)
            dx, dy = directions[act]
            dx *= 2
            dy *= 2
            # Pick how many consecutive steps to move in this direction
            steps_in_this_direction = random.randint(15, 15)
            # Adjust in case we're nearing the total step limit
            steps_to_execute = min(
                steps_in_this_direction, total_steps - steps_completed
            )

            for _ in range(steps_to_execute):
                # Calculate new position
                # Draw a thin line (slightly paler red) from old position to new position
                # (B, G, R) = (0, 0, 150) for a somewhat paler red
                cv2.line(background, (old_x, old_y), (x, y), (0, 0, 150), thickness=2)

                # Make a copy of the background to draw the bright red dot (0, 0, 255)
                frame = background.copy()

                cv2.circle(frame, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

                # Write the current frame to the video
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                frame_tensor = torch.from_numpy(frame)
                frames.append(frame_tensor)
                actions_record.append(act)
                old_x, old_y = x, y
                x = x + dx
                y = y + dy
                # Clamp to stay within the image bounds
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))

            steps_completed += steps_to_execute
        episode = {
            "observations": torch.stack(frames),
            "actions": torch.tensor(actions_record),
        }
        torch.save(episode, os.path.join(args.save_dir, f"episode_{episode_id}.pt"))

        out.release()


if __name__ == "__main__":

    import argparse

    argparse = argparse.ArgumentParser(description="Collect toy data")
    argparse.add_argument(
        "--num_episodes", type=int, default=1, help="Number of episodes to collect"
    )
    argparse.add_argument(
        "--save_dir",
        type=str,
        default="toy_data",
        help="Directory to save the collected data",
    )
    args = argparse.parse_args()
    main(args)
