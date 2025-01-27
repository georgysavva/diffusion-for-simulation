import json

# Load the JSON data
with open(
    "/scratch/gs4288/shared/diffusion_for_simulation/data/mario/original/test/episodes_info.json"
) as f:
    data = json.load(f)

# Initialize variables to track the smallest length and corresponding episode
min_length = float("inf")
min_episode = None

# Iterate through the episodes to find the one with the smallest length
for episode in data["episodes"]:
    if episode["length"] < min_length:
        min_length = episode["length"]
        min_episode = episode

# Print the episode with the smallest length
print(f"Episode with the smallest length: {min_episode}")
