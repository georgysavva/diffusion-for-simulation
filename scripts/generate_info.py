import json
import os

save_dir = "/data/georgy/dfs/data/toy/latent/test/"
episodes_info = {
    "episodes_num": 100,
    "episodes": [{"episode_id": i, "length": 250} for i in range(100)],
}
with open(os.path.join(save_dir, "episodes_info.json"), "w") as f:
    json.dump(episodes_info, f, indent=4)
