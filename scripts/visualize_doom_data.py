import os
from PIL import Image
from torchvision.utils import make_grid
from diffusers import AutoencoderKL
import torch
from torch.utils.data import DataLoader
from src.data import (
    BatchSampler,
    Dataset,
    collate_segments_to_batch,
)


data_path = '/scratch/gs4288/shared/diffusion_for_simulation/data/doom/latent_act_repeat/train'
out_dir = 'temp'
os.makedirs(out_dir, exist_ok=True)
device = torch.device('cuda')



vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
vae.decoder.load_state_dict(torch.load('/scratch/gs4288/shared/diffusion_for_simulation/vae/trained_vae_decoder.pth', weights_only=True, map_location=device))
vae.eval()


action_to_name = ["TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD", "MOVE_LEFT", "MOVE_RIGHT", "NOOP"]
train_dataset = Dataset(data_path, cache_in_ram=False)
batch_sampler = BatchSampler(
    train_dataset,
    rank=0,
    world_size=1,
    batch_size=32,
    seq_length=50,
    guarantee_full_seqs=True,
)
data_loader_train = DataLoader(
    dataset=train_dataset,
    collate_fn=collate_segments_to_batch,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    pin_memory_device=str(device),
    batch_sampler=batch_sampler,
)
batch = next(iter(data_loader_train))
obs, acts = batch.obs.to(device), batch.act.to(device)

# for single_obs, single_acts in zip(obs, acts):
#     for i, act in enumerate(single_acts):
#         if action_to_name[act] == 'NOOP' and i < len(single_obs) - 1:
#             stacked_obs = torch.stack([single_obs[i], single_obs[i+1]])
#             out_obs = vae.decode(stacked_obs / 0.18215).sample.clamp(-1, 1)
#             out_obs = ((out_obs * 0.5 + 0.5) * 255.0).clamp(0, 255).byte()



with torch.no_grad():
    num_frame = obs.shape[1]
    decoded_samples = []
    for frame_i in range(num_frame):
        decoded_sample = vae.decode(obs[:, frame_i] / 0.18215).sample.clamp(-1, 1)
        decoded_samples.append(decoded_sample)
    decoded_samples = torch.stack(decoded_samples, dim=1)
    decoded_samples = ((decoded_samples * 0.5 + 0.5) * 255.0).clamp(0, 255).byte()

for sample_i, sample in enumerate(decoded_samples):
    grid = make_grid(sample, nrow=sample.shape[0], padding=2)
    image = Image.fromarray(grid.permute(1, 2, 0).cpu().numpy())
    image.save(os.path.join(out_dir, f'{sample_i}.jpg'))
    action_names = [action_to_name[action] for action in acts[sample_i].tolist()]
    print(f'action{sample_i}:', action_names)
