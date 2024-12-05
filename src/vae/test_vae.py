# python src/vae/test_vae.py

from diffusers import AutoencoderKL
import torch
from PIL import Image
from torchvision import transforms
import copy


# Load the VAE model
vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema').cuda()
# vae = AutoencoderKL.from_pretrained('pt-sk/stable-diffusion-1.5', subfolder='vae').cuda()

saved_params = {}
for n, p in vae.named_parameters():
    saved_params[n] = copy.deepcopy(p.data)

vae.decoder.load_state_dict(torch.load('vae_output/1128_0_train_vae_lr3e-6/vae_25000.pth', weights_only=True))
for n, p in vae.named_parameters():
    if 'encoder' in n.lower():
        assert torch.equal(saved_params[n], p.data)
    elif 'decoder' in n.lower():
        assert not torch.equal(saved_params[n], p.data)
    else:
        print(n)
        print(torch.equal(saved_params[n], p.data))

# Preprocess the input image
image_path = 'image.png'
image = Image.open(image_path).convert('RGB')
preprocess = transforms.ToTensor()
image_tensor = preprocess(image).unsqueeze(0).to('cuda')
image_tensor = (image_tensor - 0.5) / 0.5

# Encode the image
with torch.no_grad():
    latents = vae.encode(image_tensor).latent_dist.sample()
    decoded_image = vae.decode(latents).sample.clamp(-1, 1)[0]

# Convert to PIL image and save
decoded_image = (decoded_image + 1) / 2  # Transform from [-1, 1] to [0, 1]
print(decoded_image.shape)
transforms.ToPILImage()(decoded_image).save('decoded_image.jpg')
