# python src/vae/train_vae.py --tag test --learning_rate 1e-4 --max_train_steps 100 --checkpointing_steps 25 --wandb

import argparse
import logging
import math
import os
import random

import datasets
import diffusers
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers

# wandb
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm

from src.utils import normalize_img, prepare_image_obs


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # model
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='stabilityai/sd-vae-ft-ema', help="Path to pretrained model or model identifier from huggingface.co/models.",)
    # data
    parser.add_argument("--data_dir", type=str, default='vae_data', help="path to dataset",)
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--dataloader_num_workers", type=int, default=16, help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    # training
    parser.add_argument("--max_train_steps", type=int, default=25000, help="Total number of training steps to perform.")
    parser.add_argument("--checkpointing_steps", type=int, default=1000, help="Save checkpoint every X updates steps.")
    # eval
    parser.add_argument("--n_viz", type=int, default=16, help="number of images to qualitatively eval")
    parser.add_argument("--n_eval_times", type=int, default=100, help="number of times to eval for less noise")
    # optimizer
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--no_scale_lr", action="store_true", help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    # logging
    parser.add_argument("--tag", type=str, default=None, required=True, help='tag for output dir and wandb run')
    parser.add_argument("--output_dir", type=str, default="vae_output", help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--wandb", action='store_true', help="whether to log wandb")
    parser.add_argument("--tracker_project_name", type=str, default="diffusion-for-simulation", help="The `project_name` argument passed to tor.init_trackers")
    # misc
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    args.output_dir = os.path.join(args.output_dir, args.tag)
    assert args.n_viz <= args.batch_size

    return args


def set_up_main_process_logger(accelerator, logger):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


class VAEDataset(Dataset):
    def __init__(self, data_dir, resolution):
        self.data_dir = data_dir
        self.resolution = resolution
        self.files = [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir)) if f.endswith(".pt")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=True)  # Load 4D tensor: (N, H, W, C)
        # Randomly select one image
        img = data[torch.randint(0, data.shape[0], (1,)).item()]  # Select one image (H, W, C)
        img = prepare_image_obs(img, self.resolution)
        img = normalize_img(img)
        return img


def main(args):
    # setup
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        log_with='wandb' if args.wandb else None,
        project_config=accelerator_project_config,
    )
    set_up_main_process_logger(accelerator, logger)
    set_seed(args.seed)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        tracker_config = dict(vars(args))
        accelerator.init_trackers(
            args.tracker_project_name,
            tracker_config,
            init_kwargs={"wandb": {"name": args.tag}}
        )
        if args.wandb:
            wandb.define_metric('Steps')
            wandb.define_metric("*", step_metric="Steps")
    torch.backends.cuda.matmul.allow_tf32 = True

    # vae
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path).to(accelerator.device)
    vae.encoder.requires_grad_(False)
    vae.decoder.requires_grad_(True)

    # optimizer
    if not args.no_scale_lr:
        args.learning_rate *= args.batch_size * accelerator.num_processes
    optimizer = torch.optim.AdamW(vae.decoder.parameters(), lr=args.learning_rate)
    optimizer.zero_grad()
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # dataset
    dataset = VAEDataset(args.data_dir, resolution=args.resolution)
    data_len = len(dataset)
    eval_len = int(0.05 * data_len)
    train_len = data_len - eval_len
    train_dataset, eval_dataset = random_split(dataset, [train_len, eval_len])

    # dataloader
    def collate_fn(batch):
        return torch.stack(batch, dim=0)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        drop_last=False,
    )

    # save some qualitative eval images
    viz_indices = random.sample(range(len(eval_dataset)), args.n_viz)
    viz_batch = collate_fn([eval_dataset[i] for i in viz_indices]).to(accelerator.device)

    # accelerator prepare
    vae, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(vae, optimizer, train_dataloader, eval_dataloader, lr_scheduler)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_epochs = math.ceil(args.max_train_steps / len(train_dataloader))

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num videos = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed) = {args.batch_size * accelerator.num_processes}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            vae.train()

            # Convert images to latent space
            latents = vae.encode(batch).latent_dist.sample()
            pred_images = vae.decode(latents).sample.clamp(-1, 1)
            gt_images = batch.clamp(-1, 1)
            loss = F.mse_loss(pred_images.float(), gt_images.float(), reduction="mean")

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(vae.decoder.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # logging
            progress_bar.update(1)
            global_step += 1
            accelerator.log({"loss/train": loss.detach().item(), "Steps": global_step})

            if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                vae.eval()

                # save
                model_path = os.path.join(args.output_dir, f'vae_{global_step}.pth')
                torch.save(vae.decoder.state_dict(), model_path)
                logger.info(f'saved model to {model_path}')

                # evaluate quantitative
                # HACK: since each video samples only one frame, sample multiple times
                logger.info('evaluating model quantitative')
                eval_loss = 0.0
                for _ in tqdm(range(args.n_eval_times)):
                    for batch in eval_dataloader:
                        with torch.no_grad():
                            latents = vae.encode(batch).latent_dist.sample()
                            pred_images = vae.decode(latents).sample.clamp(-1, 1)
                        gt_images = batch.clamp(-1, 1)
                        eval_loss += F.mse_loss(pred_images.float(), gt_images.float(), reduction="mean").detach().item()
                eval_loss = eval_loss / len(eval_dataloader) / args.n_eval_times
                accelerator.log({"loss/eval": eval_loss, "Steps": global_step})

                # evaluate qualitative
                logger.info('evaluating model qualitative')
                latents = vae.encode(viz_batch).latent_dist.sample()
                pred_images = vae.decode(latents).sample.clamp(-1, 1)
                gt_images = viz_batch.clamp(-1, 1)
                tensor_to_pil = lambda x: transforms.ToPILImage()(x * 0.5 + 0.5)
                accelerator.log({
                    "pred_images": [wandb.Image(tensor_to_pil(img), caption=f'pred{i}') for i, img in enumerate(pred_images)],
                    'Steps': global_step,
                })
                if global_step == args.checkpointing_steps:
                    accelerator.log({
                        "gt_images": [wandb.Image(tensor_to_pil(img), caption=f'gt{i}') for i, img in enumerate(gt_images)],
                        'Steps': global_step,
                    })

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
