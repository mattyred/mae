# credits to https://github.com/IcarusWizard/MAE/blob/main/mae_pretrain.py

import os
import argparse
import math
import torch
import numpy as np
import torchvision
import wandb
import random
from einops import rearrange
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

from src.models.mae import MAE_ViT

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--max_device_batch_size', type=int, default=512)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=2000)
    parser.add_argument('--warmup_epoch', type=int, default=200)
    parser.add_argument('--model_path', type=str, default='ckpts/')

    args = parser.parse_args()

    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

     # --- wandb init (replaces SummaryWriter) ---
    wandb.init(
        project='masked-autoencoders',
        config={
            "seed": args.seed,
            "batch_size": args.batch_size,
            "max_device_batch_size": args.max_device_batch_size,
            "base_learning_rate": args.base_learning_rate,
            "weight_decay": args.weight_decay,
            "mask_ratio": args.mask_ratio,
            "total_epoch": args.total_epoch,
            "warmup_epoch": args.warmup_epoch,
            "model_path": args.model_path,
        },
    )

    model = MAE_ViT(mask_ratio=args.mask_ratio).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    step_count = 0
    optim.zero_grad()
    for e in range(args.total_epoch):
        model.train()
        losses = []
        for img, label in tqdm(iter(dataloader)):
            step_count += 1
            img = img.to(device)
            predicted_img, mask = model(img)
            loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        wandb.log(
            {
                "train/mae_loss": avg_loss,
                "train/lr": lr_scheduler.get_last_lr()[0]
            },
            step=e,
        )
        print(f'Epoch {e} average traning loss: {avg_loss:.4f}.')

        ''' visualize the first 16 predicted images on val dataset'''
        model.eval()
        with torch.no_grad():
            val_img = torch.stack([val_dataset[i][0] for i in range(16)])
            val_img = val_img.to(device)
            predicted_val_img, mask = model(val_img)
            predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
            img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
            img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
            img = (img + 1) / 2  # bring back to [0,1] for logging

            # --- log image grid to wandb (replaces writer.add_image) ---
            grid = img.detach().cpu().numpy()         # (C, H, W)
            grid = np.transpose(grid, (1, 2, 0))      # (H, W, C)
            wandb.log({"mae_image": wandb.Image(grid)}, step=e)
        
        ''' save model '''
        torch.save(model, args.model_path + f'mae_cifar10_epoch{e}.pth')