import os
import torch
import tqdm
import argparse
from torch.utils.data import DataLoader
from datasets_proc import data_transform, CelebADataset, UCMercedLandUseDataset
from train_denoise_sde import forward_diffusion_gbm, get_noise_schedule, save_image_2

# given a time step t
# run forward diffusion to t
# then save images

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Generate noisy images", description="Generate noisy images"
    )
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=28)
    parser.add_argument("--no-rgb", action="store_true")
    parser.add_argument("--dataset", type=str, default="celeba", required=True)
    parser.add_argument(
        "--img-dir", type=str, default="datasets/img_celeba", required=True
    )
    parser.add_argument(
        "--img-out", type=str, required=True
    )
    parser.add_argument("--cpu-workers", type=int, default=5)
    parser.add_argument("--ddpm-num-steps", type=int, default=500)
    parser.add_argument("--ddpm-target-steps", type=int, required=True)
    parser.add_argument("--img-data-amount", type=str, default="test")


    args = parser.parse_args()

    device = "cuda:0"

    # Init diffusion params
    batch_size = args.batch_size
    T = args.ddpm_num_steps
    sigmas = get_noise_schedule(T)
    # repeat into batch_size for easier indexing by t
    sigmas = sigmas.unsqueeze(0).repeat(batch_size, 1).to(device)

    # init output folder
    if not os.path.exists(args.img_out):
        os.makedirs(args.img_out)

    # init dataset
    img_size = args.img_size
    batch_size = args.batch_size
    rgb = True
    if args.no_rgb:
        rgb = False
    data_transform = data_transform(img_size=img_size, rgb=rgb)
    img_dir = args.img_dir

    if args.dataset == "celeba":
        data = CelebADataset(
            img_dir=img_dir, transform=data_transform, amount=args.img_data_amount
        )
    elif args.dataset == "landuse":
        data = UCMercedLandUseDataset(
            img_dir=img_dir, transform=data_transform, amount=args.img_data_amount
        )
    else:
        raise ValueError(f"{args.dataset} not recognized")

    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.cpu_workers,
        drop_last=True,
    )

    total = len(loader)

    print(f"Generating noisy images from {args.img_dir}")
    print(f"Dataset={args.dataset}, DDPM T={args.ddpm_num_steps}, t={args.ddpm_target_steps}")
    print(f"Images will be saved to {args.img_out}")

    pbar = tqdm.tqdm(enumerate(loader), total=total)
    for idx, log_x in pbar:
        log_x = log_x[0]
        log_x = log_x.to(device)
        t = torch.ones(size=(batch_size, 1), dtype=torch.int64).to(device)
        t = t * args.ddpm_target_steps

        log_x_in, noise_in, var_in = forward_diffusion_gbm(
            log_x, t, sigmas, device=device
        )

        for idx_, im in enumerate(log_x_in):
            num = idx * batch_size + idx_
            path = os.path.join(args.img_out, f"{num}.png")
            save_image_2(im, path)