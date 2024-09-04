# from unet import UNet
from unet_ref import UNet
from model_helpers import save_model, load_model
from datasets_proc import (
    data_transform,
    CelebADataset,
    UCMercedLandUseDataset,
)
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from PIL import Image
import os
import torch
import numpy as np
import tqdm
import time
import wandb
import argparse


@torch.no_grad()
def save_image_2(
    tensor,
    fp,
    format=None,
    **kwargs,
) -> None:
    """
    Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    # tensor = (tensor + 1) * np.log(2) / 2 # rescale to [0, log2]
    tensor = torch.exp(tensor)  # rescale to [1, 2]
    tensor = tensor - 1  # rescale to [0, 1]
    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = (
        grid.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)


def get_noise_schedule(num_steps, start=0.0001, end=0.02):
    """
    Return linear noise schedule

    Args:
        num_steps (int): number of diffusion steps
        start (float): starting noise level
        end (float): ending noise level
    Return:
        list of noise levels
    """
    return torch.linspace(start, end, num_steps)


def forward_diffusion_gbm(log_x0, t, sigmas, device="cpu"):
    """
    Forward diffusion using Geometric Brownian Motion
    In logarithmic domain

    Args:
        log_x0 (Tensors): a batch of input images, in logarithmic domain
        t (Tensors): time step to diffuse
        sigmas (Tensors): noise levels

    Return:
        (Tensors) Diffused batch of images
    """
    noise = torch.randn_like(log_x0).to(device)

    scale_t = torch.gather(sigmas, 1, t)
    scale_t = scale_t[:, :, None, None]
    scale_0 = torch.gather(sigmas, 1, torch.zeros_like(t))
    scale_0 = scale_0[:, :, None, None]

    var = scale_t - scale_0
    mean = log_x0 - 0.5 * var
    # mean = log_x0
    log_xt = mean + torch.sqrt(var) * noise
    return log_xt.to(device), noise, var


@torch.no_grad()
def one_step_denoising(
    model,
    log_x,
    t,
    sigmas,
    device="cpu",
):
    """
    One step reverse diffusion (stochastic), calculate x_{t-1} from x_{t} and predicted score function

    Args:
        model (torch model): trained score/ddpm model
        log_x (Tensors): batch of input images at time t
        t (Tensors): diffusion step t
        sigmas (Tensors): noise levels

    Return:
        (Tensors) x_{t-1}, in log domain
    """
    scale_t = torch.gather(sigmas, 1, t + 1)
    scale_t = scale_t[:, :, None, None]

    scale_0 = torch.gather(sigmas, 1, torch.zeros_like(t))
    scale_0 = scale_0[:, :, None, None]

    scale_t_prev = torch.gather(sigmas, 1, t)
    scale_t_prev = scale_t_prev[:, :, None, None]

    z = torch.randn_like(log_x)

    noise_pred = model(log_x, t[0] + 1)

    var = scale_t - scale_0
    var_t = scale_t - scale_t_prev
    mean = log_x + 0.5 * var_t

    # denoising equation, note that this equation makes use of the relationship 
    # between SGMs and DDPMs
    log_x_out = mean - var_t / torch.sqrt(var) * noise_pred + torch.sqrt(var_t) * z

    return log_x_out.to(device)


@torch.no_grad()
def one_step_denoising_deterministic(
    model,
    log_x,
    t,
    sigmas,
    device="cpu",
):
    """
    One step reverse diffusion (ODE), calculate x_{t-1} from x_{t} and predicted score function

    Args:
        model (torch model): trained score/ddpm model
        log_x (Tensors): batch of input images at time t
        t (Tensors): diffusion step t
        sigmas (Tensors): noise levels

    Return:
        (Tensors) x_{t-1}, in log domain
    """
    scale_t = torch.gather(sigmas, 1, t + 1)
    scale_t = scale_t[:, :, None, None]

    scale_0 = torch.gather(sigmas, 1, torch.zeros_like(t))
    scale_0 = scale_0[:, :, None, None]

    scale_t_prev = torch.gather(sigmas, 1, t)
    scale_t_prev = scale_t_prev[:, :, None, None]

    noise_pred = model(log_x, t[0] + 1)

    var = scale_t - scale_0
    var_t = scale_t - scale_t_prev
    mean = log_x + 0.5 * var_t

    # ODE sampling
    log_x_out = mean - 0.5 * var_t / torch.sqrt(var) * noise_pred

    return log_x_out.to(device)


@torch.no_grad()
def one_step_denoising_ddim_deterministic(
    model,
    log_x,
    t,
    sigmas,
    device="cpu",
):
    """
    One step reverse diffusion (DDIM), calculate x_{t-1} from x_{t} and predicted score function

    Args:
        model (torch model): trained score/ddpm model
        log_x (Tensors): batch of input images at time t
        t (Tensors): diffusion step t
        sigmas (Tensors): noise levels

    Return:
        (Tensors) x_{t-1}, in log domain
    """
    scale_t = torch.gather(sigmas, 1, t + 1)
    scale_t = scale_t[:, :, None, None]

    scale_0 = torch.gather(sigmas, 1, torch.zeros_like(t))
    scale_0 = scale_0[:, :, None, None]

    scale_t_prev = torch.gather(sigmas, 1, t)
    scale_t_prev = scale_t_prev[:, :, None, None]

    noise_pred = model(log_x, t[0] + 1)

    var = scale_t - scale_0
    var_prev = scale_t_prev - scale_0

    # DDIM deterministic sampling
    # predicted x0
    log_x0_pred = log_x - torch.sqrt(var) * noise_pred + 0.5 * var

    # predict x_t-1
    log_x_out = (
        log_x0_pred
        - 0.5 * var_prev
        + torch.sqrt(var_prev / var) * (log_x - log_x0_pred + 0.5 * var)
    )

    return log_x_out.to(device)


@torch.no_grad()
def sample_noise_removal(
    x_0: torch.Tensor,
    model: UNet,
    timesteps: int,
    sigmas: torch.Tensor,
    device: str = "cpu",
    return_history=False,
    num_samples=1,
    smooth_indices=None,
    deterministic=False,
    ddim=False,
):
    """
    Generating clean images from noised ones

    Args:
        x_0 (Tensor): input batch of noisy images
        model (torch model): trained ddpm network
        timesteps (Tensor): number of steps to run reverse diffusion
        sigmas (Tensor): noise schedule
        device (str): "cpu" or "gpu"
        return_history (Boolean): return intermediate images or not
        num_samples (int): perform average (for stochastic only) using multiple samples
        smooth_indices (list of int): which samples to be averaged, if the previous option > 1
        deterministic (Boolean): perform deterministic sampling or not

    """
    x = x_0
    hist = [x]
    for idx, t_ in tqdm.tqdm(
        enumerate(reversed(range(timesteps - 1))), total=timesteps - 1
    ):
        t = torch.ones(size=(x.shape[0], 1), dtype=int) * t_
        t = t.to(device)
        x_list = []
        if smooth_indices is not None and idx in smooth_indices:
            for i in range(num_samples):
                if not deterministic:
                    x_ = one_step_denoising(model, x, t, sigmas, device=device)
                else:
                    x_ = one_step_denoising_deterministic(
                        model, x, t, sigmas, device=device
                    )
                x_list.append(x_)
                # if idx > timesteps-20 or (idx + 1) % 10 == 0:
                #     hist.append(x)
            x = torch.mean(torch.stack(x_list), 0)
        else:
            if not deterministic:
                x = one_step_denoising(model, x, t, sigmas, device=device)
            else:
                if ddim:
                    x = one_step_denoising_ddim_deterministic(
                        model, x, t, sigmas, device=device
                    )
                else:
                    x = one_step_denoising_deterministic(
                        model, x, t, sigmas, device=device
                    )
        hist.append(x)
    # x = torch.clamp(x, 0, np.log(2))
    if return_history:
        return x, hist
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Multiplicative noise removal", description="Multiplicative noise removal"
    )
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=28)
    parser.add_argument("--dataset", type=str, default="celeba")
    parser.add_argument(
        "--img-dir", type=str, default="./datasets/img_celeba"
    )
    parser.add_argument("--img-data-amount", type=str, default="tiny")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--exp-name", type=str, default="noise-removal")
    parser.add_argument("--cpu-workers", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--lr-scale-factor", type=float, default=0.7)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--load-model", action="store_true")
    parser.add_argument("--load-model-path", type=str, default="")
    parser.add_argument("--unet-channels", type=int, default=128)
    parser.add_argument("--unet-in-ch", type=int, default=1)
    parser.add_argument("--unet-out-ch", type=int, default=1)
    parser.add_argument("--unet-num-res", type=int, default=2)
    parser.add_argument("--ddpm-num-steps", type=int, default=1000)

    args = parser.parse_args()

    for i in range(torch.cuda.device_count()):
        print(
            "found GPU",
            torch.cuda.get_device_name(i),
            "capability",
            torch.cuda.get_device_capability(i),
        )
    rgb = True
    if args.unet_in_ch == 1:
        rgb = False

    # # Init dataset celeba
    img_size = args.img_size
    batch_size = args.batch_size
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
        shuffle=True,
        num_workers=args.cpu_workers,
        drop_last=True,
    )

    # Init model
    device = torch.device(args.device.split(",")[0])
    unet_conf = {
        "init_channels": args.unet_channels,
        "in_channels": args.unet_in_ch,
        "out_channels": args.unet_out_ch,
        "num_res_blocks": args.unet_num_res,
        "attn_resolutions": (16,),
        "input_img_resolution": img_size,
        "channels_multipliers": (1, 1, 2, 2, 4, 4),
    }
    model = UNet(**unet_conf).to(device)

    total_params = sum([p.numel() for p in model.parameters()])
    print("Model initialized, total params = ", total_params)

    # Init diffusion params
    T = args.ddpm_num_steps
    sigmas = get_noise_schedule(T)
    # repeat into batch_size for easier indexing by t
    sigmas = sigmas.unsqueeze(0).repeat(batch_size, 1).to(device)

    # torch.autograd.set_detect_anomaly(True)

    # # Get a sample batch
    x = next(iter(loader))
    x = x[0]
    t = torch.randint(low=1, high=T, size=(batch_size, 1))
    print(x.shape)
    # Exp name
    exp_name = args.exp_name
    if not os.path.exists(os.path.join("models", exp_name)):
        os.makedirs(os.path.join("models", exp_name))
    if not os.path.exists(f"./sampling_images/{exp_name}"):
        os.makedirs(f"./sampling_images/{exp_name}")
    current_time = time.time()

    # train params
    epochs = args.epochs
    save_every = args.save_every
    start_epoch = args.start_epoch
    load_model = args.load_model
    load_model_path = args.load_model_path

    # load from save
    if load_model:
        print(f"loading model from epoch {load_model_path}")
        model.load_state_dict(
            torch.load(
                load_model_path,
                map_location=device,
            )
        )
    print(args.device)
    device_list = args.device.split(":")[-1].split(",")
    device_list = [int(d) for d in device_list]
    print(device_list)
    model = torch.nn.DataParallel(model, device_ids=device_list)

    # re-init dataloader
    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.cpu_workers,
        drop_last=True,
    )

    # Init optimizer
    lr = args.lr
    factor = args.lr_scale_factor
    pt = args.lr_patience
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    opt.zero_grad()

    total = len(loader)
    print("num batches = ", total)
    wandb.init(
        project=exp_name,
        config={
            "dataset": args.dataset,
            "data_path": args.img_dir,
            "data_amount": args.img_data_amount,
            "devices": args.device,
            "load_model": load_model,
            "load_model_path": load_model_path,
            "lr": lr,
            "lr_scale_factor": factor,
            "lr_patience": pt,
            "epochs": epochs,
            "save_every_epochs": save_every,
            "start_epoch": start_epoch,
            "unet_config": unet_conf,
            "num_steps": T,
        },
    )

    for e in np.arange(start_epoch, start_epoch + epochs):
        loss_ema = None
        pbar = tqdm.tqdm(enumerate(loader), total=total)
        for idx, log_x in pbar:
            log_x = log_x[0]
            log_x = log_x.to(device)
            t = torch.randint(low=0, high=T, size=(batch_size, 1)).to(device)

            log_x_in, noise_in, var_in = forward_diffusion_gbm(
                log_x, t, sigmas, device=device
            )

            noise_pred = model(log_x_in, t.squeeze())
            loss = torch.nn.functional.mse_loss(noise_in, noise_pred, reduction="mean")
            loss.backward()

            if loss.item() == torch.nan:
                raise Exception("loss becomes nan")

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")

            opt.step()

            # log to wandb
            wandb.log({"loss": loss.item()}, step=e * total + idx)
            wandb.log(
                {"learning_rate": opt.param_groups[-1]["lr"]}, step=e * total + idx
            )

        # linear lrate decay
        opt.param_groups[0]["lr"] = lr * (1 - e / (start_epoch + epochs))

        print(f"epoch {e} loss={loss.item()}")

        # show image and save model every 100 epochs
        if e % save_every == 0 or e == epochs - 1:
            loader_test = DataLoader(
                data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=args.cpu_workers,
                drop_last=True,
            )
            x = next(iter(loader_test))
            log_x = x[0]

            log_x = log_x.to(device)
            t = torch.randint(low=0, high=T, size=(batch_size, 1)).to(device)
            t = t.to(device)
            log_x_in, noise_in, var_in = forward_diffusion_gbm(
                log_x, t, sigmas, device=device
            )

            num_show_images = 4
            save_model(f"models/{exp_name}/model_{e}.pkl", model, True)
            print("Generating sample images")
            save_image_2(
                log_x[:num_show_images],
                f"sampling_images/{exp_name}/original_epoch_{e}.png",
            )
            save_image_2(
                log_x_in[:num_show_images],
                f"sampling_images/{exp_name}/noised_epoch_{e}.png",
            )
            xs = []
            for j in range(num_show_images):  # use 4 images only
                log_x0 = log_x_in[j]
                log_x0 = log_x0[None, :, :, :]
                K = t[j][0].to("cpu").item()
                log_x_denoised = sample_noise_removal(
                    log_x0,
                    model,
                    K,
                    sigmas,
                    device=device,
                )
                log_x_denoised = log_x_denoised.to("cpu")
                xs.append(log_x_denoised)
            xsout = torch.cat(xs)
            save_image_2(xsout, f"sampling_images/{exp_name}/sample_epoch_{e}.png")
            save_image_2(xsout, f"sampling_images/{exp_name}/latest.png")
