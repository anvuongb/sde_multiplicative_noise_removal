import bm3d
import tqdm
from skimage.restoration import estimate_sigma
import argparse
import os
import cv2
from multiprocessing import Pool, cpu_count

def process(fn, img_out_path):
    no_rgb = False
    # denoise RGB
    if not no_rgb:
        img_noised = cv2.imread(fn)
        sigma = estimate_sigma(img_noised, channel_axis=-1)
        img_denoised = bm3d.bm3d(img_noised, sigma_psd=sigma, stage_arg=bm3d.BM3DStages.ALL_STAGES)
    # denoise grayscale
    else:
        img_noised = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
        sigma = estimate_sigma(img_noised)
        img_denoised = bm3d.bm3d(img_noised, sigma_psd=sigma, stage_arg=bm3d.BM3DStages.ALL_STAGES)
    cv2.imwrite(img_out_path, img_denoised)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Noise removal using BM3D", description="Noise removal using BM3D"
    )
    parser.add_argument("--no-rgb", action="store_true")
    parser.add_argument(
        "--img-dir", type=str, default="./datasets/img_celeba", required=True
    )
    parser.add_argument(
        "--img-out", type=str, required=True
    )
    parser.add_argument("--ddpm-target-steps", type=int, default=500)

    args = parser.parse_args()

    outdir = args.img_out + f"_bm3d_t{args.ddpm_target_steps}"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print(f"Generating denoised images from {args.img_dir}")
    print(f"Images will be saved to {outdir}")
    
    filenames = [x for x in os.listdir(args.img_dir) if x.endswith(".png")]

    input_list = [os.path.join(args.img_dir, x) for x in filenames]
    output_list = [os.path.join(outdir, x) for x in filenames]

    mapping = list(zip(input_list, output_list))
    with Pool(6) as pool:
        result = list(tqdm.tqdm(pool.starmap(process, mapping), total=len(output_list)))

    
