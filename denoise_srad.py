import tqdm
import argparse
import os
import cv2
from multiprocessing import Pool
from srad import SRADRGB, SRAD

def process(fn, img_out_path):
    iterationMaxStep, timeSize, decayFactor = 200,.05,1
    no_rgb = False
    # denoise RGB
    if not no_rgb:
        img_noised = cv2.imread(fn)
        img_denoised = SRADRGB(img_noised, iterationMaxStep, timeSize, decayFactor)
    # denoise grayscale
    else:
        img_noised = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
        img_denoised = SRAD(img_noised, iterationMaxStep, timeSize, decayFactor)
    cv2.imwrite(img_out_path, img_denoised)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Noise removal using SRAD", description="Noise removal using SRAD"
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

    outdir = args.img_out + f"_srad_t{args.ddpm_target_steps}"
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

    
