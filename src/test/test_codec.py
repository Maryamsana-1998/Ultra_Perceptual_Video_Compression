import sys

if "./" not in sys.path:
    sys.path.append("./")
from utils.share import *
import utils.config as config

import einops
import numpy as np
import cv2
import torch
from pytorch_lightning import seed_everything
from annotator.util import HWC3
from models.util import create_model, load_state_dict
from models.ddim_hacked import DDIMSampler
from pathlib import Path
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

a_prompt = "best quality, extremely detailed"
n_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
num_samples = 1
image_resolution = 512
ddim_steps = 60
strength = 1
scale = 7.5
seed = 42
eta = 0.0
global_strength = 1

def process(
    model,
    canny_image,
    frame_image,
    prompt,
    a_prompt,
    n_prompt,
    num_samples,
    image_resolution,
    ddim_steps,
    strength,
    scale,
    seed,
    eta,
    global_strength
):

    seed_everything(seed)
    ddim_sampler = DDIMSampler(model)

    with torch.no_grad():
        W, H = image_resolution, image_resolution

        canny_image = cv2.resize(canny_image, (W, H))
        canny_detected_map = HWC3(canny_image)
        # print(canny_detected_map.shape)

        frame_map =  cv2.resize(HWC3(frame_image), (W, H))
        # print(frame_map.shape)

        content_emb = np.zeros((768))

        detected_maps = np.concatenate([canny_detected_map, frame_map], axis=2)

        local_control = torch.from_numpy(detected_maps.copy()).float().cuda() / 255.0
        local_control = torch.stack([local_control for _ in range(num_samples)], dim=0)
        local_control = einops.rearrange(local_control, "b h w c -> b c h w").clone()
        global_control = torch.from_numpy(content_emb.copy()).float().cuda().clone()
        global_control = torch.stack([global_control for _ in range(num_samples)], dim=0)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        uc_local_control = local_control
        uc_global_control = torch.zeros_like(global_control)
        cond = {
            "local_control": [local_control],
            "c_crossattn": [
                model.get_learned_conditioning([prompt + ", " + a_prompt] * num_samples)
            ],
            "global_control": [global_control],
        }
        un_cond = {
            "local_control": [uc_local_control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)],
            "global_control": [uc_global_control],
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength] * 13
        samples, _ = ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=True,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond,
            global_strength=global_strength,
        )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (
            (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
            .cpu()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )
        results = [x_samples[i] for i in range(num_samples)]

    return (results, [canny_detected_map, frame_map])

def get_recons_img(model,prompt, canny_image, frame_image):
    pred = process(model,canny_image,frame_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, strength, scale, seed, eta, global_strength)
    pred_img = pred[0][0]
    return pred_img


def load_images(image_paths, color_conversion=cv2.COLOR_BGR2RGB):
    images = []
    for path in image_paths:
        try:
            img = cv2.imread(path)
            if img is None:
                print(f"Warning: Failed to read image at {path}")
                continue
            img = cv2.cvtColor(img, color_conversion)
            images.append(img)
        except Exception as e:
            print(f"Error processing {path}: {e}")

    return images


def process_images(config_path, ckpt_path, image_paths, canny_paths, prompt, previous_frames_paths, pred_folder, gop=6):
    """
    Processes a given set of images, generates predictions, and calculates residues.

    Args:
    - config_path (str): Path to the model configuration file.
    - ckpt_path (str): Path to the model checkpoint file.
    - image_paths (list): List of paths to the original images.
    - canny_paths (list): List of paths to the Canny images.
    - prompt (str): The text prompt for processing.
    - previous_frames_paths (list): List of paths to the previous frame images.
    - pred_folder (str): Path to the folder for saving prediction images.
    - num_images (int): Number of images to process (default is 15).

    Returns:
    - original_images (list): List of the original images.
    - predictions (list): List of the predicted images.
    """
    # Load the model
    print(f"Loading model with config: {config_path} and checkpoint: {ckpt_path}")
    model = create_model(config_path).cpu()
    ckpt = load_state_dict(ckpt_path, location="cuda")
    # Filter out extra keys
    model_keys = set(model.state_dict().keys())
    filtered_ckpt = {k: v for k, v in ckpt.items() if k in model_keys}

    # Load the filtered state dict
    model.load_state_dict(filtered_ckpt)
    model = model.cuda()
    # Ensure the output directory exists
    os.makedirs(pred_folder, exist_ok=True)

    # Load original images, Canny images, and previous frame images
    original_images = load_images(image_paths)
    canny_images = load_images(canny_paths)
    previous_frames_paths_gop = [previous_frames_paths[i // gop] for i in range(len(image_paths))]
    previous_frames = load_images(previous_frames_paths_gop)
    predictions = []

    total_frames = len(image_paths)

    all_indices = list(range(total_frames))
    intra_indices = list(range(0, total_frames, gop))
    inter_indices = [i for i in all_indices if i not in intra_indices]

    # Process each image up to the specified number
    for i in range(0, len(original_images)):

        pred_image_path = os.path.join(pred_folder, f"im{(i + 1):05d}_pred.png")

        if i % gop == 0: 
            print('Intra Coded: ', previous_frames_paths_gop[i])
            pred_image = previous_frames[i]
            cv2.imwrite(pred_image_path, cv2.cvtColor(pred_image, cv2.COLOR_RGB2BGR))

        else:

            if Path(pred_image_path).exists():

                pred_image = cv2.imread(pred_image_path)
                pred_image = cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB)
                print('already done')
                continue
            else:
                try: 
                    print('Inter Coded with :', canny_paths[inter_indices.index(i)] ,previous_frames_paths_gop[i] )
                except Exception as e:
                    print('UNFINISHED', e,len(canny_paths), inter_indices.index(i) ,previous_frames_paths_gop[i], i )
                    continue

                canny_image = canny_images[inter_indices.index(i)]
                frame_image = previous_frames[i]
        
                pred_image = get_recons_img(
                    model,
                    prompt=prompt,
                    canny_image=canny_image,
                    frame_image=frame_image
                )
                cv2.imwrite(pred_image_path, cv2.cvtColor(pred_image, cv2.COLOR_RGB2BGR))
                print(f"Saved prediction image: {pred_image_path}")

            predictions.append(pred_image)
             
    # Return the original and predicted images
    return original_images, predictions
