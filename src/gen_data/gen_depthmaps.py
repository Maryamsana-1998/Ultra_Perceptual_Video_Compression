import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import glob

# Setup device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_type = "DPT_Large"  # or DPT_Hybrid

midas = torch.hub.load("intel-isl/MiDaS", model_type).to(device).eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


def process_batch(image_paths, save_paths):
    batch_tensors = []
    valid_save_paths = []

    for img_path, save_path in zip(image_paths, save_paths):
        if os.path.exists(save_path):
            print(f"Skipping {save_path} (already exists)")
            continue

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        input_tensor = transform(img).squeeze(0)
        batch_tensors.append(input_tensor)
        valid_save_paths.append(save_path)

    if not batch_tensors:
        print("All images in batch already processed.")
        return

    input_batch = torch.stack(batch_tensors).to(device)
    print('input shape:', input_batch.shape)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=target_size,
            mode="bicubic",
            align_corners=False
        )

    prediction = prediction.squeeze(1)

    for i, depth_map in enumerate(prediction):
        depth_np = depth_map.cpu().numpy()
        depth_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_norm.astype(np.uint8)
        cv2.imwrite(valid_save_paths[i], depth_uint8)
        print(f"Saved: {valid_save_paths[i]}")


# Root directory
base_dir = "/data/datasets/kinetics400/train/sequences_sampled"
target_size = (512, 512)
batch_size = 4
image_paths = []
save_paths = []


folders = glob.glob(base_dir+'/*/*')
# Walk through 'depth' folders and collect paths
for root in folders:
    files = os.listdir(root)
    if "r2.png" in files:
        folder_image_paths = sorted([
            os.path.join(root, f) for f in os.listdir(root)
            if f.endswith(('.png', '.jpg')) and f not in ['r1.png', 'r2.png']
        ])
        image_paths.extend(folder_image_paths)

        depth_dir = os.path.join(root, 'depth')
        os.makedirs(depth_dir, exist_ok=True)  # safer than mkdir

        folder_save_paths = [
            os.path.join(depth_dir, os.path.splitext(os.path.basename(f))[0] + "_depth.png")
            for f in folder_image_paths
        ]
        save_paths.extend(folder_save_paths)

# Debug info
print("*****************************************")
print(' image and save paths ', len(image_paths), len(save_paths))
print("*****************************************")
print(' image and save paths ', image_paths[:10], save_paths[:10])

# Batch processing
for i in tqdm(range(0, len(image_paths), batch_size), desc=f"Processing"):
    batch_imgs = image_paths[i:i+batch_size]
    batch_saves = save_paths[i:i+batch_size]
    process_batch(batch_imgs, batch_saves)

