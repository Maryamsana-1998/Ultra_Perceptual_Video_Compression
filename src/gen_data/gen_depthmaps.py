import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

# Setup device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_type = "DPT_Large"  # or DPT_Hybrid

midas = torch.hub.load("intel-isl/MiDaS", model_type).to(device).eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# Root directory
root = "/data2/local_datasets/vimeo_septuplet/sequences"
target_size = (512, 512)
batch_size = 4

def process_batch(image_paths, save_paths):
    batch_tensors = []
    # print(save_paths)

    for img_path in image_paths:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        input_tensor = transform(img).squeeze(0)
        batch_tensors.append(input_tensor)

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
        cv2.imwrite(save_paths[i], depth_uint8)
        print('saved')


# image_paths = []
# # Walk through 'depth' folders and process in batches
# for dirpath, dirnames, filenames in tqdm(os.walk(root), desc="Searching folders"):
#     if os.path.basename(dirpath) == "depth":
#         image_files = sorted([f for f in filenames if f.endswith(('.png', '.jpg'))])
#         if not image_files:
#             continue

#         image_paths.extend([os.path.join(dirpath, f) for f in image_files])


# save_paths = [os.path.join(dirpath, os.path.splitext(f)[0] + "_depth.png") for f in image_paths]
# print("*****************************************")
# print(' image and save paths ', len(image_paths), len(save_paths))
# print("*****************************************")
# print(' image and save paths ', image_paths[0:10], save_paths[0:10])



# for i in tqdm(range(0, len(image_paths), batch_size), desc=f"Processing {dirpath}"):
#     batch_imgs = image_paths[i:i+batch_size]
#     batch_saves = save_paths[i:i+batch_size]
#     process_batch(batch_imgs, batch_saves)
