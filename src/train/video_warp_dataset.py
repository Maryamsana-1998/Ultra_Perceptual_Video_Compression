import os
import random
import cv2
import albumentations as A
import numpy as np
import glob
from torch.utils.data import Dataset
from pathlib import Path
from annotator.content import ContentDetector

from .util import *
from utils.flow_utils import load_flo_file  

def normalize_for_warping(flow, target_shape=(128,128)):
    h, w = target_shape
    flow[..., 0] /= (w / 2)
    flow[..., 1] /= (h / 2)
    return np.transpose(flow, (2, 0, 1))  # [2, H, W]

def adaptive_weighted_downsample(flow, target_h=128, target_w=128):
    H, W = flow.shape[:2]
    output = np.zeros((target_h, target_w, 2), dtype=np.float32)

    # Compute bounds of each block
    h_bounds = np.linspace(0, H, target_h + 1, dtype=int)
    w_bounds = np.linspace(0, W, target_w + 1, dtype=int)

    for i in range(target_h):
        for j in range(target_w):
            h_start, h_end = h_bounds[i], h_bounds[i + 1]
            w_start, w_end = w_bounds[j], w_bounds[j + 1]

            block = flow[h_start:h_end, w_start:w_end]
            flat = block.reshape(-1, 2)

            # Weighted average by flow magnitude
            mag = np.linalg.norm(flat, axis=1)
            if mag.sum() > 0:
                weighted_avg = (flat * mag[:, None]).sum(axis=0) / (mag.sum() + 1e-6)
            else:
                weighted_avg = flat.mean(axis=0)  # fallback

            output[i, j] = weighted_avg

    return output  # shape: (128,128,2)


class UniDataset(Dataset):
    def __init__(self,
                 anno_path,
                 root_dir,
                 local_type_list,
                 resolution,
                 drop_txt_prob,
                 global_type_list,
                 keep_all_cond_prob,
                 drop_all_cond_prob,
                 drop_each_cond_prob,
                 transform):
     
     self.local_type_list = local_type_list
     self.global_type_list = global_type_list
     self.resolution = resolution
     self.drop_txt_prob = drop_txt_prob
     self.keep_all_cond_prob = keep_all_cond_prob
     self.drop_all_cond_prob = drop_all_cond_prob
     self.drop_each_cond_prob = drop_each_cond_prob

     self.sequences = glob.glob(root_dir+'/*/*')
     self.file_ids, self.annos = read_anno(anno_path)
     self.global_processor = ContentDetector()
     self.transform = transform
     self.aug_targets = {'target_image': 'image', 'intra_frame': 'I_frame'}

     if 'depth' in self.local_type_list:
        self.aug_targets = {'target_image': 'image', 'intra_frame': 'I_frame', 'depth_frame': 'depth'} 

     if 'r2' in self.global_type_list:
         self.aug_targets = {'target_image': 'image', 'global_frame': 'global'} 



     if self.transform:
        self.augmentation = A.Compose([
        # A.RandomCrop(height=512, width=512),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.4,
            p=1.0
        ),
        # A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT, p=1.0),
        # A.Affine(scale=(0.8, 1.2), translate_percent=(0.2, 0.2), p=1.0),
    ],  additional_targets=self.aug_targets,)

     self.video_frames = []
     for video_dir in self.sequences:
        frames = sorted([
                os.path.join(video_dir, f)
                for f in os.listdir(video_dir)
                if f.endswith(('.jpg', '.png')) and f not in ['r1.png','r2.png']
            ])
        self.video_frames.extend(frames)

    def __len__(self):
        return len(self.video_frames)
        
    def __getitem__(self, index):
        
        img_path = Path(self.video_frames[index])
        parts = os.path.normpath(img_path).split(os.sep)
        sequence_id = f"{parts[-3]}_{parts[-2]}"
        idx = self.file_ids.index(sequence_id)
        anno = self.annos[idx]

        # === Load input image ===
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # === Prepare local condition paths ===
        local_files = {}
        flow_path = None
        for local_type in self.local_type_list:
            if local_type == 'r1':
                local_files['r1'] = img_path.with_name('r1.png')
            elif local_type == 'depth':
                local_files['depth'] = img_path.parent / 'depth' / img_path.name.replace('.png', '_depth.png')
            elif local_type == 'flow':
                flow_path = img_path.parent / 'Flow' / img_path.name.replace('.png', '.flo')

        # === Load r1 only if exists (for augmentation) ===
        r1_img = None
        if 'r1' in local_files and local_files['r1'].exists():
            r1_img = cv2.imread(str(local_files['r1']))
            r1_img = cv2.cvtColor(r1_img, cv2.COLOR_BGR2RGB)

        # === Apply augmentations only on image + r1 ===
        if r1_img is not None and self.transform:
            augmented = self.augmentation(image=image, r1=r1_img)
            image = augmented['image']
            local_condition_map = {'r1': augmented['r1']}
        else:
            image = cv2.resize(image, (self.resolution, self.resolution))
            local_condition_map = {}

        image = (image.astype(np.float32) / 127.5) - 1.0  # Normalize to [-1,1]

        # === Load depth (no augmentation) ===
        if 'depth' in local_files and local_files['depth'].exists():
            depth_img = cv2.imread(str(local_files['depth']))
            depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2RGB)
            depth_img = cv2.resize(depth_img, (self.resolution, self.resolution))
            depth_img = depth_img.astype(np.float32) / 255.0
            local_condition_map['depth'] = depth_img

        # === Format local conditions ===
        local_conditions = list(local_condition_map.values())
        local_conditions = keep_and_drop(local_conditions, self.keep_all_cond_prob,
                                        self.drop_all_cond_prob, self.drop_each_cond_prob)
        if len(local_conditions):
            local_conditions = np.concatenate(local_conditions, axis=2)

        # === Global conditions (not augmented) ===
        global_conditions = []
        for global_type in self.global_type_list:
            if global_type == 'r2':
                r2_path = img_path.with_name('r2.png')
                if r2_path.exists():
                    r2_img = cv2.imread(str(r2_path))
                    r2_img = cv2.cvtColor(r2_img, cv2.COLOR_BGR2RGB)
                    global_conditions.append(self.global_processor(r2_img))
        global_conditions = keep_and_drop(global_conditions, self.keep_all_cond_prob,
                                        self.drop_all_cond_prob, self.drop_each_cond_prob)
        if len(global_conditions):
            global_conditions = np.concatenate(global_conditions)

        # === Flow (downsample + normalize) ===
        flow = None
        if flow_path and flow_path.exists():
            flow = load_flo_file(flow_path)
            flow = adaptive_weighted_downsample(flow, target_h=self.resolution, target_w=self.resolution)
            flow = normalize_for_warping(flow)

        if random.random() < self.drop_txt_prob:
            anno = ''

        return dict(
            jpg=image,
            txt=anno,
            local_conditions=local_conditions if len(local_conditions) else None,
            global_conditions=global_conditions if len(global_conditions) else None,
            flow=flow
        )
