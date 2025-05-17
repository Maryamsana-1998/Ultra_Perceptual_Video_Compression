import os
import random
import cv2
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
                 drop_each_cond_prob):
     
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
        
        try:
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.resolution, self.resolution))
            image = (image.astype(np.float32) / 127.5) - 1.0

        except Exception as e:
            print('error: ',e,img_path)
            raise e  

        # needs to expanded
        global_files = []
        for global_type in self.global_type_list:
            if global_type == 'r2':
                global_files.append(img_path.with_name('r2.png'))

        local_files = []
        flow_path = None
        for local_type in self.local_type_list:
           if local_type =='r1':  
              local_files.append(img_path.with_name('r1.png'))
           if local_type == 'flow':
              flow_path  = img_path.parent / 'Flow' / img_path.name.replace('.png', '.flo')
           if local_type == 'depth':
              new_path = img_path.parent / 'depth' / img_path.name.replace('.png', '_depth.png')
              local_files.append(new_path)


        local_conditions = []
        for local_file in local_files: 
            condition = cv2.imread(str(local_file))
            try:    
                condition = cv2.cvtColor(condition, cv2.COLOR_BGR2RGB)
                condition = cv2.resize(condition, (self.resolution, self.resolution))
                condition = condition.astype(np.float32) / 255.0
                local_conditions.append(condition)
            except Exception as e:
                print('missing', e,  local_file)
                raise e
            
        global_conditions = []
        for global_file in global_files:
            global_img = cv2.imread(global_file)
            global_img = cv2.cvtColor(global_img, cv2.COLOR_BGR2RGB)
            condition = self.global_processor(global_img)
            global_conditions.append(condition)

        if random.random() < self.drop_txt_prob:
            anno = ''
        
        local_conditions = keep_and_drop(local_conditions, self.keep_all_cond_prob, self.drop_all_cond_prob, self.drop_each_cond_prob)
        global_conditions = keep_and_drop(global_conditions, self.keep_all_cond_prob, self.drop_all_cond_prob, self.drop_each_cond_prob)
        
        if len(local_conditions) != 0:
            local_conditions = np.concatenate(local_conditions, axis=2)
        if len(global_conditions) != 0:
            global_conditions = np.concatenate(global_conditions)

        if flow_path is not None:
            flow = load_flo_file(flow_path)
            flow = adaptive_weighted_downsample(flow, target_h=128, target_w=128)
            flow = normalize_for_warping(flow)

        return dict(jpg=image, txt=anno, local_conditions=local_conditions, flow=flow, global_conditions=global_conditions)
           