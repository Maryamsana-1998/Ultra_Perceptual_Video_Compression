import os
import random
import cv2
import numpy as np
import glob
from torch.utils.data import Dataset
from pathlib import Path

from .util import *
from utils.flow_utils import load_flo_file  

def process_flow_with_scaling(flo_path, target_shape=(128,128)):
    """
    Resize optical flow to a new shape and scale the motion vectors accordingly.

    Args:
        flow: np.ndarray of shape (H, W, 2) — optical flow in pixel units
        target_shape: tuple (new_H, new_W) — desired output shape

    Returns:
        resized_flow: np.ndarray of shape (new_H, new_W, 2)
    """
    flow = load_flo_file(flo_path)
    H, W = flow.shape[:2]
    new_H, new_W = target_shape

    # Resize both flow channels
    resized_flow = cv2.resize(flow, (new_W, new_H), interpolation=cv2.INTER_LINEAR)

    # Scale vector magnitudes to match new resolution
    scale_x = new_W / W
    scale_y = new_H / H
    resized_flow[..., 0] *= scale_x
    resized_flow[..., 1] *= scale_y
    # Normalize flow to [-1, 1] for grid_sample or softsplat
    resized_flow[..., 0] /= (new_W / 2)
    resized_flow[..., 1] /= (new_H / 2)

    # Transpose to [2, H, W]
    normalized_flow = np.transpose(resized_flow, (2, 0, 1))  # [H, W, 2] → [2, H, W]

    return normalized_flow.astype(np.float32)


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

        global_files = []
        for global_type in self.global_type_list:
            global_files.append(self.global_paths[global_type][index])

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
            condition = np.load(global_file)
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
            flow = process_flow_with_scaling(flow_path)

        return dict(jpg=image, txt=anno, local_conditions=local_conditions, flow=flow, global_conditions=global_conditions)
           