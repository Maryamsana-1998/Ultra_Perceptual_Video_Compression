from models.util import create_model, load_state_dict
import torch
import numpy as np
from src.train.video_warp_dataset import *
import matplotlib.pyplot as plt
import os
from annotator.util import HWC3
from utils.share import *
import utils.config as config
import einops
import cv2
from src.train.video_warp_dataset import process_flow_with_scaling
from test_warp import process

def load_model(config_path, ckpt_path):
    model = create_model(config_path).cpu()
    ckpt = load_state_dict(ckpt_path, location='cuda')
    model.load_state_dict(ckpt, strict=False)
    return model.cuda()

local_img_paths = [
    'data/sequences/00009/0348/r1.png',
    'data/sequences/00009/0348/depth/im5_depth.png'
]
flow_path = 'data/sequences/00009/0348/Flow/im5.flo'
prompt = 'a man with hand on his head'
local_images = []
for path in local_img_paths:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    local_images.append(img)

flow = process_flow_with_scaling(flow_path)

model_warp = load_model('configs/bi_directional_warp/uni_v15.yaml', 'experiments/bi_directional_warp/uni_test.ckpt' )

pred = process(model_warp, local_images,flow,prompt)

img_rgb = cv2.cvtColor(pred[0][0], cv2.COLOR_RGB2BGR)
plt.imshow(img_rgb)
plt.axis("off")
plt.tight_layout()
plt.savefig("warp_test.png", dpi=300)