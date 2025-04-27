import os
import glob
 # Assuming this provides 'process' and 'video_details'

# === Global Prompt Settings ===
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

# === Reconstruction Function ===
def get_reconstructed_image(model, prompt, r1_image, r2_image, depth_image, flow_image):
    pred = process(
        model,
        r1_image,
        r2_image,
        depth_image,
        flow_image,
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
    )
    pred_img = pred[0][0]
    return pred_img

video_details = {
    "Beauty": {
        "prompt": "A beautiful blonde girl smiling with pink lipstick with black background",
        "path": "Beauty"
    }}
# === Data Paths Setup ===
video = 'Beauty'
details = video_details[video]

original_folder = os.path.join('data/UVG', video, "1080p")
decoded_folder = os.path.join('data/UVG', video, "1080p", "decoded_q4")
prediction_folder = os.path.join('experiments/preds', video, "1080p")
optical_flow_folder = os.path.join('data/UVG', video, "1080p", "optical_flow_gop_8")

# === Load File Lists ===
image_paths = sorted(glob.glob(os.path.join(original_folder, "*.png")))[:8]
flow_paths = sorted(glob.glob(os.path.join(optical_flow_folder, "*.png")))[0:7]
depth_frames_paths = sorted(glob.glob(os.path.join('data/depth_outputs', "*.png")))
# === Prompt for this video ===
prompt = details["prompt"]

# === Debug (Optional) ===
# print(f"Loaded {len(image_paths)} images")
# print(f"Loaded {len(previous_frames_paths)} previous frames")
# print(f"Loaded {len(flow_paths)} optical flows")
import cv2
from models.util import create_model, load_state_dict

model = create_model('configs/uni_v15.yaml').cpu()
ckpt = load_state_dict('experiments/vimeo_temp/uni.ckpt', location="cuda")
# Filter out extra keys
model_keys = set(model.state_dict().keys())
filtered_ckpt = {k: v for k, v in ckpt.items() if k in model_keys}

# Load the filtered state dict
model.load_state_dict(filtered_ckpt)
model = model.cuda()

def process(
    model,
    r1_image,
    r2_image,
    depth_image,
    flow_image,
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

        local_conditions=[]
        local_images = [r1_image,r2_image,depth_image, flow_image]

        for image in local_images:
            image = cv2.resize(image, (W, H))
            detected_map = HWC3(image)
            local_conditions.append(detected_map)
            
        content_emb = np.zeros((768))

        detected_maps = np.concatenate([condition for condition in local_conditions], axis=2)

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

    return (results, detected_maps)


r1_frame_path = 'data/UVG/Beauty/1080p/decoded_q4/decoded_frame_0000.png'
r2_frame_path = 'data/UVG/Beauty/1080p/decoded_q4/decoded_frame_0008.png'

# === Load r1 and r2 frames ===
r1_image = cv2.imread(r1_frame_path)
r2_image = cv2.imread(r2_frame_path)

r1_image = cv2.cvtColor(r1_image, cv2.COLOR_BGR2RGB)
r2_image = cv2.cvtColor(r2_image, cv2.COLOR_BGR2RGB)

print("✅ Loaded r1 and r2 frames.")

pred_images = []
for depth_path, flow_path in zip(depth_frames_paths, flow_paths):
    # Load depth image
    depth_img = cv2.imread(depth_path)
    depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2RGB)

    # Load flow image
    flow_img = cv2.imread(flow_path)
    flow_img = cv2.cvtColor(flow_img, cv2.COLOR_BGR2RGB)

    pred = get_reconstructed_image(model, prompt, r1_image, r2_image, depth_img, flow_img)
    pred_images.append(pred)

