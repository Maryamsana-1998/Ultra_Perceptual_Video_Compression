import math
from pytorch_msssim import ms_ssim
from torchvision import transforms
import torch
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
import lpips
import matplotlib.pyplot as plt
from fvd_utils.my_utils import calculate_fvd


# Initialize LPIPS and FID models
lpips_model = lpips.LPIPS(net='alex').to('cuda' if torch.cuda.is_available() else 'cpu')
fid_model = FrechetInceptionDistance(feature=64).to('cuda' if torch.cuda.is_available() else 'cpu')

# Preprocessing transform
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize to the input size expected by the model
    transforms.ToTensor(),
    lambda x: x * 255  # Scale to 0-255
])

def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255) -> float:
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())


def calculate_metrics_batch(original_images, pred_images):
    """
    Efficiently calculates PSNR, MS-SSIM, LPIPS, FID and FVD for a batch.
    Args:
        original_images (list[PIL.Image]): originals
        pred_images     (list[PIL.Image]): predictions
    Returns:
        dict: {PSNR, MS-SSIM, LPIPS, FID, FVD}
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1) Transform all images and stack into batched tensors once
    orig_tensors = torch.stack([transform(img) for img in original_images], dim=0).to(device)
    pred_tensors = torch.stack([transform(img) for img in pred_images],  dim=0).to(device)

    # 2) Compute PSNR per frame
    psnr_vals = psnr(orig_tensors, pred_tensors)              # shape [N]
    # filter out absurd values
    valid_mask = psnr_vals <= 1000
    psnr_vals = psnr_vals[valid_mask]

    # 3) Compute MS-SSIM per frame
    ms_ssim_vals = ms_ssim(orig_tensors, pred_tensors,
                           data_range=255, size_average=False)  # shape [N]
    ms_ssim_vals = ms_ssim_vals[valid_mask]

    # 4) Compute LPIPS per frame (expects inputs in [0,1])
    lpips_vals = lpips_model(orig_tensors/255.0, pred_tensors/255.0)
    lpips_vals = lpips_vals.view(-1)[valid_mask]

    # 5) Compute FID in one go
    fid_model.reset()
    fid_model.update(orig_tensors.to(torch.uint8), real=True)
    fid_model.update(pred_tensors.to(torch.uint8), real=False)
    fid_value = fid_model.compute().item()

    # 6) Prepare for FVD: (B, T, C, H, W).  
    #    Here we assume B=1, T=N, so we reshape accordingly.
    #    If your FVD expects different layout, adjust permute.
    org_video  = orig_tensors.unsqueeze(0)  # [1, N, C, H, W]
    pred_video = pred_tensors.unsqueeze(0)
    # some FVD implementations want longer sequences; you can repeat if needed
    fvd_value = calculate_fvd(org_video, pred_video)

    # 7) Aggregate
    return {
        "PSNR":    psnr_vals.mean().item(),
        "MS-SSIM": ms_ssim_vals.mean().item(),
        "LPIPS":   lpips_vals.mean().item(),
        "FID":     fid_value,
        "FVD":     fvd_value
    }
