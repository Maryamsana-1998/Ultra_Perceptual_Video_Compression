import os
import glob
import argparse
import json
import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from pytorch_lightning import seed_everything
from src.test.test_codec import process_images
from test_utils import calculate_metrics_batch

video_details = {
    # "Beauty": {
    #     "prompt": "A beautiful blonde girl smiling with pink lipstick with black background",
    #     "path": "Beauty"
    # },
    # "Jockey": {
    #     "prompt": "A man riding a brown horse, galloping through a green race track. The man is wearing a yellow and red shirt and also a yellow hat",
    #     "path": "Jockey"
    # },
    "Bosphorus": {
        "prompt": "A man and a woman sitting together on a boat sailing in water. They are both wearing ties. There is also a red flag at end of boat",
        "path": "Bosphorus"
    },
    "ShakeNDry": {
        "prompt": "A German shepherd shakes off water in the middle of a forest trail",
        "path": "ShakeNDry"
    },
    "YachtRide": {
        "prompt": "A sleek black-and-wood luxury yacht cruises through calm blue waters, carrying three passengers — two in conversation at the front and one steering under a shaded canopy.",
        "path": "YachtRide"
    },
    "HoneyBee": {
        "prompt": "Honeybees hover among blooming purple flowers",
        "path": "HoneyBee"
    },
    "ReadySteadyGo": {
        "prompt": "The moment of launch at Türkiye Jokey Kulübü, as jockeys and their horses surge out of the starting gates on a lush green turf",
        "path": "ReadySteadyGo"
    }
}


def plot_images(original_images, predictions, save_location, start_index=4, end_index=9, dpi=300):
    fig, axes = plt.subplots(2, end_index - start_index, figsize=(30, 10))
    for i in range(start_index, end_index):
        axes[0, i - start_index].imshow(original_images[i])
        axes[0, i - start_index].set_title(f"Original {i + 1}")
        axes[0, i - start_index].axis('off')
        axes[1, i - start_index].imshow(predictions[i])
        axes[1, i - start_index].set_title(f"Prediction {i + 1}")
        axes[1, i - start_index].axis('off')
    plt.tight_layout()
    plt.savefig(save_location, dpi=dpi, bbox_inches='tight')
    print(f"Plot saved at {save_location}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Video Metrics and Generate Predictions")
    parser.add_argument("--original_root", type=str, required=True)
    parser.add_argument("--pred_root", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    # parser.add_argument("--videos", nargs='+', default=["Beauty", "Jockey", "Bosphorus"],
    #                     help="Select one or more videos: Beauty, Jockey, Bosphorus")
    parser.add_argument("--gop", type=int, default=8)
    parser.add_argument("--intra_quality", type=int, default=4, choices=[1, 2, 4])
    parser.add_argument("--resolution", type=str, default="1080p", choices=["512p", "1080p"])

    args = parser.parse_args()

    seed_everything(42)
    all_metrics = {}

    for video in video_details.keys():
        details = video_details[video]
        print(f"\nProcessing video: {video}")

        original_folder = os.path.join(args.original_root, video, f"{args.resolution}")
        optical_flow_folder = os.path.join(args.original_root, video, f"{args.resolution}", f"optical_flow_gop_{args.gop}")
        previous_frame_folder = os.path.join(args.original_root, video, f"{args.resolution}", f"decoded_q{args.intra_quality}")
        pred_folder = os.path.join(args.pred_root, video, args.resolution)

        image_paths = sorted(glob.glob(os.path.join(original_folder, "*.png")))
        flow_paths = sorted(glob.glob(os.path.join(optical_flow_folder, "*.png")))
        previous_frames_paths = sorted(glob.glob(os.path.join(previous_frame_folder, "*.png")))[::args.gop // 4]

        prompt = details["prompt"]

        os.makedirs(pred_folder, exist_ok=True)

        original_images, predictions = process_images(
            config_path=args.config,
            ckpt_path=args.ckpt,
            image_paths=image_paths,
            canny_paths=flow_paths,
            prompt=prompt,
            previous_frames_paths=previous_frames_paths,
            pred_folder=pred_folder,
            gop=args.gop
        )

        plot_save_location = os.path.join(pred_folder, "original_vs_predicted.png")
        if len(original_images) > 5 and len(predictions) > 5:
            plot_images(
                original_images=original_images,
                predictions=predictions,
                save_location=plot_save_location,
                start_index=10,
                end_index=18,
                dpi=300
            )

    #     original_eval_images = []
    #     pred_eval_images = []
    #     for i in range(2, len(image_paths)):
    #         # print(original_folder,)
    #         original_path = os.path.join(original_folder, f"frame_{i-1:04d}.png")
    #         pred_path = os.path.join(pred_folder, f"im{i:05d}_pred.png")
    #         # print(original_folder,original_path, pred_path)

    #         if os.path.exists(original_path) and os.path.exists(pred_path):
    #             original_eval_images.append(Image.open(original_path).convert("RGB"))
    #             pred_eval_images.append(Image.open(pred_path).convert("RGB"))
    #         else:
    #             print(f"Warning: Missing image for {video} frame {i}")

    #     if original_eval_images and pred_eval_images:
    #         metrics = calculate_metrics_batch(original_eval_images, pred_eval_images)
    #         all_metrics[video] = metrics
    #         print(f"Metrics for {video}:", metrics)
    #     else:
    #         print(f"No images found or incomplete data for video {video}. Skipping metrics.")

    # print("\nFinal Metrics Summary for All Videos:")
    # for video, metrics in all_metrics.items():
    #     print(f"{video} Metrics: {metrics}")

    # metrics_json_path = os.path.join(args.pred_root, f"all_videos_metrics_{args.resolution}_q{args.intra_quality}.json")
    # with open(metrics_json_path, "w") as f:
    #     json.dump(all_metrics, f, indent=4)

    # print(f"\nAll metrics saved to {metrics_json_path}")

if __name__ == "__main__":
    main()
