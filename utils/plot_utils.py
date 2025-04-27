import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import re
import os
import cv2

def natural_sort_key(filename):
    """Extracts numbers from filenames for proper sorting."""
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', filename)]

def load_images_from_directory(directory, num_images=10, sort=False):
    """Load image file paths from a directory, with optional sorting."""
    image_files = sorted([f for f in os.listdir(directory) if f.endswith(('.png', '.jpg'))])
    
    if sort:
        image_files = sorted(image_files, key=natural_sort_key)  # Apply sorting only if needed

    return [os.path.join(directory, f) for f in image_files[6:num_images+6]]

def read_and_resize_images(image_paths, target_size=(1920, 1080)):
    """Read images and resize to 1080p resolution."""
    images = [cv2.resize(cv2.imread(img_path), target_size) for img_path in image_paths]
    return images

def extract_vertical_patch(image, patch_width=300):
    """Extract a central vertical patch from an image."""
    h, w, _ = image.shape
    center_x = w // 2  # Find the center width
    left_x = max(0, center_x - patch_width // 2)
    right_x = min(w, center_x + patch_width // 2)
    return image[:, left_x:right_x]  # Extract vertical patch

def load_and_label_csvs(csv_paths: dict):
    dfs = []
    for label, path in csv_paths.items():
        df = pd.read_csv(path)
        df["grid_label"] = label
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def average_per_gop_grid(df):
    """Normalize GOP values and return average metrics grouped by grid + gop."""

    # Convert 'gop4' → 4 only if it's a string
    df["gop"] = df["gop"].apply(lambda x: int(str(x).replace("gop", "")) if isinstance(x, str) else int(x))

    return df.groupby(["grid_label", "gop"], as_index=False).mean(numeric_only=True)



def plot_all_metrics_line(df_avg, metrics, output_path="benchmark/grid_comparison_plots/line_all_metrics.png"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    grid_labels = df_avg["grid_label"].unique()
    gop_markers = {4: 'o', 8: 's', 16: 'D'}
    colors = {"grid_3": "red", "grid_9": "blue", "grid_15": "green", "no_grid": "black"}

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for label in grid_labels:
            df_sub = df_avg[df_avg["grid_label"] == label]
            for _, row in df_sub.iterrows():
                gop = int(row["gop"])
                ax.scatter(row["bpp"], row[metric], label=f"{label} - GOP{gop}",
                           marker=gop_markers.get(gop, 'x'),
                           color=colors.get(label, 'gray'))

            # Optionally connect points per grid
            ax.plot(df_sub["bpp"], df_sub[metric], color=colors.get(label, 'gray'), alpha=0.6)

        ax.set_title(f"{metric} vs BPP")
        ax.set_xlabel("BPP")
        ax.set_ylabel(metric)
        ax.grid(True)
        ax.legend(fontsize="small")

    # Remove empty subplot if metrics < 6
    if len(metrics) < 6:
        for j in range(len(metrics), 6):
            fig.delaxes(axes[j])

    fig.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Saved multi-metric plot: {output_path}")
    plt.close()

def plot_gop8_bars(df_avg, metrics, output_path="benchmark/grid_comparison_plots/bar_gop8_comparison.png"):
    df_gop8 = df_avg[df_avg["gop"] == 8]
    grid_labels = df_gop8["grid_label"]
    x = range(len(grid_labels))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = df_gop8[metric]
        ax.bar(x, values, tick_label=grid_labels, color="skyblue")
        ax.set_title(f"GOP 8 - {metric}")
        ax.set_ylabel(metric)
        ax.grid(axis='y')

    # Remove empty subplot if metrics < 6
    if len(metrics) < 6:
        for j in range(len(metrics), 6):
            fig.delaxes(axes[j])

    fig.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Saved GOP 8 bar plot: {output_path}")
    plt.close()


def plot_all_metrics_vs_bpp(h264_df, h265_df, unicontrol_df, metrics, save_path="benchmark/plots/all_metrics_vs_bpp.png"):
    codec_colors = {
        "h264": "blue",
        "h265": "green",
        "unicontrol": "red"
    }
    codec_labels = {
        "h264": "H.264",
        "h265": "H.265",
        "unicontrol": "UniControl"
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.plot(h264_df["BPP"], h264_df[metric], marker='o', color=codec_colors["h264"], label=codec_labels["h264"])
        ax.plot(h265_df["BPP"], h265_df[metric], marker='s', color=codec_colors["h265"], label=codec_labels["h265"])
        ax.plot(unicontrol_df["bpp"], unicontrol_df[metric], marker='D', color=codec_colors["unicontrol"], label=codec_labels["unicontrol"])

        ax.set_title(f"{metric} vs BPP")
        ax.set_xlabel("Bits Per Pixel (BPP)")
        ax.set_ylabel(metric)
        ax.grid(True)
        ax.legend()

    # If less than 4 plots, hide remaining axes
    for j in range(len(metrics), 4):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Saved: {save_path}")
    plt.close()