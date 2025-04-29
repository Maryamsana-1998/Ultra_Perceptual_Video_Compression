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
    colors = {"grid_3": "red", "grid_9": "blue", "grid_15": "green", "no_grid": "black"}

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=200)
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for label in grid_labels:
            df_sub = df_avg[df_avg["grid_label"] == label]

            # Scatter points (without labels)
            ax.scatter(df_sub["bpp"], df_sub[metric], color=colors.get(label, 'gray'))

            # Plot the connecting line (with one label only here)
            ax.plot(df_sub["bpp"], df_sub[metric], label=label, color=colors.get(label, 'gray'), alpha=0.8)

        ax.set_title(f"{metric} vs BPP", fontsize=12)
        ax.set_xlabel("BPP", fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize="small", loc="best", frameon=True)

    # Remove extra empty subplot if needed
    if len(metrics) < 6:
        for j in range(len(metrics), 6):
            fig.delaxes(axes[j])

    fig.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Saved multi-metric plot: {output_path}")
    plt.close()


def plot_gop8_bars(df_avg, metrics, output_path="benchmark/grid_comparison_plots/bar_gop8_comparison.png"):
    df_gop8 = df_avg[df_avg["gop"] == 8]
    grid_labels = df_gop8["grid_label"].tolist()
    x = np.arange(len(grid_labels))

    # Custom colors
    custom_colors = {
        "grid_3": "red",
        "grid_9": "blue",
        "grid_15": "green",
        "no_grid": "gray"
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=200)  # Higher DPI
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        if i >= len(axes):
            break
        ax = axes[i]

        # Values for the metric
        values = df_gop8[metric].tolist()
        colors = [custom_colors.get(label, "black") for label in grid_labels]

        bars = ax.bar(x, values, color=colors, width=0.5)  # Wider spacing (smaller width)

        # Bar labels (legend entries)
        for idx, bar in enumerate(bars):
            bar.set_label(grid_labels[idx])

        ax.set_title(f"GOP 8 - {metric}", fontsize=12)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(grid_labels, fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Add legend once per subplot
        ax.legend(fontsize=8, loc='best')

    # Remove empty plots if fewer than 6 metrics
    if len(metrics) < 6:
        for j in range(len(metrics), 6):
            fig.delaxes(axes[j])

    fig.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Saved enhanced GOP 8 bar plot: {output_path}")
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