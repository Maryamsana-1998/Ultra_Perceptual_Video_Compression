import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import re
import os
import cv2

# === Plot 1: Metrics vs Grid Size ===
def plot_metrics_vs_grid(df: pd.DataFrame, metrics: list, output_file: str):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=300)
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.plot(df["grid_size"], df[metric], marker='o', linestyle='-', color='blue')
        for x, y in zip(df["grid_size"], df[metric]):
            ax.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"{metric} vs Grid Size")
        ax.set_xlabel("Grid Size")
        ax.set_ylabel(metric)
        ax.grid(True)

    for j in range(len(metrics), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"✅ Saved metric plot: {output_file}")
    plt.close()


def plot_bpp_vs_grid(df: pd.DataFrame, output_file: str):
    if "bpp" not in df.columns:
        if "total_bits" in df.columns:
            width, height, frames = 1920, 1080, 97
            total_pixels = width * height * frames
            df["bpp"] = df["total_bits"] / total_pixels
        else:
            raise ValueError("BPP or total_bits column is missing.")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # Scale marker sizes based on grid_size (you can adjust the scaling factor)
    marker_sizes = df["grid_size"] * 30  # Scale factor (adjust as needed)
    
    ax.scatter(df["grid_size"], df["bpp"], s=marker_sizes, 
               color='purple', alpha=0.6, label="Average BPP")
    
    # Connect points with lines (optional)
    ax.plot(df["grid_size"], df["bpp"], linestyle='-', color='purple', alpha=0.3)
    
    for x, y in zip(df["grid_size"], df["bpp"]):
        ax.text(x, y, f"{y:.4f}", ha="center", va="bottom", fontsize=8)
    
    ax.set_title("BPP vs Grid Size (Marker Size ∝ Grid Size)")
    ax.set_xlabel("Grid Size")
    ax.set_ylabel("Average BPP")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"✅ Saved BPP plot: {output_file}")
    plt.close()

# === Load and aggregate ===
def load_and_average(csv_path: str, label: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if "gop" in df.columns:
        df["gop"] = df["gop"].astype(str).str.extract(r"(\d+)").astype(int)
        df = df[df["gop"] == 8]
    df["grid_label"] = label
    return df.groupby("grid_label", as_index=False).mean(numeric_only=True)



def plot_metrics_comparison(avg_h264: pd.DataFrame, avg_h265: pd.DataFrame, avg_df: pd.DataFrame,
                            metrics_to_plot, color_map, label_map,
                            output_path="benchmark/plots/uvg_metrics_vs_bpp_gop8.png",
                            scatter_points=None):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]

        # Plot each method
        ax.plot(avg_h264["bpp"], avg_h264[metric], marker='o', linestyle='-', color=color_map["h264"], label=label_map["h264"])
        ax.plot(avg_h265["bpp"], avg_h265[metric], marker='s', linestyle='--', color=color_map["h265"], label=label_map["h265"])
        ax.plot(avg_df["bpp"], avg_df[metric], marker='D', linestyle='-.', color=color_map["unicontrol"], label=label_map["unicontrol"])

        # Optional: annotate UniControl points
        for x, y in zip(avg_df["bpp"], avg_df[metric]):
            ax.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=8)

        # Add scatter point if provided
        if scatter_points and metric in scatter_points:
            sp_x, sp_y = scatter_points[metric]
            ax.scatter(sp_x, sp_y, color="black", marker="*", s=100, label="Key Point")
            ax.annotate(scatter_points['name'], (sp_x, sp_y), textcoords="offset points", xytext=(5, 5), ha='left', fontsize=9)

        ax.set_xlabel("BPP")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs BPP")
        ax.grid(True)
        ax.legend(fontsize="small")

    # Remove extra subplots
    if len(metrics_to_plot) < len(axes):
        for j in range(len(metrics_to_plot), len(axes)):
            fig.delaxes(axes[j])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✅ Saved plot to {output_path}")
    plt.show()

# === Helper functions

def load_and_prepare_codec_df(csv_path: str,quality_order: list) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Standardize columns (remove leading spaces)
    df.columns = df.columns.str.strip()

    # If 'BPP' exists, rename to 'bpp'
    if 'BPP' in df.columns:
        df.rename(columns={'BPP': 'bpp'}, inplace=True)
    elif 'bpp' not in df.columns:
        raise ValueError(f"No BPP column found in {csv_path}")

    # Group by CRF and average
    df_avg = df.groupby('CRF', as_index=False).mean(numeric_only=True)

    # Sort by quality order
    df_avg["CRF"] = pd.Categorical(df_avg["CRF"], categories=quality_order, ordered=True)
    df_avg = df_avg.sort_values("CRF").sort_values("bpp")

    return df_avg



def plot_codec_comparison(models, metrics, metric_labels, ylabels, colors, output_path):
    """Creates a 1-row bar chart comparison for multiple metrics."""
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(4 * num_metrics, 5), dpi=300)
    fig.suptitle("Comparison of Our Results with Perco", fontsize=16, fontweight='bold')

    for i, (label, values) in enumerate(metrics.items()):
        create_comparison_bar(
            ax=axes[i],
            models=models,
            values=values,
            title=f"{label} Comparison",
            ylabel=ylabels[i],
            colors=colors
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"✅ Saved plot to {output_path}")
    plt.show()



def create_comparison_bar(ax, models, values, title, ylabel, colors):
    """Plots a single comparison bar chart on a given axis."""
    x = np.arange(len(models))
    bar_width = 0.3

    ax.bar(x, values, color=colors, width=bar_width)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

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