import os
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count

# Define constants
CODEC_SCRIPT = "../../CompressAI/examples/codec.py"
MODEL = "mbt2018-mean"
QUALITY = 4
CUDA_FLAG = "--cuda"  # If you want to use GPUs

def process_image(image_path, grid_folder, cuda_flag):
    """Encodes and decodes the image, writes BPP, and cleans up .flo (always)."""

    encoded_bin = image_path.with_suffix(".bin")
    decoded_png = image_path.with_suffix(".decoded.png")
    bpp_report_path = grid_folder / "bpp_report.txt"


    if image_path.suffix != ".png" or image_path.name.endswith(".decoded.png"):
        return f"❌ Skipped {image_path.name} (invalid input)"

    # Encoding
    encode_cmd = [
        "python", CODEC_SCRIPT, "encode", str(image_path),
        "--model", MODEL, "-q", str(QUALITY),
        "-o", str(encoded_bin), cuda_flag
    ]
    subprocess.run(encode_cmd, check=True)

    # Decoding
    decode_cmd = [
        "python", CODEC_SCRIPT, "decode", str(encoded_bin),
        "-o", str(decoded_png), cuda_flag
    ]
    subprocess.run(decode_cmd, check=True)

    # Report BPP
    bin_size = encoded_bin.stat().st_size
    with open(bpp_report_path, "a") as report_file:
        report_file.write(f"{image_path.name}: {bin_size} bytes\n")

    # Cleanup
    encoded_bin.unlink()
    image_path.unlink()  # remove original .flo

    return f"✅ Processed {image_path.name}, BPP size: {bin_size} bytes"


def process_grid(grid_folder):
    """Process only raw .png files (not .decoded.png) in a grid folder."""
    # Get only base .png files (exclude .decoded.png)
    png_files = [p for p in grid_folder.glob("*.png") if not p.name.endswith(".decoded.png")]

    tasks = [(image, grid_folder, CUDA_FLAG) for image in png_files]

    with Pool(processes=min(cpu_count(), 16)) as pool:
        results = pool.starmap(process_image, tasks)

    return results

def process_video(video_folder):
    """Process all grid folders for each video."""
    # Iterate through all GOP folders (e.g., optical_gop4, optical_gop8, etc.)
    for gop_folder in video_folder.glob("optical_flow_gop_*"):
        for grid_folder in gop_folder.glob("grid_*"):
            print(f"Processing folder: {grid_folder}")
            process_grid(grid_folder)

def main():
    """Main function to process all videos."""
    # Define the main directory for all videos
    video_base_dir = Path("./UVG")

    # Process each video folder
    for video_folder in video_base_dir.glob("*"):
        if video_folder.is_dir():
            video_1080p_folder = video_folder / "1080p"
            print(f"Starting to process video: {video_1080p_folder}")
            process_video(video_1080p_folder)

if __name__ == "__main__":
    main()
