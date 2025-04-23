import os
import subprocess
from pathlib import Path

# Configuration
VIDEOS = [
    "Beauty",
    "Jockey",
    "Bosphorus"
]
COMPRESSAI_PATH = Path("/data/maryam.sana/CompressAI/examples/codec.py") 

FRAMES = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96]
MODEL = "mbt2018-mean"
QUALITY = 4
GOP = 4  # Matches frame selection pattern
CUDA_FLAG="--cuda"

def process_video(video_name):
    input_dir = Path(f"UVG/{video_name}/1080p")
    output_dir = Path(f"UVG/{video_name}/1080p/decoded_q4/")
    output_dir.mkdir(exist_ok=True)
    
    report_path = Path(f"UVG/{video_name}/1080p/decoded_q4/compression_report.txt")
    with open(report_path, "a") as report:
        report.write(f"# Compression Report for model={MODEL}, quality={QUALITY}, GOP={GOP}\n")
        report.write(f"## Video: {video_name} | Resolution: 1080p\n\n")
        
        for frame_num in FRAMES:
            frame_file = f"frame_{frame_num:04d}.png"
            input_path = input_dir / frame_file
            bin_file = output_dir / f"{frame_file.split('.')[0]}.bin"
            if not input_path.exists() and not bin_file.exists():
                continue
                
            encode_cmd = [
                "python", str(COMPRESSAI_PATH), "encode",
                 str(input_path),
                "-o", str(bin_file),
                "--model", MODEL,
                "-q", str(QUALITY),
                CUDA_FLAG
            ]
            subprocess.run(encode_cmd, check=True)
            
            # Decode
            decoded_file = output_dir / f"decoded_{frame_file}"
            decode_cmd = [
                "python", str(COMPRESSAI_PATH), "decode",
                 str(bin_file),
                "-o", str(decoded_file), CUDA_FLAG
            ]
            subprocess.run(decode_cmd, check=True)
            
            # Get file size
            size_kb = os.path.getsize(bin_file) / 1024
            
            # Write to report
            report.write(f"- Frame: {frame_file} → {size_kb:.2f} KB\n")
            
            print(f"Processed {video_name}/{frame_file} ({size_kb:.2f} KB)")

if __name__ == "__main__":
    for video in VIDEOS:
        print(f"\nProcessing {video}...")
        process_video(video)
    print("\nAll videos processed. Reports saved in each video's directory.")