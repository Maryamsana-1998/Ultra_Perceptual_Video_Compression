#!/bin/bash

input_dir="./"
output_dir="test_codec"

mkdir -p "$output_dir"

# Iterate over all .mp4 files in input_dir
for file in "$input_dir"/*.mp4; do
    filename=$(basename "$file")
    
    # Updated regex to match filename pattern
    if [[ "$filename" =~ ^(.*)_1920x1080_.*_YUV\.yuv_(h26[45])_crf([0-9]+)_gop([0-9]+)\.mp4$ ]]; then
        video="${BASH_REMATCH[1]}"
        codec="${BASH_REMATCH[2]}"
        crf="${BASH_REMATCH[3]}"
        gop="${BASH_REMATCH[4]}"
        
        # Define structured output path
        out_path="$output_dir/$video/$codec/crf$crf"
        mkdir -p "$out_path"

        echo "🎞️ Processing: $filename → $out_path"

        # Extract frames
        ffmpeg -i "$file" -q:v 2 "$out_path/frame_%04d.png"

        # Save compression details
        cat <<EOF > "$out_path/details.txt"
Video: $video
Codec: $codec
CRF: $crf
GOP: $gop
Preset: medium
EOF
    else
        echo "❌ Skipping unrecognized filename pattern: $filename"
    fi
done

echo -e "\n✅ All videos processed and organized into $output_dir/"
