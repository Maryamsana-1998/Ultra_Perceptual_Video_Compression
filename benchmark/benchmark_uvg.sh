#!/bin/bash

# === Settings ===
VIDEOS=(
  'Beauty_1920x1080_120fps_420_8bit_YUV.yuv'
  'Bosphorus_1920x1080_120fps_420_8bit_YUV.yuv'
  'HoneyBee_1920x1080_120fps_420_8bit_YUV.yuv'
  'ShakeNDry_1920x1080_120fps_420_8bit_YUV.yuv'
  'Jockey_1920x1080_120fps_420_8bit_YUV.yuv'
  'ReadySteadyGo_1920x1080_120fps_420_8bit_YUV.yuv'
  'YachtRide_1920x1080_120fps_420_8bit_YUV.yuv'
)

WIDTH=1920
HEIGHT=1080
FRAMES=97
GOP=8

FPS=120   # Adjust if you want automatic detection per video name
PIX_FMT="yuv420p"  # Default; special case for 10-bit content below

# === Encoding Bitrates Based on BPP Targets ===
BPP_TARGETS=(0.01 0.05 0.10)

# Calculate Bitrate for each BPP
calc_bitrate() {
  local bpp=$1
  echo $(( $(echo "$bpp * $WIDTH * $HEIGHT * $FPS" | bc | cut -d'.' -f1) ))
}

# === Processing Loop ===
for video in "${VIDEOS[@]}"; do
  echo "🔵 Processing $video ..."
  FPS=120
  PIX_FMT="yuv420p"

  INPUT_FMT="-f rawvideo -pix_fmt $PIX_FMT -s ${WIDTH}x${HEIGHT} -r $FPS"
  INPUT_PATH="../../UVG/$video"  # Adjust this if your YUVs are elsewhere

  BASENAME=$(basename "$video" .yuv)

  for BPP in "${BPP_TARGETS[@]}"; do
    BITRATE=$(calc_bitrate $BPP)  # bitrate in bps
    BITRATE_K=$((BITRATE / 1000)) # bitrate in kbps for ffmpeg

    echo "  ➡️ BPP=$BPP, Target Bitrate=${BITRATE_K}k"

    # Encode with H264
    ffmpeg $INPUT_FMT -i "$INPUT_PATH" \
      -frames:v $FRAMES \
      -c:v libx264 -b:v ${BITRATE_K}k -g $GOP -keyint_min $GOP \
      -preset slow \
      "./outputs/${BASENAME}_bpp${BPP}_h264.mp4"

    # Encode with H265
    ffmpeg $INPUT_FMT -i "$INPUT_PATH" \
      -frames:v $FRAMES \
      -c:v libx265 -b:v ${BITRATE_K}k -g $GOP -keyint_min $GOP \
      -preset slow \
      "./outputs/${BASENAME}_bpp${BPP}_h265.mp4"

  done

  echo "✅ Finished $video"
done

echo "🎉 All videos processed!"
