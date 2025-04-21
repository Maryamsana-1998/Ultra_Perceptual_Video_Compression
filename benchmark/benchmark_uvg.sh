#!/bin/bash

VIDEOS=('Bosphorus_1920x1080_120fps_420_8bit_YUV.yuv'
 'HoneyBee_1920x1080_120fps_420_8bit_YUV.yuv'
 'ShakeNDry_1920x1080_120fps_420_8bit_YUV.yuv'
 'CityAlley_1920x1080_50fps_420_10bit_YUV.yuv'
 'Jockey_1920x1080_120fps_420_8bit_YUV.yuv'
 'ReadySteadyGo_1920x1080_120fps_420_8bit_YUV.yuv'
 'YachtRide_1920x1080_120fps_420_8bit_YUV.yuv')

WIDTH=1920
HEIGHT=1080
FRAMES=97
GOP=8

# CRF values
CRF_H264=(18 23 28)
CRF_H265=(25 28 30)

for VIDEO in "${VIDEOS[@]}"; do
    echo "Processing $VIDEO"

    for i in {0..2}; do
        CRF_X264=${CRF_H264[$i]}
        CRF_X265=${CRF_H265[$i]}

        echo " - H264 | CRF: $CRF_X264"
        ffmpeg -s ${WIDTH}x${HEIGHT} -pix_fmt yuv420p -f rawvideo -i "$VIDEO" -frames:v $FRAMES \
            -c:v libx264 -crf $CRF_X264 -preset medium -g $GOP "${VIDEO}_h264_crf${CRF_X264}_gop${GOP}.mp4"

        echo " - H265 | CRF: $CRF_X265"
        ffmpeg -s ${WIDTH}x${HEIGHT} -pix_fmt yuv420p -f rawvideo -i "$VIDEO" -frames:v $FRAMES \
            -c:v libx265 -crf $CRF_X265 -preset medium -g $GOP "${VIDEO}_h265_crf${CRF_X265}_gop${GOP}.mp4"
    done
done

echo "✅ All videos processed!"
