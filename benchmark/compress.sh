#!/bin/bash

# Directory containing the flow images
FLOW_DIR="data/downsample/flows"
# Output directory for encoded/decoded files
OUTPUT_DIR="data/downsample/compressed_flows"
# Path to CompressAI codec.py
CODEC_SCRIPT="../CompressAI/examples/codec.py"
# Model to use
MODEL="mbt2018-mean"
# Quality level
QUALITY=4
# Use CUDA if available
CUDA_FLAG="--cuda"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Process each grid size
for grid in 3 5 9 12 16; do
    echo "Processing grid size $grid"
    
    # Input and output filenames
    INPUT_PNG="$FLOW_DIR/flow_grid${grid}.png"
    ENCODED_BIN="$OUTPUT_DIR/grid_${grid}.bin"
    DECODED_PNG="$OUTPUT_DIR/grid_${grid}_decoded.png"
    
    # Encode the flow image
    echo "Encoding $INPUT_PNG to $ENCODED_BIN"
    python "$CODEC_SCRIPT" encode "$INPUT_PNG" \
        --model "$MODEL" \
        -q "$QUALITY" \
        -o "$ENCODED_BIN" \
        $CUDA_FLAG
    
    # Decode the encoded file
    echo "Decoding $ENCODED_BIN to $DECODED_PNG"
    python "$CODEC_SCRIPT" decode "$ENCODED_BIN" \
        -o "$DECODED_PNG" \
        $CUDA_FLAG
    
    echo "Finished processing grid size $grid"
    echo "----------------------------------"
done

echo "All grid sizes processed!"
echo "Original files: $FLOW_DIR/flow_grid*.png"
echo "Encoded files: $OUTPUT_DIR/grid_*.bin"
echo "Decoded files: $OUTPUT_DIR/grid_*_decoded.png"