import os
from pathlib import Path
import numpy as np
from collections import Counter
import struct
import flowiz as fz
import cv2

# --- Provided helper functions ---
def load_flo_file(file_path):
    with open(file_path, 'rb') as f:
        magic = f.read(4)
        if magic != b'PIEH':
            raise Exception('Invalid .flo file')

        width = struct.unpack('i', f.read(4))[0]
        height = struct.unpack('i', f.read(4))[0]

        data = np.fromfile(f, np.float32, count=2 * width * height)
        flow = np.resize(data, (height, width, 2))
        return flow

def compute_block_vector(block, method='mean'):
    flat = block.reshape(-1, 2)
    if method == 'mean':
        return flat.mean(axis=0)
    elif method == 'median':
        return np.median(flat, axis=0)
    elif method == 'mode':
        quantized = (flat * 10).astype(int)
        tupled = [tuple(v) for v in quantized]
        most_common = Counter(tupled).most_common(1)[0][0]
        return np.array(most_common) / 10.0
    elif method == 'weighted':
        mag = np.linalg.norm(flat, axis=1)
        return (flat * mag[:, None]).sum(0) / (mag.sum() + 1e-6)
    else:
        raise ValueError(f"Unknown method: {method}")

def reconstruct_flow(flow, grid_size, method):
    H, W, _ = flow.shape
    h_blocks = H // grid_size
    w_blocks = W // grid_size
    out = np.zeros((h_blocks, w_blocks, 2), dtype=np.float32)
    
    for i in range(h_blocks):
        for j in range(w_blocks):
            block = flow[i*grid_size:(i+1)*grid_size, j*grid_size:(j+1)*grid_size]
            out[i, j] = compute_block_vector(block, method)
    
    return out


def write_flo_file(flow, filename):
    assert flow.ndim == 3 and flow.shape[2] == 2
    with open(filename, 'wb') as f:
        f.write(struct.pack('f', 202021.25))
        f.write(struct.pack('i', flow.shape[1]))  # width
        f.write(struct.pack('i', flow.shape[0]))  # height
        flow.astype(np.float32).tofile(f)

# --- Main Script ---
def process_all_flows(base_dir):
    base_dir = Path(base_dir)
    grid_sizes = [3, 9, 15]

    for root, dirs, files in os.walk(base_dir):
        flo_files = [f for f in files if f.endswith('.flo')]
        if not flo_files:
            continue

        flow_dir = Path(root)
        print(f"Processing directory: {flow_dir}")

        for flo_file in flo_files:
            input_path = flow_dir / flo_file
            flow = load_flo_file(input_path)

            for grid in grid_sizes:
                output_dir = flow_dir / f"grid_{grid}"
                output_dir.mkdir(exist_ok=True)

                processed_flow = reconstruct_flow(flow, grid_size=grid, method='weighted')
                output_path = output_dir / flo_file
                write_flo_file(processed_flow, output_path)
                png_output = output_path.with_suffix(".png")
                flo_img = fz.convert_from_file(str(output_path))
                cv2.imwrite(str(png_output),flo_img[..., ::-1])

                print(f"Saved: {output_path}")


process_all_flows("data/UVG")