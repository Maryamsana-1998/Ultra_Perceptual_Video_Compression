import os
import torch
import multiprocessing as mp
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# === Dataset Class ===
class VideoDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        return image, str(path)

def custom_collate(batch):
    images, paths = zip(*batch)
    return list(images), list(paths)

# === Worker Function ===
def caption_worker(gpu_id, image_paths, return_dict, batch_size=4):
    device = f"cuda:{gpu_id}"
    print('processing')
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

    dataset = VideoDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate)

    results = []
    for images, paths in dataloader:
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30)
        captions = processor.batch_decode(out, skip_special_tokens=True)
        results.extend([(p, c) for p, c in zip(paths, captions)])

    return_dict[gpu_id] = results

# === Run Multiprocessing ===
def run_multi_gpu(root_dir, gpu_ids=[0, 1,2,3,4,5,6,7], batch_size=4):
    # Collect image paths
    all_folders = list(Path(root_dir).glob("*/*"))
    image_paths = [str(f / "r1.png") for f in all_folders if (f / "r1.png").exists()]
    
    chunks = [image_paths[i::len(gpu_ids)] for i in range(len(gpu_ids))]

    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    for i, gpu_id in enumerate(gpu_ids):
        p = mp.Process(target=caption_worker, args=(gpu_id, chunks[i], return_dict, batch_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    all_results = []
    for gpu_id in gpu_ids:
        all_results.extend(return_dict[gpu_id])

    # Optionally save to file
    with open("captions_blip.txt", "w") as f:
        for path, caption in all_results:
            f.write(f"{path}: {caption}\n")

    print("✅ All captions saved.")
    return all_results

# === Entry Point ===
if __name__ == "__main__":
    run_multi_gpu("/data2/local_datasets/vimeo_sequences/", gpu_ids=[0, 1])
