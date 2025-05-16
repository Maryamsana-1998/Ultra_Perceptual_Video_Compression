import os
import torch
import multiprocessing as mp
from PIL import Image
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
import glob


# ==== Configuration ====
VIMEO_ROOT = "data/sequences"
BATCH_SIZE = 1
NUM_WORKERS = 8  # CPU threads per GPU process
GPU_IDS = [0]  # List of GPU IDs to use
SAVE_PATH = "data/caption_test.txt"
# ========================

def custom_collate(batch):
    seqs, images = zip(*batch)  # separate tuples
    return list(seqs), list(images)


# ==== Dataset ====

class Im1Dataset(Dataset):
    def __init__(self, folders):
        self.image_paths = []
        for seq in folders:
            print(seq)
            seq_path = seq
            img_path = os.path.join(seq_path,"r1.png")
            if os.path.exists(img_path):
                self.image_paths.append((seq, img_path))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        seq, path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        return seq, image

# ==== Worker function ====
def caption_worker(gpu_id, image_list, return_dict):
    device = torch.device(f"cuda:{gpu_id}")

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16
    ).to(device)

    dataset = Im1Dataset(image_list)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=custom_collate)

    results = []
    for batch in tqdm(dataloader, desc=f"GPU {gpu_id}", position=gpu_id):
        seqs, images = batch
        inputs = processor(images=list(images), return_tensors="pt", padding=True).to(device, torch.float16)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=20)
        captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for seq, cap in zip(seqs, captions):
            results.append(f"{seq}: {cap.strip()}")

    return_dict[gpu_id] = results

# ==== Main process ====
if __name__ == "__main__":
    all_sequences = sorted(glob.glob(VIMEO_ROOT+'*/*/*'))
    results = {}
    caption_worker(0, all_sequences,results )
    print(results)
    # print(all_sequences)
    # chunk_size = len(all_sequences) // len(GPU_IDS)
    # chunks = [all_sequences[i:i + chunk_size] for i in range(0, len(all_sequences), chunk_size)]

    # manager = mp.Manager()
    # return_dict = manager.dict()
    # processes = []

    # for i, gpu_id in enumerate(GPU_IDS):
    #     chunk = chunks[i] if i < len(chunks) else []
    #     p = mp.Process(target=caption_worker, args=(gpu_id, chunk, return_dict))
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()

    # # Write all results to file
    # with open(SAVE_PATH, "w") as f:
    #     for gpu_id in return_dict.keys():
    #         for line in return_dict[gpu_id]:
    #             f.write(line + "\n")

    print(f"✅ Done! Captions saved to: {SAVE_PATH}")
