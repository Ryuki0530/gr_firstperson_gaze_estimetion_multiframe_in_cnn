import numpy as np
import os
from tqdm import tqdm

def split_dataset(input_path, train_path, val_path, val_ratio=0.2, seed=42):
    np.random.seed(seed)
    
    print(f"[INFO] Loading: {input_path}")
    data = np.load(input_path)
    images = data["images"]  # shape: (N, 2, 3, 224, 224)
    gazes = data["gaze"]     # shape: (N, 2)
    N = len(images)
    print(f"[INFO] Loading complete. Total samples: {N}")
    indices = np.arange(N)
    np.random.shuffle(indices)

    split_idx = int(N * (1 - val_ratio))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    print(f"[INFO] Splitting into train ({len(train_idx)}) and val ({len(val_idx)})")
    train_images = np.empty((len(train_idx), *images.shape[1:]), dtype=np.float32)
    train_gazes = np.empty((len(train_idx), 2), dtype=np.float32)
    print(f"[INFO] Creating train set:")
    for i, idx in enumerate(tqdm(train_idx, desc="Train set")):
        train_images[i] = images[idx]
        train_gazes[i] = gazes[idx]

    val_images = np.empty((len(val_idx), *images.shape[1:]), dtype=np.float32)
    val_gazes = np.empty((len(val_idx), 2), dtype=np.float32)
    print(f"[INFO] Creating val set:")
    for i, idx in enumerate(tqdm(val_idx, desc="Val set")):
        val_images[i] = images[idx]
        val_gazes[i] = gazes[idx]

    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(val_path), exist_ok=True)

    print(f"[INFO] Saving train set to {train_path}")
    np.savez_compressed(train_path, images=train_images, gaze=train_gazes)

    print(f"[INFO] Saving val set to {val_path}")
    np.savez_compressed(val_path, images=val_images, gaze=val_gazes)

    print(f"[DONE] Train: {len(train_images)} samples")
    print(f"[DONE] Val:   {len(val_images)} samples")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="data/processed/gtea_single_with_past.npz")
    parser.add_argument('--train_output', type=str, default="data/processed/gtea_train.npz")
    parser.add_argument('--val_output', type=str, default="data/processed/gtea_val.npz")
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    split_dataset(args.input, args.train_output, args.val_output, args.val_ratio, args.seed)
