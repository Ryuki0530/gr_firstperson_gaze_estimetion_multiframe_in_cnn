import os
import cv2
import numpy as np
import pandas as pd
import random
from glob import glob
from tqdm import tqdm

# === 設定 ===
CUR_SIZE = (224, 224)
PAST_SIZE = (224, 224)  # 必要ならあとで112×112に縮小
FRAME_DIFF = 90
MAX_SAMPLES_PER_VIDEO = 2000

def read_gaze_labels(txt_path):
    try:
        df = pd.read_csv(txt_path, sep='\t', engine='python')
        required_cols = ['Frame', 'L POR X [px]', 'L POR Y [px]']
        if not all(col in df.columns for col in required_cols):
            print(f"[WARN] Missing required columns in {txt_path}: {df.columns.tolist()}")
            return None

        df = df[required_cols].dropna()
        df_grouped = df.groupby('Frame').mean()
        gaze_map = {
            int(frame): (float(row['L POR X [px]']), float(row['L POR Y [px]']))
            for frame, row in df_grouped.iterrows()
        }
        return gaze_map
    except Exception as e:
        print(f"[ERROR] Failed to read {txt_path}: {e}")
        return None

def process_video(video_path, label_path):
    cap = cv2.VideoCapture(video_path)
    gaze_map = read_gaze_labels(label_path)
    if gaze_map is None:
        return np.array([]), np.array([])

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    valid_indices = [i for i in gaze_map if i >= FRAME_DIFF and i < total_frames]
    if len(valid_indices) == 0:
        return np.array([]), np.array([])

    random.shuffle(valid_indices)
    selected_indices = valid_indices[:MAX_SAMPLES_PER_VIDEO]

    images = []
    gazes = []

    video_name = os.path.basename(video_path).replace(".avi", "")
    progress = tqdm(total=len(selected_indices), desc=f"Processing {video_name}")

    for i in selected_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i - FRAME_DIFF)
        ret1, past = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret2, current = cap.read()
        if not (ret1 and ret2):
            continue

        if i not in gaze_map:
            continue

        gx, gy = gaze_map[i]
        gaze = (gx / W, gy / H)

        # OpenCV: BGR → RGB + CHW + 正規化
        cur_img = cv2.resize(current, CUR_SIZE)[..., ::-1].transpose(2, 0, 1) / 255.0
        past_img = cv2.resize(past, PAST_SIZE)[..., ::-1].transpose(2, 0, 1) / 255.0

        images.append([cur_img, past_img])
        gazes.append(gaze)
        progress.update(1)

    cap.release()
    progress.close()
    return np.array(images, dtype=np.float32), np.array(gazes, dtype=np.float32)

def main(input_dir, output_path, seed=42):
    random.seed(seed)
    all_images = []
    all_gazes = []

    video_paths = glob(os.path.join(input_dir, "*.avi"))
    for video_path in tqdm(video_paths, desc="All videos"):
        base = os.path.splitext(os.path.basename(video_path))[0]
        label_path = os.path.join(input_dir, base + ".txt")
        if not os.path.exists(label_path):
            print(f"[WARN] Missing label for {video_path}")
            continue

        imgs, gazes = process_video(video_path, label_path)
        if len(imgs) == 0:
            print(f"[INFO] No valid samples from {base}")
            continue

        all_images.append(imgs)
        all_gazes.append(gazes)

    if len(all_images) == 0:
        print("[ERROR] No data processed from any video.")
        return

    all_images = np.concatenate(all_images, axis=0)
    all_gazes = np.concatenate(all_gazes, axis=0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, images=all_images, gaze=all_gazes)
    print(f"[DONE] Saved {len(all_images)} samples to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="data/raw/")
    parser.add_argument('--output', type=str, default="data/processed/gtea_single_with_past.npz")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args.input_dir, args.output, args.seed)
