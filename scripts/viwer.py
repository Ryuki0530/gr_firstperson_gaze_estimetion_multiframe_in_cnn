# tools/view_npz_gtea_single.py

import numpy as np
import matplotlib.pyplot as plt

# .npzファイルのパス
NPZ_PATH = "data/processed/gtea_train.npz"

# データ読み込み
data = np.load(NPZ_PATH)
images = data["images"]  # shape: (N, 2, 3, 224, 224)
gazes = data["gaze"]     # shape: (N, 2)

print(f"Total samples: {len(images)}")
print(f"Image shape: {images.shape}")
print(f"Gaze shape: {gazes.shape}")

def view_sample(idx):
    """1サンプル（現在+過去フレーム）の視覚確認"""
    if idx >= len(images):
        print(f"Index {idx} is out of range.")
        return

    current = images[idx, 0].transpose(1, 2, 0)[..., ::-1]  # (224, 224, 3) BGR→RGB
    past    = images[idx, 1].transpose(1, 2, 0)[..., ::-1]

    gaze = gazes[idx]
    gx = int(gaze[0] * 224)
    gy = int(gaze[1] * 224)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(current)
    axs[0].set_title(f"Current Frame\nGaze: ({gaze[0]:.3f}, {gaze[1]:.3f})", color='red')
    axs[0].scatter([gx], [gy], color='red', s=100, marker='x')
    axs[0].axis('off')

    axs[1].imshow(past)
    axs[1].set_title("Past Frame (≈3s ago)")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

def interactive_viewer():
    """ユーザー入力で任意サンプルを確認"""
    while True:
        try:
            user_input = input(f"\nEnter sample index (0-{len(images)-1}) or 'q' to quit: ")
            if user_input.lower() == 'q':
                break

            idx = int(user_input)
            view_sample(idx)
        except ValueError:
            print("Please enter a valid number or 'q'")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

# 初期表示
print("\nShowing first sample...")
view_sample(0)

print("\n" + "="*50)
print("Interactive viewer mode")
interactive_viewer()
