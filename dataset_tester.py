import numpy as np
import matplotlib.pyplot as plt

# 読み込むファイルのパス
npz_path = "data/processed/gtea_train.npz"

# npzファイルを読み込み
data = np.load(npz_path)

# どのキーがあるか確認（デバッグ用）
print("Available keys:", data.files)

# "gaze" や "labels" などのキーで視線座標を取り出す
# 例：data["labels"] = [ [x1, y1], [x2, y2], ... ]
if "gaze" in data:
    gaze = data["gaze"]
elif "labels" in data:
    gaze = data["labels"]
else:
    raise KeyError("gaze or labels キーが npz に見つかりません")

# gaze の x, y を取り出す
x_coords = gaze[:, 0]
y_coords = gaze[:, 1]

# プロット
plt.figure(figsize=(6, 6))
plt.hist2d(x_coords, y_coords, bins=50, range=[[0, 1], [0, 1]], cmap="jet")
plt.xlabel("Gaze X")
plt.ylabel("Gaze Y")
plt.title("Gaze Point Distribution")
plt.colorbar(label="Count")
plt.tight_layout()
plt.show()
