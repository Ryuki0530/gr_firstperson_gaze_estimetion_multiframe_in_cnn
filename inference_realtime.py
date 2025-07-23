import os
import cv2
import torch
import numpy as np
from collections import deque
from models.gaze_mobilenetv3 import GazeMobileNetV3

VIDEO_PATH = "data/raw/Ahmad_American.avi"
MODEL_PATH = "checkpoints/model_epoch750.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
FRAME_HISTORY = 90  # 約3秒 (30fps前提)

def preprocess_frame(frame):
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = frame.astype(np.float32) / 255.0
    frame = torch.tensor(frame).permute(2, 0, 1)  # (C, H, W)
    return frame

def main():
    # モデルの読み込み
    model = GazeMobileNetV3().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 動画読み込み
    cap = cv2.VideoCapture(VIDEO_PATH)
    assert cap.isOpened(), f"Cannot open video: {VIDEO_PATH}"
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_queue = deque(maxlen=FRAME_HISTORY)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current = preprocess_frame(frame)
        frame_queue.append(current)

        # 十分な履歴がある場合のみ処理
        if len(frame_queue) < FRAME_HISTORY:
            continue

        past = frame_queue[0]  # 最も古い（約3秒前）のフレーム

        # 合成
        past = past * 0.3  # 過去フレームの重み
        stacked = torch.cat([current, past], dim=0).unsqueeze(0).to(DEVICE)  # (1, 6, H, W)

        with torch.no_grad():
            gaze = model(stacked)[0].cpu().numpy()  # (x, y)

        # 可視化
        h, w = frame.shape[:2]
        gaze_x = int(gaze[0] * w)
        gaze_y = int(gaze[1] * h)
        cv2.circle(frame, (gaze_x, gaze_y), 10, (0, 255, 0), -1)
        cv2.imshow("Gaze Estimation", frame)

        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
