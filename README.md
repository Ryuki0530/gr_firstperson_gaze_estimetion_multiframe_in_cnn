# マルチフレーム視線推定システム

GTEAGaze＋データセットを使用した一人称視点動画における視線推定プロジェクトです。現在のフレームと過去のフレーム情報を組み合わせてCNNでに入力することで、精度向上を図ることを目標としています。

## プロジェクト概要

このプロジェクトは、GTEA Gaze+データセットを使用して、一人称視点での視線推定を行う深層学習システムです。単一フレーム処理ではなく、過去のフレームと組み合わせた入力を行うことにより、より安定した視線推定を目指しています。

### 主な特徴

- **マルチフレーム処理**: 現在フレーム + 過去フレーム(3秒前)の情報を活用
- **CNN特徴融合**: ２フレームの情報を単一のモデルに入力
- **リアルタイム推論**: 単一動画での即座な視線推定

## プロジェクト構造

```
./
│
├── data/
│   ├── raw/                           # GTEA Gaze+ の .avi / .txt
│   └── processed/
│       ├── gtea_train.npz             # 学習用データ
│       ├── gtea_val.npz               # 検証用データ
│       └── gtea_single_with_past.npz  # 前処理済み (現+過去フレーム + ラベル)
│
├── models/
│   └── gaze_cnn_fusion.py             # CNNベース特徴抽出＋結合モデル
│
├── scripts/
│   └── preprocess_single_with_past.py # .npz生成前処理スクリプト
│
├── outputs/
│   ├── checkpoints/                   # 保存モデル (.pth)
│   ├── logs/                          # 学習中のログ・loss可視化
│   └── results/                       # 推論結果・ヒートマップ等
│
├── train.py                           # 学習実行スクリプト
├── inference_realtime.py              # 推論実行スクリプト
├── requirements.txt                   # ライブラリ依存関係
└── README.md
```

## セットアップ

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 2. データセットの準備

1. GTEA Gaze+データセットを `data/raw/` ディレクトリに配置
2. 前処理スクリプトを実行:

```bash
python scripts/preprocess_single_with_past.py
```
3. 作成したデータを学習用と検証用に分離するため下記のスクリプトを実行:

```bash 
python scripts/split_data.py
```


## 使用方法

### 学習
学習を開始するには、以下のコマンドを実行してください。学習は `train/train.py` 内で定義された設定に基づいて行われます。
```bash
python train/train.py --epochs <エポック数を指定>
```

### 推論
inference_realtime.py内の使用するモデルと動画のpathを各環境に合わせて書き換え、実行してください。
```bash
python inference_realtime.py
```

## 主要コンポーネント

### データ処理
- **前処理**: マルチフレームデータの生成と正規化

### モデルアーキテクチャ
- **特徴抽出**: CNNベースのフレーム別特徴抽出
- **時系列融合**: 複数フレーム特徴の効果的な結合
- **視線推定**: 回帰または分類による視線座標予測

### モデルアーキテクチャ
- **リアルタイム推論**: 実装のサンプル

## 期待される成果
- リアルタイム処理での視線推定

## ライセンス
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
