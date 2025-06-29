# WD14 Tagger CLI

Stable Diffusion用のアニメ画像タガーのコマンドライン版です。WD14（Waifu Diffusion 1.4）モデルを使用して画像からタグを自動生成します。

## インストール

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本的な使用方法
```bash
python3 tagger.py image.jpg
```

出力例:
```
1girl, solo, long hair, brown hair, school uniform, looking at viewer, smile
```

### オプション

```bash
# モデルを指定
python3 tagger.py --model wd14-convnext-v2 image.jpg

# 閾値を変更（デフォルト: 0.35）
python3 tagger.py --threshold 0.5 image.jpg

# レーティング情報も表示
python3 tagger.py --show-ratings image.jpg

# 信頼度スコアも表示
python3 tagger.py --show-confidence image.jpg
```

### 利用可能なモデル

- `wd14-vit-v2` (デフォルト) - Vision Transformer モデル
- `wd14-convnext-v2` - ConvNeXT モデル  
- `wd14-swinv2-v1` - Swin Transformer V2 モデル

## 特徴

- 軽量なスタンドアロンCLIツール
- Hugging Faceから自動でモデルをダウンロード
- 複数のWD14モデルに対応
- カスタマイズ可能な閾値設定
- レーティング情報の表示

## 必要な環境

- Python 3.7+
- 対応画像形式: JPG, PNG, WebP等（PILでサポートされる形式）

## ライセンス

stable-diffusion-webui-wd14-taggerをベースにしています。