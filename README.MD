# Audio Stream Processing Application

リアルタイムの音声認識とテキスト読み上げを組み合わせたストリーミングアプリケーションです。入力された音声をWhisper.cppを使って認識し、認識されたテキストをNijiVoice APIを使って音声合成します。

## 機能

- リアルタイム音声入力とVAD（Voice Activity Detection）による音声区間検出
- Whisper.cppを使用したリアルタイム音声認識
- NijiVoice APIを使用したテキスト読み上げ
- 高品質な音声処理（ローパスフィルター、リサンプリング、正規化）

## 必要条件

- Python 3.8以上
- Whisper.cpp
- CUDA対応GPU（Whisper.cppの実行に推奨）

## セットアップ

1. 依存パッケージのインストール:

```bash
uv sync
```

2. Whisper.cppのセットアップ:

```bash
# Whisper.cppをクローン
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp

# ストリーミング機能を含むバージョンをビルド
make stream

# モデルをダウンロード（例：small）
bash ./models/download-ggml-model.sh small

# モデルファイルの場所を確認
ls -l models/ggml-small.bin
```

注意：
- CUDAを使用する場合は`make stream CUDA=1`でビルドしてください
- ダウンロードできるモデルは他にも`tiny`、`base`、`medium`、`large`があります
- モデルのパスは`.env`ファイルの`WHISPER_MODEL_PATH`で指定します

3. 環境変数の設定:

`.env`ファイルを作成し、以下の設定を行ってください：

```env
# 音声ストリーム設定
SAMPLE_RATE=16000
BLOCK_SIZE=4096
CHANNELS=1

# デバイスID設定
AUDIO_INPUT_DEVICE_ID=3
AUDIO_OUTPUT_DEVICE_ID=4

# Whisper設定
WHISPER_PATH=../../nagato0614-whisper.cpp/stream
WHISPER_MODEL_PATH=../../nagato0614-whisper.cpp/models/ggml-small-q8_0.bin
WHISPER_LANGUAGE=ja
WHISPER_THREADS=8
WHISPER_DEVICE_ID=2
WHISPER_STEP=0
WHISPER_LENGTH=5000
WHISPER_THRESHOLD=0.6

# NijiVoice API設定
NIJIVOICE_API_URL=https://api.nijivoice.com/api/platform/v1/voice-actors
NIJIVOICE_VOICE_ACTOR_ID=dba2fa0e-f750-43ad-b9f6-d5aeaea7dc16
NIJIVOICE_API_KEY=xxxxx # あなたのAPIキーを設定してください

# 音声処理設定
AMPLITUDE_THRESHOLD=1000
VAD_THRESHOLD=0.3
VOICE_SPEED=1
AUDIO_FORMAT=wav
```

## 使用方法

1. アプリケーションの起動:

```bash
python audio_stream.py
```

2. 利用可能な音声入力デバイスが表示されます。

3. 音声入力を開始すると、以下の処理が実行されます：
   - VADによる音声区間の検出
   - Whisper.cppによる音声認識
   - NijiVoice APIによるテキスト読み上げ

4. 終了するには`Ctrl+C`を押してください。

## 主要な機能の説明

### 音声処理

- `apply_lowpass_filter`: ローパスフィルターを適用して高周波ノイズを除去
- `high_quality_resample`: 高品質な音声リサンプリングを実行
- `normalize_audio`: 音声のRMSレベルを適切な値に正規化

### 音声認識と合成

- VADモデルによる音声区間の検出
- Whisper.cppによるリアルタイム音声認識
- NijiVoice APIを使用した音声合成

## エラーハンドリング

- 音声デバイスのエラー処理
- API通信のエラー処理
- 音声処理中のエラー処理

## 注意事項

- 音声入力デバイスとWhisper.cppの設定は環境に応じて適切に調整してください
- NijiVoice APIの利用には有効なAPIキーが必要です
- 処理性能はハードウェア環境に依存します
- APIキーなどの機密情報は必ず環境変数で管理し、Gitなどのバージョン管理システムにコミットしないでください
- `.env`ファイルは`.gitignore`に追加することを推奨します
