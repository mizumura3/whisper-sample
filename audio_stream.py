import re

import sounddevice as sd
import numpy as np
import subprocess
import queue
import threading

import torch
from silero_vad import load_silero_vad, get_speech_timestamps

# 音声ストリーム設定
SAMPLE_RATE = 16000
BLOCK_SIZE = 4096
CHANNELS = 1

# 使用するデバイスID（指定）
DEVICE_ID = 3  # MacBook Air のマイク

# Whisper.cpp 実行設定
WHISPER_PATH = "../../nagato0614-whisper.cpp/stream"
WHISPER_OPTIONS = [
    "-l", "ja",  # 日本語モード
    "-t", "8",   # スレッド数
    "-c", "2",   # デバイスの ID
    "--step", "0",  # ステップサイズ
    "--length", "2000",  # 長さ
    "-vth", "0.6",
    "-m", "../../nagato0614-whisper.cpp/models/ggml-small-q8_0.bin"  # モデルファイル
]

# 出力キュー
transcription_queue = queue.Queue()

# Silero VAD モデルのロード
# Silero VAD モデルのロード
vad_model_result = load_silero_vad()

# 結果が tuple の場合、モデル部分を取得
if isinstance(vad_model_result, tuple):
    vad_model = vad_model_result[0]
else:
    vad_model = vad_model_result

print(f"Loaded VAD model: {vad_model}")

def list_input_devices():
    print("Available audio input devices:")
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"{idx}: {device['name']}")

def log_speech_timestamps(speech_timestamps):
    """ログとして音声区間を表示"""
    for timestamp in speech_timestamps:
        start = timestamp['start'] / SAMPLE_RATE
        end = timestamp['end'] / SAMPLE_RATE
        print(f"Detected speech from {start:.2f}s to {end:.2f}s")

def extract_text_from_transcription(transcription):
    """タイムスタンプ付きの文字列からテキスト部分のみを抽出"""
    match = re.search(r']\s*(.*)', transcription)
    return match.group(1) if match else transcription

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Audio status: {status}")

    # マイクからの音声データを Whisper に送信
    audio = (indata[:, 0] * 32768).astype(np.int16)
    wav_tensor = torch.from_numpy(audio / 32768.0).float().unsqueeze(0)

    max_amplitude = np.max(np.abs(audio))
    if max_amplitude < 1000:  # 音量が低い場合は無音と判断
        # print("Low amplitude detected, not sending to Whisper", flush=True)
        return

    # デバッグ表示
    speech_timestamps = get_speech_timestamps(
        audio=wav_tensor,
        model=vad_model,
        sampling_rate=SAMPLE_RATE,
        threshold=0.3
    )
    # VAD の結果をデバッグ表示
    # print(f"Speech timestamps: {speech_timestamps}", flush=True)

    if speech_timestamps:
        # print("Speech detected:", speech_timestamps, flush=True)
        log_speech_timestamps(speech_timestamps)  # 音声区間をログ出力
        try:
            process.stdin.write(audio.tobytes())
        except Exception as e:
            print(f"Error writing to Whisper: {e}")
    # else:
    #     print("Silence detected", flush=True)

def main():
    global process
    try:
        # Whisper.cpp をサブプロセスとして起動
        process = subprocess.Popen(
            [WHISPER_PATH] + WHISPER_OPTIONS,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
        )

        # Whisper の出力を別スレッドで読み取る
        def read_whisper_output():
            for line in iter(process.stdout.readline, ""):
                transcription = line.strip().decode('utf-8', errors='ignore')
                text = extract_text_from_transcription(transcription)
                print("text:", text, flush=True)


        # Whisper のエラー出力を別スレッドで読み取る（必要に応じてスキップ）
        def read_whisper_error():
            for line in iter(process.stderr.readline, b""):
                error_message = line.decode('utf-8').strip()
                # if "ggml_backend_metal_buffer_type_alloc_buffer" in error_message or "recommendedMaxWorkingSetSize" in error_message:
                #     continue  # 抑制したいログをスキップ
                print("Whisper error:", error_message)

        thread_output = threading.Thread(target=read_whisper_output, daemon=True)
        thread_error = threading.Thread(target=read_whisper_error, daemon=True)
        thread_output.start()
        thread_error.start()

        list_input_devices()  # 入力デバイス一覧を表示

        # マイク入力のストリームを開始
        with sd.InputStream(
            device=DEVICE_ID, samplerate=SAMPLE_RATE, channels=CHANNELS, blocksize=BLOCK_SIZE, callback=audio_callback
        ):
            print("Listening... Press Ctrl+C to stop.")
            while True:
                pass
    except KeyboardInterrupt:
        print("Stopping...")
        process.terminate()
        process.wait()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
