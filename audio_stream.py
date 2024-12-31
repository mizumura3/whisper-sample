import re
import os
from dotenv import load_dotenv
import requests
import sounddevice as sd
import soundfile as sf
import io
import numpy as np
import subprocess
import queue
import threading
import torch
from scipy import signal
from silero_vad import load_silero_vad, get_speech_timestamps

# Load environment variables
load_dotenv()

# 音声ストリーム設定
SAMPLE_RATE = int(os.getenv('SAMPLE_RATE', '16000'))
BLOCK_SIZE = int(os.getenv('BLOCK_SIZE', '4096'))
CHANNELS = int(os.getenv('CHANNELS', '1'))

# デバイスID設定
DEVICE_ID = int(os.getenv('AUDIO_INPUT_DEVICE_ID', '3'))
AUDIO_OUTPUT_DEVICE_ID = int(os.getenv('AUDIO_OUTPUT_DEVICE_ID', '4'))

# Whisper.cpp 実行設定
WHISPER_PATH = os.getenv('WHISPER_PATH', '../../nagato0614-whisper.cpp/stream')
WHISPER_MODEL_PATH = os.getenv('WHISPER_MODEL_PATH', '../../nagato0614-whisper.cpp/models/ggml-small-q8_0.bin')

WHISPER_OPTIONS = [
    "-l", os.getenv('WHISPER_LANGUAGE', 'ja'),
    "-t", os.getenv('WHISPER_THREADS', '8'),
    "-c", os.getenv('WHISPER_DEVICE_ID', '2'),
    "--step", os.getenv('WHISPER_STEP', '0'),
    "--length", os.getenv('WHISPER_LENGTH', '5000'),
    "-vth", os.getenv('WHISPER_THRESHOLD', '0.6'),
    "-m", WHISPER_MODEL_PATH
]

# API設定
NIJIVOICE_API_URL = os.getenv('NIJIVOICE_API_URL', 'https://api.nijivoice.com/api/platform/v1/voice-actors')
NIJIVOICE_VOICE_ACTOR_ID = os.getenv('NIJIVOICE_VOICE_ACTOR_ID', 'dba2fa0e-f750-43ad-b9f6-d5aeaea7dc16')
NIJIVOICE_API_KEY = os.getenv('NIJIVOICE_API_KEY')

# 出力キュー
transcription_queue = queue.Queue()

# VADモデルのロード
vad_model_result = load_silero_vad()
vad_model = vad_model_result[0] if isinstance(vad_model_result, tuple) else vad_model_result
print(f"Loaded VAD model: {vad_model}")


def apply_lowpass_filter(audio_data, sample_rate, cutoff_freq):
    """ローパスフィルタを適用する"""
    nyquist = sample_rate / 2.0
    normalized_cutoff_freq = cutoff_freq / nyquist
    # butterは2つのパラメータ（b, a）のみを返すので、その2つだけを受け取る
    b, a = signal.butter(N=4, Wn=normalized_cutoff_freq, btype='low')
    return signal.filtfilt(b, a, audio_data)


def high_quality_resample(audio_data, source_rate, target_rate):
    """高品質なリサンプリングを行う"""
    # 1. 最初にローパスフィルタを適用（ナイキスト周波数の80%をカットオフ周波数として使用）
    cutoff_freq = min(source_rate, target_rate) * 0.4
    filtered_audio = apply_lowpass_filter(audio_data, source_rate, cutoff_freq)

    # 2. リサンプリングの比率を計算
    resample_ratio = target_rate / source_rate

    # 3. リサンプリング用の高品質なFIRフィルタを設計
    if resample_ratio > 1:
        # アップサンプリングの場合
        filter_length = int(64 * max(1, resample_ratio))
    else:
        # ダウンサンプリングの場合
        filter_length = int(64 * max(1, 1 / resample_ratio))

    # 4. resample_polyで高品質なリサンプリングを実行
    resampled = signal.resample_poly(filtered_audio,
                                     up=target_rate,
                                     down=source_rate,
                                     window=('kaiser', 5.0),
                                     padtype='line')

    # 5. 出力にも軽いローパスフィルタを適用してエッジを滑らかにする
    output_cutoff = target_rate * 0.4
    resampled = apply_lowpass_filter(resampled, target_rate, output_cutoff)

    return resampled


def normalize_audio(audio_data, target_rms=-20):
    """音声データのRMSを指定したdB値に正規化する"""
    current_rms = 20 * np.log10(np.sqrt(np.mean(audio_data ** 2)))
    gain = 10 ** ((target_rms - current_rms) / 20)
    normalized = audio_data * gain

    # クリッピング防止
    max_val = np.max(np.abs(normalized))
    if max_val > 1.0:
        normalized = normalized / max_val

    return normalized


def play_audio_from_url(url, device_id=AUDIO_OUTPUT_DEVICE_ID):
    """URLから音声ファイルをダウンロードして特定のデバイスで再生する"""
    try:
        device_info = sd.query_devices(device_id)
        device_sample_rate = int(device_info['default_samplerate'])
        print(f"Using audio device: {device_info['name']}")

        response = requests.get(url)
        response.raise_for_status()

        audio_data, source_sample_rate = sf.read(io.BytesIO(response.content))

        if len(audio_data.shape) == 1:
            audio_data = np.column_stack((audio_data, audio_data))

        audio_data = audio_data.astype(np.float32)

        print(f"Source sample rate: {source_sample_rate}Hz")
        print(f"Device sample rate: {device_sample_rate}Hz")

        # 常に44.1kHzにリサンプリング
        target_sample_rate = 44100

        if source_sample_rate == target_sample_rate:
            print("Sample rates match - no resampling needed")
        else:
            print(f"Sample rates differ - using high-quality resampling")
            print(f"Resampling from {source_sample_rate}Hz to {device_sample_rate}Hz")
            resampled_channels = []
            for channel in range(audio_data.shape[1]):
                resampled = high_quality_resample(
                    audio_data[:, channel],
                    source_sample_rate,
                    target_sample_rate
                )
                resampled_channels.append(resampled)
            audio_data = np.column_stack(resampled_channels)

        # 音声の正規化（RMSベース）
        audio_data = normalize_audio(audio_data)

        try:
            sd.play(audio_data, target_sample_rate, device=device_id)
            sd.wait()
        except sd.PortAudioError as e:
            print(f"Error during playback: {e}")

    except Exception as e:
        print(f"Error playing audio: {e}")
        raise


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

    audio = (indata[:, 0] * 32768).astype(np.int16)
    wav_tensor = torch.from_numpy(audio / 32768.0).float().unsqueeze(0)

    max_amplitude = np.max(np.abs(audio))
    if max_amplitude < int(os.getenv('AMPLITUDE_THRESHOLD', '1000')):
        return

    speech_timestamps = get_speech_timestamps(
        audio=wav_tensor,
        model=vad_model,
        sampling_rate=SAMPLE_RATE,
        threshold=float(os.getenv('VAD_THRESHOLD', '0.3'))
    )

    if speech_timestamps:
        log_speech_timestamps(speech_timestamps)
        try:
            process.stdin.write(audio.tobytes())
        except Exception as e:
            print(f"Error writing to Whisper: {e}")

def main():
    global process
    try:
        process = subprocess.Popen(
            [WHISPER_PATH] + WHISPER_OPTIONS,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
        )

        def read_whisper_output():
            last_valid_text = ""
            previous_text = ""
            audio_queue = queue.Queue()

            def audio_worker():
                while True:
                    url = audio_queue.get()
                    if url is None:
                        break
                    play_audio_from_url(url)
                    audio_queue.task_done()

            audio_thread = threading.Thread(target=audio_worker, daemon=True)
            audio_thread.start()

            for line in iter(process.stdout.readline, ""):
                transcription = line.strip().decode('utf-8', errors='ignore')
                text = extract_text_from_transcription(transcription)
                cleaned_text = text.replace('[2K', '')
                lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]

                if lines:
                    current_text = lines[-1]
                    if "Transcription" in current_text and ("START" in current_text or "END" in current_text):
                        continue

                    if current_text != previous_text:
                        last_valid_text = current_text
                        previous_text = current_text
                        print("text:", last_valid_text, flush=True)

                        url = f"{NIJIVOICE_API_URL}/{NIJIVOICE_VOICE_ACTOR_ID}/generate-voice"
                        payload = {
                            "script": last_valid_text,
                            "speed": os.getenv('VOICE_SPEED', '1'),
                            "format": os.getenv('AUDIO_FORMAT', 'wav'),
                        }
                        headers = {
                            "accept": "application/json",
                            "content-type": "application/json",
                            "x-api-key": NIJIVOICE_API_KEY
                        }

                        try:
                            response = requests.post(url, json=payload, headers=headers)
                            response.raise_for_status()
                            response_data = response.json()

                            if 'generatedVoice' in response_data and 'audioFileUrl' in response_data['generatedVoice']:
                                audio_url = response_data['generatedVoice']['audioFileUrl']
                                audio_queue.put(audio_url)

                        except Exception as e:
                            print(f"Error with API request: {e}")

        def read_whisper_error():
            for line in iter(process.stderr.readline, b""):
                error_message = line.decode('utf-8').strip()
                print("Whisper info:", error_message)

        thread_output = threading.Thread(target=read_whisper_output, daemon=True)
        thread_error = threading.Thread(target=read_whisper_error, daemon=True)
        thread_output.start()
        thread_error.start()

        list_input_devices()

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