import sounddevice as sd
import numpy as np


def list_audio_devices():
    """オーディオデバイスの詳細情報を表示する関数"""
    devices = sd.query_devices()

    print("\n=== オーディオ出力デバイス一覧 ===")
    print("\n[出力デバイス]")
    for idx, device in enumerate(devices):
        if device['max_output_channels'] > 0:
            print(f"\nデバイスID: {idx}")
            print(f"デバイス名: {device['name']}")
            print(f"チャンネル数: {device['max_output_channels']}")
            print(f"サンプルレート: {device['default_samplerate']}Hz")
            print(f"低レイテンシー対応: {device['default_low_output_latency']:.3f}秒")
            print(f"高レイテンシー対応: {device['default_high_output_latency']:.3f}秒")

    print("\n[入力デバイス]")
    for idx, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"\nデバイスID: {idx}")
            print(f"デバイス名: {device['name']}")
            print(f"チャンネル数: {device['max_input_channels']}")
            print(f"サンプルレート: {device['default_samplerate']}Hz")
            print(f"低レイテンシー対応: {device['default_low_input_latency']:.3f}秒")
            print(f"高レイテンシー対応: {device['default_high_input_latency']:.3f}秒")

    # デフォルトデバイスの情報表示
    default_input = sd.query_devices(kind='input')
    default_output = sd.query_devices(kind='output')

    print("\n=== デフォルトデバイス ===")
    print(f"デフォルト入力デバイス: ID {default_input['index']} - {default_input['name']}")
    print(f"デフォルト出力デバイス: ID {default_output['index']} - {default_output['name']}")


def test_audio_device(device_id, duration=1.0, frequency=440.0):
    """
    指定されたデバイスでテスト音を再生する関数

    Parameters:
    - device_id: テストする出力デバイスのID
    - duration: テスト音の長さ（秒）
    - frequency: テスト音の周波数（Hz）
    """
    try:
        # デバイスの情報を取得
        device_info = sd.query_devices(device_id)
        samplerate = int(device_info['default_samplerate'])

        # テスト用の正弦波を生成
        t = np.linspace(0, duration, int(samplerate * duration), False)
        test_tone = 0.3 * np.sin(2 * np.pi * frequency * t)  # 音量を0.3に設定

        print(f"\nデバイス '{device_info['name']}' でテスト音を再生します...")
        sd.play(test_tone, samplerate, device=device_id)
        sd.wait()  # 再生が終わるまで待機
        print("テスト音の再生が完了しました")

    except Exception as e:
        print(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    # デバイス一覧を表示
    list_audio_devices()

    # テスト音を再生するかどうかを確認
    try:
        choice = input("\nテスト音を再生しますか？ (y/n): ")
        if choice.lower() == 'y':
            device_id = int(input("テストする出力デバイスのIDを入力してください: "))
            test_audio_device(device_id)
    except ValueError:
        print("無効な入力です")
    except KeyboardInterrupt:
        print("\nプログラムを終了します")