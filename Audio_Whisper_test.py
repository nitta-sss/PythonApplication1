import pyaudio
import numpy as np
import wave
import time
from datetime import datetime
from faster_whisper import WhisperModel
import threading
import keyboard

# -----------------------------
# 設定
# -----------------------------
SAMPLE_RATE = 16000     # Whisper推奨
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1024            # 1回に読むフレーム数
SILENCE_THRESHOLD = 600 # 無音判定の音量閾値
SILENCE_DURATION = 1  # 無音が0.8秒続いたら「発話終了」
OUTPUT_FILE = datetime.now().strftime("%Y%m%d_%H%M") + ".txt"

# Whisperモデル
model = WhisperModel("small", device="cpu", compute_type="int8")

# 音声バッファ
audio_buffer = []
last_voice_time = time.time()
lock = threading.Lock()

# ---------------------------------------
# 無音判定（RMS）
# ---------------------------------------
def is_silent(data):
    audio = np.frombuffer(data, dtype=np.int16)
    rms = np.sqrt(np.mean(audio**2))
    return rms < SILENCE_THRESHOLD

# ---------------------------------------
# 音声認識（Whisper）
# ---------------------------------------
def transcribe_audio(wav_path):
    segments, info = model.transcribe(wav_path, beam_size=3,language="ja") #beam_size:音声の候補の数(1=高速だけど誤認しやすい　5=遅いけど正確)
    text = "".join([seg.text for seg in segments])
    return text

# ---------------------------------------
# 発話が終わったら Whisper で認識
# ---------------------------------------
def process_buffer():
    global audio_buffer
    if not audio_buffer:
        return

    wav_path = "temp.wav"
    
    # wav保存
    wf = wave.open(wav_path, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b"".join(audio_buffer))
    wf.close()

    # Whisperで認識
    text = transcribe_audio(wav_path)

    # テキスト保存
    if text.strip():
        print(">>", text)
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    # バッファクリア
    audio_buffer = []


# ---------------------------------------
# メイン：マイク読み取りループ
# ---------------------------------------
def main():
    global audio_buffer, last_voice_time

    print("🎤 リアルタイム文字起こし開始（Whisper / オフライン）")

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
 
    try:
        while True:
            data = stream.read(CHUNK)

            with lock:
                audio_buffer.append(data)

            # 音がある？
            if not is_silent(data):
                last_voice_time = time.time()
            else:
                # 無音が一定時間続いた？
                if time.time() - last_voice_time > SILENCE_DURATION: #現在の時刻　－　最後に音があった時刻　＞　無音間隔
                    with lock:
                        print("Whisper処理開始")
                        process_buffer()
                        
                    last_voice_time = time.time()
    except KeyboardInterrupt:
        print("\n Ctrl+C detected Stopping,,,")

    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


if __name__ == "__main__":
    main()
