

import pyaudio
import numpy as np
import wave
import time
from datetime import datetime
from faster_whisper import WhisperModel
import threading
import keyboard
recording = False
# -----------------------------
# 設定
# -----------------------------

with open("C:/Users/232144/Desktop/HaLu/src/Audio/for_HaLu.txt",mode="r+")as f:
    f.truncate(0)#現在のファイルサイズを０にする

SAMPLE_RATE = 16000     # Whisper推奨
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1024            # 1回に読むフレーム数
OUTPUT_FILE = "C:/Users/232144/Desktop/HALU/src/Audio/"+datetime.now().strftime("For_HaLu") + ".txt"

# Whisperモデル
model = WhisperModel("small", device="cpu", compute_type="int8")

# 音声バッファ
audio_buffer = []
last_voice_time = time.time()
lock = threading.Lock()

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

    wav_path = "C:/Users/232144/Desktop/HALU/src/Audio/temp.wav"
    
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
            with lock:
                if recording:
                    print("録音中...")
                    data = stream.read(CHUNK)
                    audio_buffer.append(data)
                    
                    if recording==False:
                        process_buffer()
                
                          
    except KeyboardInterrupt:
        print("\n Ctrl+C detected Stopping,,,")

    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


threading.Thread(target=main, daemon=True).start()
# ホットキーで録音開始・停止
keyboard.on_press_key("r", lambda e: toggle_record())


def toggle_record():
    global recording
    recording = not recording
    if recording:
        print("録音開始")
        audio_buffer = []  # 録音開始時に空のリストにする
    else:
        print("録音停止")




# メインスレッドはそのままターミナルで動かす
print("Rキーを押すと録音開始、離すと録音停止")
keyboard.wait()  # 無限ループで待機
if __name__ == "__main__":
    main()
