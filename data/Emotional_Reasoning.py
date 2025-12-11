#predict_emotion.py
# ---------------------------
# 1. Python標準ライブラリ
# ---------------------------
import time
from datetime import datetime
import threading

# ---------------------------
# 2. 外部ライブラリ
# ---------------------------
import pyaudio
import keyboard
from faster_whisper import WhisperModel

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle



# Tokenizer 読み込み
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# モデル読み込み
model = load_model("emotion_model_bilstm.h5")

def predict_emotion(text):
    seq = tokenizer.texts_to_sequences([text])
    x = pad_sequences(seq, maxlen=30)
    pred = model.predict(x)[0]

    valence, arousal = pred[0], pred[1]
    return valence, arousal

# テスト実行
val, aro = predict_emotion(text)

print("入力文：", text)
print("予測 Valence:", val)
print("予測 Arousal:", aro)


