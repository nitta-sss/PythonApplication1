import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# ========== 1. モデル読込 ==========
model = load_model("emotion_model.h5")

# Tokenizer読込
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ========== 2. 感情→円環モデル座標マップ ==========
emotion_map = {
    0: {"name": "喜び",   "valence": +0.8, "arousal": +0.4},
    1: {"name": "楽しい", "valence": +0.9, "arousal": +0.6},
    2: {"name": "怒り",   "valence": -0.8, "arousal": +0.8},
    3: {"name": "悲しみ", "valence": -0.7, "arousal": -0.6},
    4: {"name": "無感情", "valence":  0.0, "arousal":  0.0},
    5: {"name": "驚き",   "valence": +0.2, "arousal": +0.8},
    6: {"name": "心配",   "valence": -0.4, "arousal": +0.3},
}

# ========== 3. 推論関数 ==========
def predict_emotion_and_plot(text):

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=30)

    # softmax 予測
    pred = model.predict(padded)[0]

    # 最も確率が高い感情
    top_idx = np.argmax(pred)
    top_emotion = emotion_map[top_idx]["name"]

    print(f"入力: {text}")
    print(f"判定感情: {top_emotion}")
    print(f"softmax: {pred}")

    # ===== 円環モデル座標を計算（加重平均） =====
    valence = sum(pred[i] * emotion_map[i]["valence"] for i in range(7))
    arousal = sum(pred[i] * emotion_map[i]["arousal"] for i in range(7))

    print(f"Valence: {valence:.3f}, Arousal: {arousal:.3f}")

    # ========== 4. プロット ==========
    plt.figure(figsize=(6, 6))
    plt.axhline(0, color="gray")
    plt.axvline(0, color="gray")

    # 座標を点で表示
    plt.scatter(valence, arousal, s=200)

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title("Russell Circumplex Model (簡易版)")
    plt.xlabel("Valence（快 ←→ 不快）")
    plt.ylabel("Arousal（沈静 ←→ 覚醒）")

    # 感情名を表示
    plt.text(valence + 0.03, arousal + 0.03, top_emotion, fontsize=12)

    plt.grid(True)
    plt.show()

# ========== 5. テスト ==========
predict_emotion_and_plot("うれしいなぁ")
