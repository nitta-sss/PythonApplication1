from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# モデル読み込み
model = load_model("emotion_model.h5")

# tokenizer 読み込み
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# 感情ラベル（あなたのDB基準）
emotion_labels = {
    0: "joy",
    1: "happy",
    2: "anger",
    3: "sad",
    4: "neutral",
    5: "surprise",
    6: "worry"
}

def predict_emotion(text):
    # Tokenizerで数値化
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=30)

    # 推論
    pred = model.predict(seq)[0]

    # 一番高い確率の感情ID
    emotion_id = pred.argmax()

    # 感情名
    emotion_name = emotion_labels[emotion_id]

    return emotion_id, emotion_name, pred

emotion_id, emotion_name, pred = predict_emotion("かなしい")
print(emotion_id, emotion_name, pred)
