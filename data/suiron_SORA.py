import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# モデル読み込み
model = load_model("emotion_model.h5")

# tokenizer 読み込み
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

emotion_labels = {
    0: "neutral",
    1: "joy",
    2: "happy",
    3: "anger",
    4: "sad",
    5: "surprise",
    6: "worry"
}

def predict_emotion(text):
    # 1. Tokenizer で数値化
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=30)

    # 2. モデルで予測
    pred = model.predict(seq)[0]

    # 3. 最も確率が高い感情IDを取る
    emotion_id = pred.argmax()

    # 4. 感情名に変換
    emotion_name = emotion_labels[emotion_id]

    return emotion_id, emotion_name, pred

text = "めっちゃ嬉しい！"
emotion_id, emotion_name, pred = predict_emotion(text)

print("テキスト:", text)
print("予測ID:", emotion_id)
print("感情:", emotion_name)
print("確率分布:", pred)

