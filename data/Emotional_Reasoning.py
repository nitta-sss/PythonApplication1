# predict_emotion.py
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
text = "明日の試験の勉強全くしてないから落ちるかもしれない"
val, aro = predict_emotion(text)

print("入力文：", text)
print("予測 Valence:", val)
print("予測 Arousal:", aro)
