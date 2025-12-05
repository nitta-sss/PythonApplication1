from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# ① モデルとトークナイザを読み込む
model = load_model("emotion_model.h5",compile=False)

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ② 推論関数
def predict_emotion(text):
    # テキストを数値化
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=20)

    # 推論
    pred = model.predict(seq)[0]

    valence = float(pred[0])   # 快・不快（横軸）
    arousal = float(pred[1])   # 覚醒・沈静（縦軸）

    return valence, arousal

# ③ テスト
text = "眠いなぁ"
v, a = predict_emotion(text)

print("Valence（快-不快）:", v)
print("Arousal（覚醒-沈静）:", a)
