from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
import pickle
 
# tokenizer 読み込み
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
 
# モデル読み込み（mseバグ対策）
model = load_model(
    "emotion_model_bilstm.h5",
    custom_objects={
        "mse": MeanSquaredError(),
        "mae": MeanAbsoluteError()
    }
)
 
def get_emotion_values(text):
    seq = tokenizer.texts_to_sequences([text])
    x = pad_sequences(seq, maxlen=30)
    pred = model.predict(x)[0]
    valence, arousal = float(pred[0]), float(pred[1])
    return valence, arousal
"""
from predict_emotion import get_emotion_values
 
text = "怒りで震えてる！許せない！！"
val, aro = get_emotion_values(text)
 
print("予測 快-不快度:", val)
print("予測 覚醒度:", aro)
"""
 