from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
import sqlite3
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

# DB取得
def load_data_db():
    conn = sqlite3.connect("AI.db")
    query = """
            SELECT t.テキストデータ, k.感情ID
            FROM TestData t 
            JOIN Emotion k ON t.感情ID = k.感情ID
            """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

df = load_data_db()
print(df.head())

# トークナイザーでテキストを数値化
texts = df["テキストデータ"].values

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

seq = tokenizer.texts_to_sequences(texts)
X = pad_sequences(seq, maxlen=30)

print("X の形：", X.shape)

# valence/arousal 定義
emotion_map = {
    0: {"valence": +0.8, "arousal": +0.4},  # 喜び
    1: {"valence": +0.9, "arousal": +0.6},  # 楽しい
    2: {"valence": -0.8, "arousal": +0.8},  # 怒り
    3: {"valence": -0.7, "arousal": -0.6},  # 悲しみ
    4: {"valence":  0.0, "arousal":  0.0},  # 無感情
    5: {"valence": +0.1, "arousal": +0.9},  # 驚き
    6: {"valence": -0.3, "arousal": +0.3},  # 心配
}

emotion_ids = df["感情ID"].astype(int).values

valence = np.array([emotion_map[i]["valence"] for i in emotion_ids])
arousal = np.array([emotion_map[i]["arousal"] for i in emotion_ids])

y_reg = np.vstack([valence, arousal]).T

print("y_reg の形：", y_reg.shape)

# BiLSTM モデル定義
model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=30),
    Bidirectional(LSTM(24, dropout=0.3, recurrent_dropout=0.3)),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(2, activation='tanh')  # valence, arousal
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="mse",
    metrics=["mae"]
)

model.summary()

# 学習
history = model.fit(
    X, y_reg,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)

# グラフ表示
plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Training Loss")
plt.legend()
plt.show()

plt.plot(history.history["mae"], label="mae")
plt.plot(history.history["val_mae"], label="val_mae")
plt.title("Training MAE")
plt.legend()
plt.show()

# モデル保存
model.save("emotion_model_bilstm.h5")
print("モデル学習完了！！！！")


# テスト用！！！
text = ["明日の試験の勉強全くしてないから落ちるかもしれない"]

seq = tokenizer.texts_to_sequences(text)
x = pad_sequences(seq, maxlen=30)

pred = model.predict(x)[0]
valence, arousal = pred[0], pred[1]

print("予測 Valence:", valence)
print("予測 Arousal:", arousal)
