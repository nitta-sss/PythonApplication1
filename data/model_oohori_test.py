from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
import sqlite3
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import matplotlib.pyplot as plt

# 12/09 memo テキスト用のもでるにしないといけないんじゃない？


"""
DBからテキストと感情ラベルをもってくる
"""
def load_data_db():
    conn = sqlite3.connect("AI.db")
    query = """
            SELECT t.テキストデータ, k.感情ID, k.感情名
            FROM TestData t 
            JOIN Emotion k ON t.感情ID = k.感情ID
            """
    df = pd.read_sql_query(query, conn)

    conn.close()
    return df

df = load_data_db()
print("取得したデータ：")
print(df.head(), "\n")


"""
x書く (テキスト)
"""
texts = df["テキストデータ"].values

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)

# Tokenizer 保存
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

seq = tokenizer.texts_to_sequences(texts)
X = pad_sequences(seq, maxlen=30)

print("X の形：", X.shape)


"""
y書く (感情)
"""
y = df["感情ID"].astype(int).values
print("y の形：", y.shape)


# モデル定義
model = Sequential([
    Embedding(input_dim=5000, output_dim=16, input_length=30),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(7, activation='softmax')  # ← 喜び,楽しい​,怒り,悲しみ,無感情,驚き,心配
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 学習
history = model.fit(
    X, y,
    epochs=200,
    batch_size=32,
    validation_split=0.2   # ← データの 20% を検証用に使う
)

"""
グラフ表示
"""

# Loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss') 
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy') 
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

print(history)
# モデル保存
model.save("emotion_model.h5")

print("モデル学習完了!!!!!!!!")