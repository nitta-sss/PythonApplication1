from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
import sqlite3

connection = sqlite3.connect("AI.db")
cursor = connection.cursor()

"""
DBからテキストと感情ラベルをもってくる
"""
# x書く (text)

# y書く (感情)


# モデル定義
model = Sequential([
    Embedding(input_dim=5000, output_dim=16, input_length=30),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(7, activation='softmax')  # ← 喜び,楽しい​,怒り,悲しみ,無感情,驚き,心配
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 学習
model.fit(X, y, epochs=10, batch_size=32)
# モデル保存
model.save("emotion_model.h5")
