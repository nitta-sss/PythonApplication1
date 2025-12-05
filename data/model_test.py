import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten
import numpy as np

texts = [
    "今日はとても楽しい！",
    "最悪な気分だ……",
    "驚いた！なんてことだ",
    "なんか心配になってきた",
]

# ラッセルの円環モデル座標（例）
labels = [#[不快-覚醒-鎮静]
    [0.8, 0.7],   # 楽しい → Valence高, Arousal高
    [-0.7, 0.5],  # 怒り・不快
    [0.2, 0.9],   # 驚き
    [-0.5, -0.2], # 心配
]
labels = np.array(labels)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)

seqs = tokenizer.texts_to_sequences(texts)
x_train = pad_sequences(seqs, maxlen=20)


model = Sequential([
    Embedding(input_dim=5000, output_dim=32, input_length=20),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='linear')  # → Valence, Arousal
])

model.compile(
    optimizer='adam',
    loss='mse'
)

model.summary()


model.fit(x_train, labels, epochs=30, batch_size=4)
model.save("emotion_model.h5")



