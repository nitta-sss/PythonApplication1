import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

texts = [
    "今日は最高に気分がいい！",
    "なんだかイライラして落ち着かない",
    "穏やかでゆったりした時間を過ごしている",
    "腹が立って怒鳴りそうだ",
    "少し緊張しているけど楽しみだ",
    "悲しくて胸が苦しい",
    "ワクワクが止まらない！",
    "眠くてぼーっとしている",
    "とても不安で胸がざわつく",
    "心地よい風でリラックスしている",

    "驚きすぎて頭が真っ白だ",
    "嫌な予感がして落ち着かない",
    "とても気分が落ち込んでいる",
    "満ち足りていて幸せな気持ちだ",
    "わけもなくムシャクシャする",
    "仕事がうまくいって誇らしい",
    "気持ちが沈んで動きたくない",
    "緊張で手が震えている",
    "とても安心している",
    "怒りが爆発しそうだ",

    "楽しすぎてテンションが上がる！",
    "疲れていて全くやる気が出ない",
    "焦って心がざわざわする",
    "心がほっこりして優しい気持ちだ",
    "恐ろしくて震えが止まらない",
    "とてもリラックスして気分がいい",
    "強い孤独感を感じている",
    "気持ちがすっきりして爽快だ",
    "緊張で息が詰まりそう",
    "落ち着き払って冷静に判断できる",

    "嫌悪感で顔をしかめてしまう",
    "深い悲しみに包まれている",
    "期待で胸が高鳴る",
    "ゆったりとした静かな時間を楽しんでいる",
    "強烈な怒りが湧いてくる",
    "プレッシャーで押しつぶされそうだ",
    "安心して眠りにつけそうだ",
    "明るく前向きな気持ちだ",
    "退屈で仕方がない",
    "恐怖で頭が真っ白になる",

    "やる気がみなぎっている",
    "だるくて何もしたくない",
    "嬉しくて涙が出そうだ",
    "ストレスが溜まりすぎて限界だ",
    "心から笑える気分だ",
    "不気味でぞっとする",
    "優しい気持ちに包まれている",
    "焦りが止まらず胸が苦しい",
    "軽い不満を感じている",
    "穏やかで心が満たされている",

    "とても怒って感情が抑えきれない",
    "感謝で胸がいっぱいだ",
    "緊張はするが前向きな気持ちだ",
    "絶望した気持ちで何も見えない",
    "好奇心で胸が躍っている",
    "静かで落ち着いた気持ちだ"
]

labels = [
    [0.9, 0.6],
    [-0.6, 0.5],
    [0.5, -0.6],
    [-0.8, 0.8],
    [0.7, 0.4],
    [-0.7, 0.2],
    [0.8, 0.9],
    [-0.1, -0.8],
    [-0.6, 0.6],
    [0.6, -0.7],

    [0.1, 0.9],
    [-0.5, 0.3],
    [-0.8, -0.3],
    [0.8, 0.2],
    [-0.6, 0.7],
    [0.7, 0.3],
    [-0.7, -0.7],
    [-0.4, 0.8],
    [0.7, -0.4],
    [-0.9, 0.9],

    [0.9, 0.8],
    [-0.4, -0.7],
    [-0.3, 0.5],
    [0.7, -0.3],
    [-0.9, 0.8],
    [0.6, -0.6],
    [-0.6, -0.4],
    [0.8, 0.4],
    [-0.4, 0.7],
    [0.3, -0.5],

    [-0.7, 0.5],
    [-0.9, -0.4],
    [0.8, 0.7],
    [0.5, -0.7],
    [-0.8, 0.8],
    [-0.6, 0.6],
    [0.6, -0.8],
    [0.9, 0.5],
    [-0.3, -0.5],
    [-0.8, 0.9],

    [0.8, 0.6],
    [-0.5, -0.7],
    [0.9, 0.4],
    [-0.8, 0.5],
    [0.9, 0.5],
    [-0.7, 0.4],
    [0.6, -0.3],
    [-0.4, 0.7],
    [-0.3, 0.2],
    [0.7, -0.5],

    [-0.9, 0.8],
    [0.9, 0.3],
    [0.6, 0.5],
    [-1.0, -0.2],
    [0.8, 0.7],
    [0.4, -0.6]
]

labels = np.array(labels)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

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


model.fit(x_train, labels, epochs=100, batch_size=4)
model.save("emotion_model.h5")



