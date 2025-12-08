import pickle

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

print(tokenizer.word_index)