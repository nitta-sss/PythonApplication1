import sqlite3
import tkinter as tk
from tkinter import ttk, messagebox

DB_PATH = "AI.db"   # ← 必要に応じてパス修正


emotion_jp_to_en = {
    "喜び": "joy",
    "楽しい": "happy",
    "怒り": "anger",
    "悲しい": "sad",
    "無感情": "neutral",
    "驚き": "surprise",
    "心配": "worry"
}

"""
DB 挿入処理
"""
def insert_to_db():
    speaker_name = speaker_var.get()
    text = text_entry.get()
    emotion_jp = emotion_var.get()  # ←日本語

    if text.strip() == "":
        messagebox.showwarning("エラー", "テキストを入力してください")
        return

    # 日本語 → 英語に変換
    emotion_name = emotion_jp_to_en[emotion_jp]

    # DB接続
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 発言者ID取得
    cur.execute("SELECT 発言者ID FROM Speaker WHERE 発言者名 = ?", (speaker_name,))
    speaker_id = cur.fetchone()[0]

    # 感情ID取得（英語で検索）
    cur.execute("SELECT 感情ID FROM Emotion WHERE 感情名 = ?", (emotion_name,))
    emotion_id = cur.fetchone()[0]

    # 挿入
    cur.execute("""
        INSERT INTO TestData (発言者ID, テキストデータ, 感情ID)
        VALUES (?, ?, ?)
    """, (speaker_id, text, emotion_id))

    conn.commit()
    conn.close()

    messagebox.showinfo("成功", f"『{text}』をDBに追加しました！")
    text_entry.delete(0, tk.END)


"""
UI
"""
root = tk.Tk()
root.title("学習データ入力ツール")
root.geometry("380x320")

# 発言者
tk.Label(root, text="発言者（Speaker）").pack(pady=5)
speaker_var = tk.StringVar()

def load_speakers():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT 発言者名 FROM Speaker")
    speakers = [row[0] for row in cur.fetchall()]
    conn.close()
    return speakers

speaker_box = ttk.Combobox(root, textvariable=speaker_var, values=load_speakers())
speaker_box.current(0)
speaker_box.pack()

# テキスト
tk.Label(root, text="テキスト内容").pack(pady=5)
text_entry = tk.Entry(root, width=40)
text_entry.pack()

# 感情（日本語表示）
tk.Label(root, text="感情ラベル（日本語）").pack(pady=5)
emotion_var = tk.StringVar()
emotion_box = ttk.Combobox(
    root, 
    textvariable=emotion_var,
    values=list(emotion_jp_to_en.keys())  # ←日本語一覧
)
emotion_box.current(0)
emotion_box.pack()

# ボタン
insert_btn = tk.Button(root, text="DB に追加", command=insert_to_db, width=20)
insert_btn.pack(pady=20)

root.mainloop()