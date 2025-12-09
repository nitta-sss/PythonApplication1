import sqlite3
import tkinter as tk
from tkinter import ttk, messagebox

DB_PATH = "AI.db"   # ← 必要に応じてパス修正

"""
DB 挿入処理
"""
def insert_to_db():
    speaker_name = speaker_var.get()
    text = text_entry.get()
    emotion_name = emotion_var.get()

    if text.strip() == "":
        messagebox.showwarning("エラー", "テキストを入力してください")
        return

    # 発言者ID を Speaker テーブルから取得
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT 発言者ID FROM Speaker WHERE 発言者名 = ?", (speaker_name,))
    result = cur.fetchone()

    if result is None:
        messagebox.showerror("エラー", "発言者が Speaker テーブルに存在しません")
        conn.close()
        return

    speaker_id = result[0]

    # 感情ID を Emotion テーブルから取得
    cur.execute("SELECT 感情ID FROM Emotion WHERE 感情名 = ?", (emotion_name,))
    result = cur.fetchone()

    if result is None:
        messagebox.showerror("エラー", "Emotion テーブルに感情が存在しません")
        conn.close()
        return

    emotion_id = result[0]

    # DBに挿入
    cur.execute("""
        INSERT INTO TestData (発言者ID, テキストデータ, 感情ID)
        VALUES (?, ?, ?)
    """, (speaker_id, text, emotion_id))

    conn.commit()
    conn.close()

    messagebox.showinfo("成功", "データを追加しました！")
    text_entry.delete(0, tk.END)


"""
UI
"""
root = tk.Tk()
root.title("学習データ入力ツール")
root.geometry("380x300")

# 発言者ラベル
tk.Label(root, text="発言者（Speaker）").pack(pady=5)
speaker_var = tk.StringVar()

# 発言者一覧を DB から取得して Combobox に設定
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

# テキスト入力
tk.Label(root, text="テキスト内容").pack(pady=5)
text_entry = tk.Entry(root, width=40)
text_entry.pack()

# 感情選択
tk.Label(root, text="感情ラベル（Emotion）").pack(pady=5)
emotion_var = tk.StringVar()

def load_emotions():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT 感情名 FROM Emotion")
    emotions = [row[0] for row in cur.fetchall()]
    conn.close()
    return emotions

emotion_box = ttk.Combobox(root, textvariable=emotion_var, values=load_emotions())
emotion_box.current(0)
emotion_box.pack()

# DB追加ボタン
insert_btn = tk.Button(root, text="DB に追加", command=insert_to_db, width=20)
insert_btn.pack(pady=20)

root.mainloop()
