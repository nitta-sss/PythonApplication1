import sqlite3

# 接続
conn = sqlite3.connect("AI.db")
cursor = conn.cursor()

# データの取得
cursor.execute("""
SELECT t.テキストデータ, k.感情名
FROM TestData t 
JOIN Emotion k ON t.感情ID = k.感情ID
""")

rows = cursor.fetchall()  # すべての行を取得

for row in rows:
    print(row)

