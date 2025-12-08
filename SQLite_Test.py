import sqlite3

# 接続
conn = sqlite3.connect("AI.db") #各自AI.dbの相対パス入れてね
cursor = conn.cursor()

# データの取得
cursor.execute("""
INSERT INTO TestData(テストID,発言者ID,テキストデータ,感情ID)
VALUES (13,1,'NULLテスト',NULL)
""")

rows = cursor.fetchall()  # すべての行を取得

for row in rows:
    print(row)

