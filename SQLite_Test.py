import sqlite3

# 接続
conn = sqlite3.connect("AI.db") #各自AI.dbの相対パス入れてね
cursor = conn.cursor()

# データの取得
cursor.execute("""
DELETE FROM TestData;
""")

rows = cursor.fetchall()  # すべての行を取得

for row in rows:
    print(row)

