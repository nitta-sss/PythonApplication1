import sqlite3

# 接続
conn = sqlite3.connect("PythonApplication1/AI.db")
cursor = conn.cursor()

# データの取得
cursor.execute("""
SELECT * FROM TestData; 
""")

rows = cursor.fetchall()  # すべての行を取得

for row in rows:
    print(row)

