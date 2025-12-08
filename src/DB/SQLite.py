import sqlite3

conn = sqlite3.connect("AI.db")

cursor = conn.cursor()


cursor.execute("""
INSERT INTO 食べ物(食べ物ID,食べ物名)
VALUES(1,'リンゴ')
""")




conn.commit()

conn.close()
