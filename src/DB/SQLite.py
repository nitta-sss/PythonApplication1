import sqlite3

conn = sqlite3.connect("AI.db")

cursor = conn.cursor()


cursor.execute("""
INSERT INTO 感情(感情ID,感情名)
VALUES (1,'喜び'),(2,'楽しい'),(3,'怒り'),(4,'悲しみ'),(5,'無感情'),(6,'驚き'),(7,'心配')
""")




conn.commit()

conn.close()
