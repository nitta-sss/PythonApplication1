import sqlite3

conn = sqlite3.connect("AI.db")

cursor = conn.cursor()


cursor.execute("""
            SELECT t.テキストデータ, k.感情ID, k.感情名
            FROM TestData t 
            JOIN Emotion k ON t.感情ID = k.感情ID
""")




conn.commit()

conn.close()
