import sqlite3

conn = sqlite3.connect("AI.db")

cursor = conn.cursor()


cursor.execute("""
UPDATE Emotion SET 感情名 = 'anger' WHERE 感情ID = 3
""")




conn.commit()

conn.close()
