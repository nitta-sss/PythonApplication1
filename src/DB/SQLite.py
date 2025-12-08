import sqlite3

conn = sqlite3.connect("AI.db")

cursor = conn.cursor()


cursor.execute("""
DROP TABLE 食べ物
""")




conn.commit()

conn.close()
