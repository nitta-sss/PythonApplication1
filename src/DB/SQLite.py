import sqlite3

conn = sqlite3.connect("AI.db")

cursor = conn.cursor()


cursor.execute("""
DROP TABLE sqlite_sequence;
""")




conn.commit()

conn.close()
