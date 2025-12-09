import sqlite3

conn = sqlite3.connect("AI.db")

cursor = conn.cursor()


cursor.execute("""
ALTER TABLE TestData_new RENAME TO TestData;
""")




conn.commit()

conn.close()
