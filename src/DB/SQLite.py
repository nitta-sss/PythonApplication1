import sqlite3

conn = sqlite3.connect("AI.db")

cursor = conn.cursor()


cursor.execute("""
CREATE TABLE TestData(
テストID INT AUTO_INCREMENT PRIMARY KEY,
発言者ID INT NOT NULL,
テキストデータ TEXT NOT NULL,      
感情ID INT NOT NULL,
FOREIGN KEY (発言者ID) REFERENCES 発言者ID(Speaker),
FOREIGN KEY (感情ID) REFERENCES 感情ID(Emotion)
)
""")




conn.commit()

conn.close()
