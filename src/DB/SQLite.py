import sqlite3

conn = sqlite3.connect("AI.db")

cursor = conn.cursor()


cursor.execute("""
INSERT INTO テストデータ(テストID,発言者ID,テキストデータ,感情ID)
VALUES (7,1,'やったー！',1),(8,1,'おもろ',2),(9,1,'くそが',3),(10,1,'ひどい',4),(11,1,'まじか',6),(12,1,'どしたん、話聞こか？',7)
""")




conn.commit()

conn.close()
