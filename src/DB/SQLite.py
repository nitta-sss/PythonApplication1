import sqlite3

conn = sqlite3.connect("AI.db")

cursor = conn.cursor()


cursor.execute("""
INSERT INTO TestData(テストID,発言者ID,テキストデータ,感情ID)
VALUES (1,1,'うれしい',1),(2,1,'楽しみ',2),(3,1,'うるさい',3),(4,1,'さみしい',4),(5,1,'びっくり',5),(6,1,'大丈夫？',6),(7,1,'やったー！',1),(8,1,'おもろ',2),(9,1,'くそが',3),(10,1,'ひどい',4),(11,1,'まじか',5),(12,1,'どしたん、話聞こか？',6)

""")




conn.commit()

conn.close()
