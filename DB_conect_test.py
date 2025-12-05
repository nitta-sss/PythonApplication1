import mysql.connector

# MySQLに接続
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="password"
)

# カーソルを取得
cursor = conn.cursor()

# データベース作成
cursor.execute("CREATE DATABASE IF NOT EXISTS mydatabase")

# 接続を閉じる
cursor.close()
conn.close()