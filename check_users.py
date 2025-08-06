import sqlite3
conn = sqlite3.connect("C:/Users/josem/Desktop/hdcompany-chatbot/hdcompany.db")
cursor = conn.cursor()
cursor.execute("SELECT username, hashed_password FROM users WHERE username = 'admin'")
print(cursor.fetchall())
conn.close()