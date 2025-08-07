import sqlite3
conn = sqlite3.connect("C:/Users/josem/Desktop/hdcompany-chatbot/hdcompany.db")
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())
conn.close()