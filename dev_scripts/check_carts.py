import sqlite3
conn = sqlite3.connect("C:/Users/josem/Desktop/hdcompany-chatbot/hdcompany.db")
cursor = conn.cursor()
cursor.execute("SELECT * FROM carts WHERE user_phone = '4915153924850'")
print(cursor.fetchall())
conn.close()