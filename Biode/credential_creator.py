# setup_db.py
import sqlite3

conn = sqlite3.connect("users.db")
cursor = conn.cursor()

# Create table
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT NOT NULL
)
''')

# Insert two users
users = [
    ("user1", "pass"),
    ("user2", "secure")
]
cursor.executemany("INSERT OR REPLACE INTO users (username, password) VALUES (?, ?)", users)

conn.commit()
conn.close()

print("Database initialized.")
