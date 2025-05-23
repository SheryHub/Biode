import sqlite3

conn = sqlite3.connect("databases/observations.db")
cursor = conn.cursor()

# Create the observations table
cursor.execute('''
CREATE TABLE IF NOT EXISTS observations (
    id_observation INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    common_name TEXT,
    coordinates TEXT,
    notes TEXT
)
''')

conn.commit()
conn.close()

print("✅ Observation database initialized.")
