import sqlite3
import csv
from datetime import datetime

# Set today's date
today_date = "2025-05-17"

# Paths
db_path = "databases/observations.db"  # <-- Replace with your actual database filename
csv_path = "databases/observations.csv"
txt_path = "databases/observations_rag.txt"

# Connect to the database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Fetch all data
cursor.execute("SELECT id_observation, name, common_name, coordinates, notes FROM observations")
rows = cursor.fetchall()

# Write CSV
with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id_observation', 'name', 'common_name', 'coordinates', 'notes'])
    writer.writerows(rows)

# Write RAG-friendly TXT
with open(txt_path, mode='w', encoding='utf-8') as txtfile:
    for row in rows:
        id_obs, name, common_name, coordinates, notes = row
        txtfile.write(f"Observation ID: {id_obs}\n")
        txtfile.write(f"Scientific Name: {name}\n")
        txtfile.write(f"Common Name: {common_name or 'N/A'}\n")
        txtfile.write(f"Coordinates: {coordinates or 'N/A'}\n")
        txtfile.write(f"Notes: {notes or 'N/A'}\n")
        txtfile.write(f"Date: {today_date}\n")
        txtfile.write("---\n\n")

# Close connection
conn.close()

print("CSV and TXT files have been created.")
