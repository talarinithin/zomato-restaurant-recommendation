import sqlite3
import os

DB_PATH = "db/zomato.db"

os.makedirs("db", exist_ok=True)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Restaurants table
cursor.execute("""
CREATE TABLE IF NOT EXISTS restaurants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT,
    veg_nonveg_type TEXT,
    dish_liked TEXT,
    cuisines TEXT,
    rate REAL,
    success_rate REAL,
    cancellation_rate REAL,
    approx_cost REAL
)
""")

# Locations table
cursor.execute("""
CREATE TABLE IF NOT EXISTS restaurant_locations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT,
    latitude REAL,
    longitude REAL
)
""")

conn.commit()
conn.close()

print("✅ SQLite database and tables created successfully")