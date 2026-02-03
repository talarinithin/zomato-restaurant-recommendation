import sqlite3

conn = sqlite3.connect("db/zomato.db")
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM restaurants")
print("Restaurants count:", cursor.fetchone()[0])

cursor.execute("SELECT COUNT(*) FROM restaurant_locations")
print("Locations count:", cursor.fetchone()[0])

cursor.execute("SELECT name, location FROM restaurants LIMIT 5")
print(cursor.fetchall())

conn.close()