import sqlite3
import pandas as pd
import os

DB_PATH = "db/zomato.db"
os.makedirs("db", exist_ok=True)

df = pd.read_csv("data/processed/final_dataset.csv")
loc_df = pd.read_csv("data/enriched/restaurant_locations.csv")

conn = sqlite3.connect(DB_PATH)

# Let pandas CREATE tables automatically
df.to_sql("restaurants", conn, if_exists="replace", index=False)
loc_df.to_sql("restaurant_locations", conn, if_exists="replace", index=False)

conn.close()

print("✅ SQLite database created directly from CSV (SAFE & COMPLETE)")
