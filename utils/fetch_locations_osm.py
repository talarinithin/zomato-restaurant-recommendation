import os
import time
import pandas as pd
from geopy.geocoders import Nominatim

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "data", "processed", "final_dataset.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "enriched", "restaurant_locations.csv")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


df = pd.read_csv(DATASET_PATH)

# ✅ 5,000 QUALITY-FILTERED RESTAURANTS
restaurants = (
    df[(df['votes'] > 200) & (df['rate'] >= 3.5)]
    [['name', 'location']]
    .drop_duplicates()
    .head(5000)
    .reset_index(drop=True)
)

print(f"Total restaurants selected for geocoding: {len(restaurants)}")

geolocator = Nominatim(user_agent="zomato_project_osm")
results = []

print("\n📍 Fetching real restaurant locations...")
start_time = time.time()

for i, row in restaurants.iterrows():
    print(f"Processing {i+1}/{len(restaurants)} : {row['name']}")

    query = f"{row['name']} {row['location']} Bangalore"
    try:
        loc = geolocator.geocode(query, timeout=10)
        if loc:
            results.append({
                "name": row['name'],
                "location": row['location'],
                "latitude": loc.latitude,
                "longitude": loc.longitude
            })
        else:
            results.append({
                "name": row['name'],
                "location": row['location'],
                "latitude": None,
                "longitude": None
            })
        time.sleep(1)
    except:
        results.append({
            "name": row['name'],
            "location": row['location'],
            "latitude": None,
            "longitude": None
        })
        time.sleep(1)

pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

print("\n✅ Restaurant locations saved successfully")
print(f"⏱ Total time: {(time.time() - start_time)/60:.2f} minutes")
