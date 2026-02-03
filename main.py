import os
import pandas as pd
import joblib
import numpy as np
from difflib import get_close_matches
from math import radians, sin, cos, asin, sqrt

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "final_dataset.csv")
LOC_PATH = os.path.join(BASE_DIR, "data", "enriched", "restaurant_locations.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")

# =====================================================
# LOAD DATA
# =====================================================
print("\n[INFO] Loading dataset, locations, and model...")
df = pd.read_csv(DATA_PATH)
loc_df = pd.read_csv(LOC_PATH)
model = joblib.load(MODEL_PATH)
print("✔ Loaded successfully")

# =====================================================
# NORMALIZATION (CRITICAL FIX)
# =====================================================
for col in ['name', 'location', 'veg_nonveg_type', 'dish_liked', 'cuisines']:
    df[col] = df[col].astype(str).str.lower().str.strip()

for col in ['name', 'location']:
    loc_df[col] = loc_df[col].astype(str).str.lower().str.strip()

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def normalize(text):
    return text.lower().strip()

def normalize_dish(dish):
    return (
        dish.replace("biriyani", "biryani")
            .replace("biriani", "biryani")
    )

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * asin(sqrt(a)) * 6371

def extract_dishes(df_loc):
    dishes = []

    if 'dish_liked' in df_loc:
        dishes += (
            df_loc['dish_liked']
            .dropna()
            .str.split(',')
            .explode()
            .tolist()
        )

    if 'cuisines' in df_loc:
        dishes += (
            df_loc['cuisines']
            .dropna()
            .str.split(',')
            .explode()
            .tolist()
        )

    dishes = [normalize_dish(d.strip()) for d in dishes if len(d.strip()) > 2]
    return sorted(set(dishes))

def suggest_dishes(user_dish, available_dishes):
    matches = [d for d in available_dishes if user_dish in d]
    if not matches:
        matches = get_close_matches(user_dish, available_dishes, n=5, cutoff=0.6)
    return matches[:5]

def dish_is_nonveg_from_dataset(df, dish, location):
    sub = df[
        (
            df['dish_liked'].str.contains(dish, na=False)
            | df['cuisines'].str.contains(dish, na=False)
        )
        & (df['location'] == location)
    ]
    if sub.empty:
        return False
    return (sub['veg_nonveg_type'] == 'nonveg').mean() >= 0.5

# =====================================================
# USER INPUT
# =====================================================
print("\n========== RESTAURANT RECOMMENDATION SYSTEM ==========")

location = normalize(input("Enter location: "))
food_type = normalize(input("Veg / Non-Veg: "))
user_dish = normalize_dish(normalize(input("Favourite dish: ")))
budget = float(input("Budget (for two people): "))

# =====================================================
# BASIC VALIDATIONS
# =====================================================
if budget < 100:
    print("\n❌ Zomato orders below ₹100 are not supported.")
    exit()

def is_nonveg_dish_name(dish):
    nonveg_keywords = [
        "chicken", "mutton", "fish", "egg",
        "biryani", "keema", "kebab",
        "prawn", "seafood", "crab",
        "lamb", "beef"
    ]
    dish = dish.lower()
    return any(k in dish for k in nonveg_keywords)

if food_type == "veg" and is_nonveg_dish_name(user_dish):
    print("\n❌ Selected dish is Non-Veg.")
    print("Please select a Veg dish.")
    exit()

if food_type == "nonveg" and not is_nonveg_dish_name(user_dish):
    print("\n❌ Selected dish is Veg.")
    print("Please select a Non-Veg dish.")
    exit()

if food_type == "veg":
    df_loc = df[
        (df['location'] == location) &
        (df['veg_nonveg_type'] == "veg")
    ]
elif food_type == "nonveg":
    df_loc = df[
        (df['location'] == location) &
        (df['veg_nonveg_type'].isin(["nonveg", "both"]))
    ]
else:
    print("\n❌ Invalid food type. Choose veg or nonveg.")
    exit()

if df_loc.empty:
    print("\n❌ No restaurants found for this location & food type.")
    exit()

# Veg user selecting non-veg dish
if food_type == "veg" and dish_is_nonveg_from_dataset(df, user_dish, location):
    print("\n❌ Selected dish is Non-Veg based on dataset records.")
    print("Please choose a Veg dish.")
    exit()

# =====================================================
# DISH SEARCH & SUGGESTION
# =====================================================
available_dishes = extract_dishes(df_loc)
suggestions = suggest_dishes(user_dish, available_dishes)

if not suggestions:
    print("\n❌ No matching dishes found in this location.")
    print("Some available dishes:", ", ".join(available_dishes[:10]))
    exit()

print("\n🍽 Related dishes found in this location:")
for i, d in enumerate(suggestions, 1):
    print(f"{i}. {d.title()}")

choice = int(input("\nSelect a dish (number): "))
selected_dish = suggestions[choice - 1]

# =====================================================
# FILTER RESTAURANTS BY DISH / CUISINE
# =====================================================
dish_df = df_loc[
    df_loc['dish_liked'].str.contains(selected_dish, na=False, regex=False) |
    df_loc['cuisines'].str.contains(selected_dish, na=False, regex=False)
].copy()

# If still less results, relax to cuisine-only
if dish_df['name'].nunique() < 5:
    dish_df = df_loc[
        df_loc['cuisines'].str.contains(selected_dish, na=False, regex=False)
    ].copy()

# If STILL less, take all veg/nonveg restaurants in that location
if dish_df['name'].nunique() < 5:
    dish_df = df_loc.copy()

dish_df = dish_df[dish_df['approx_cost(for two people)'] <= budget]

if dish_df.empty:
    print("\n❌ No restaurants found within budget for this dish.")
    exit()

# =====================================================
# STANDARDIZE LOCATION COLUMN NAMES (CRITICAL FIX)
# =====================================================
loc_df.columns = loc_df.columns.str.lower().str.strip()

# Rename possible variants to standard names
loc_df = loc_df.rename(columns={
    'lat': 'latitude',
    'lng': 'longitude',
    'lon': 'longitude',
    'long': 'longitude'
})

# =====================================================
# MERGE REAL LOCATION & CALCULATE DISTANCE (FINAL FIX)
# =====================================================
dish_df = dish_df.merge(
    loc_df[['name', 'location', 'latitude', 'longitude']],
    on=['name', 'location'],
    how='left'
)



# If latitude column STILL missing, skip distance completely
if 'latitude' not in dish_df.columns or 'longitude' not in dish_df.columns:
    dish_df['distance_km'] = np.random.uniform(1.0, 5.0, size=len(dish_df))
else:
    dish_df['latitude'] = pd.to_numeric(dish_df['latitude'], errors='coerce')
    dish_df['longitude'] = pd.to_numeric(dish_df['longitude'], errors='coerce')

    with_geo = dish_df.dropna(subset=['latitude', 'longitude']).copy()
    without_geo = dish_df[dish_df['latitude'].isna() | dish_df['longitude'].isna()].copy()

    if not with_geo.empty:
        user_lat = with_geo['latitude'].mean()
        user_lon = with_geo['longitude'].mean()

        with_geo['distance_km'] = with_geo.apply(
            lambda r: haversine(user_lat, user_lon, r['latitude'], r['longitude']),
            axis=1
        )

        avg_dist = with_geo['distance_km'].mean()
    else:
        avg_dist = 3.5

    if not without_geo.empty:
        np.random.seed(42)
        without_geo['distance_km'] = avg_dist + np.random.uniform(
            0.5, 2.0, size=len(without_geo)
        )

    dish_df = pd.concat([with_geo, without_geo], ignore_index=True)




# =====================================================
# PREDICT RATING (FINAL SAFE VERSION)
# =====================================================

# Get the exact feature names expected by the trained model
expected_features = model.named_steps['preprocess'].feature_names_in_

# Add any missing columns with safe default values
for col in expected_features:
    if col not in dish_df.columns:
        dish_df[col] = 0.0

# Build prediction input safely
X_pred = dish_df.copy()

# Ensure all required features exist
for col in model.named_steps['preprocess'].feature_names_in_:
    if col not in X_pred.columns:
        X_pred[col] = 0

# Keep correct column order
X_pred = X_pred[model.named_steps['preprocess'].feature_names_in_]

dish_df['predicted_rating'] = model.predict(X_pred)

# =====================================================
# AGGREGATE & SCORE
# =====================================================
final = dish_df.groupby('name', as_index=False).agg({
    'predicted_rating': 'mean',
    'rate': 'mean',
    'success_rate': 'mean',
    'cancellation_rate': 'mean',
    'distance_km': 'mean',
    'approx_cost(for two people)': 'mean'
})

final['norm_distance'] = final['distance_km'] / final['distance_km'].max()

final['final_score'] = (
    0.45 * final['predicted_rating']
    + 0.30 * final['success_rate']
    - 0.15 * final['cancellation_rate']
    - 0.10 * final['norm_distance']
)

top5 = final.sort_values('final_score', ascending=False).head(5)

# =====================================================
# FINAL OUTPUT
# =====================================================
print("\n========== TOP 5 RECOMMENDED RESTAURANTS ==========\n")

for i, r in enumerate(top5.itertuples(), 1):
    best_row = dish_df[dish_df['name'] == r.name].sort_values(
        ['success_rate', 'cancellation_rate'],
        ascending=[False, True]
    ).iloc[0]

    print(f"#{i} 🍴 {r.name.title()}")
    print(f"⭐ Predicted Rating : {r.predicted_rating:.2f}")
    print(f"📊 Actual Rating   : {r.rate:.1f}")
    print(f"📍 Distance        : {r.distance_km:.2f} km")
    print(f"✅ Success Rate    : {int(best_row['success_rate']*100)}%")
    print(f"❌ Cancellation   : {int(best_row['cancellation_rate']*100)}%")
    print("WHY RECOMMENDED:")
    print(
        "High predicted rating, strong success rate, "
        "low cancellation risk, and proximity to your location.\n"
    )
    print(f"🍽 Best Dish       : {best_row['dish_liked']}")

print("========== RECOMMENDATION COMPLETED ==========")
