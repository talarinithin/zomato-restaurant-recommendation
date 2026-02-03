import os
import pandas as pd
import joblib
import numpy as np
from difflib import get_close_matches
from math import radians, sin, cos, asin, sqrt

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "final_dataset.csv")
LOC_PATH = os.path.join(BASE_DIR, "data", "enriched", "restaurant_locations.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")

# =====================================================
# LOAD RESOURCES (ONCE)
# =====================================================
df = pd.read_csv(DATA_PATH)
loc_df = pd.read_csv(LOC_PATH)
model = joblib.load(MODEL_PATH)

# =====================================================
# NORMALIZATION
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
        dishes += df_loc['dish_liked'].dropna().str.split(',').explode().tolist()

    if 'cuisines' in df_loc:
        dishes += df_loc['cuisines'].dropna().str.split(',').explode().tolist()

    dishes = [normalize_dish(d.strip()) for d in dishes if len(d.strip()) > 2]
    return sorted(set(dishes))

def suggest_dishes(user_dish, available_dishes):
    matches = [d for d in available_dishes if user_dish in d]
    if not matches:
        matches = get_close_matches(user_dish, available_dishes, n=5, cutoff=0.6)
    return matches[:5]

def is_nonveg_dish_name(dish):
    nonveg_keywords = [
        "chicken","mutton","fish","egg","biryani","kebab",
        "prawn","seafood","crab","lamb","beef"
    ]
    return any(k in dish for k in dish.lower())

# =====================================================
# MAIN SERVICE FUNCTION
# =====================================================
def recommend_restaurants(location, food_type, user_dish, budget):
    """
    Core ML service function.
    Called from Flask.
    Returns: list of dicts (Top 5 restaurants)
    """

    location = normalize(location)
    food_type = normalize(food_type)
    user_dish = normalize_dish(normalize(user_dish))

    if budget < 100:
        return {"error": "Orders below ₹100 are not supported."}

    # ---------------------------------------------
    # Filter by location & food type
    # ---------------------------------------------
    if food_type == "veg":
        df_loc = df[(df['location'] == location) & (df['veg_nonveg_type'] == "veg")]
    elif food_type == "nonveg":
        df_loc = df[(df['location'] == location) & (df['veg_nonveg_type'].isin(["nonveg", "both"]))]
    else:
        return {"error": "Invalid food type."}

    if df_loc.empty:
        return {"error": "No restaurants found for this location and food type."}

    # ---------------------------------------------
    # Dish suggestion
    # ---------------------------------------------
    available_dishes = extract_dishes(df_loc)
    suggestions = suggest_dishes(user_dish, available_dishes)

    if not suggestions:
        return {"error": "No matching dishes found."}

    selected_dish = suggestions[0]

    # ---------------------------------------------
    # Filter by dish & budget
    # ---------------------------------------------
    dish_df = df_loc[
        df_loc['dish_liked'].str.contains(selected_dish, na=False, regex=False) |
        df_loc['cuisines'].str.contains(selected_dish, na=False, regex=False)
    ].copy()

    dish_df = dish_df[dish_df['approx_cost(for two people)'] <= budget]

    if dish_df.empty:
        return {"error": "No restaurants found within budget."}

    # ---------------------------------------------
    # LOCATION MERGE
    # ---------------------------------------------
    loc_df.columns = loc_df.columns.str.lower().str.strip()
    loc_df = loc_df.rename(columns={'lat':'latitude','lng':'longitude','lon':'longitude'})

    dish_df = dish_df.merge(
        loc_df[['name','location','latitude','longitude']],
        on=['name','location'],
        how='left'
    )

    dish_df['distance_km'] = np.random.uniform(1.0, 5.0, size=len(dish_df))

    # ---------------------------------------------
    # MODEL PREDICTION
    # ---------------------------------------------
    expected_features = model.named_steps['preprocess'].feature_names_in_
    for col in expected_features:
        if col not in dish_df.columns:
            dish_df[col] = 0

    X_pred = dish_df[expected_features]
    dish_df['predicted_rating'] = model.predict(X_pred)

    # ---------------------------------------------
    # FINAL SCORING
    # ---------------------------------------------
    final = dish_df.groupby('name', as_index=False).agg({
        'predicted_rating':'mean',
        'rate':'mean',
        'success_rate':'mean',
        'cancellation_rate':'mean',
        'distance_km':'mean',
        'approx_cost(for two people)':'mean'
    })

    final['norm_distance'] = final['distance_km'] / final['distance_km'].max()
    final['final_score'] = (
        0.45 * final['predicted_rating']
        + 0.30 * final['success_rate']
        - 0.15 * final['cancellation_rate']
        - 0.10 * final['norm_distance']
    )

    top5 = final.sort_values('final_score', ascending=False).head(5)

    # ---------------------------------------------
    # FORMAT OUTPUT (FLASK FRIENDLY)
    # ---------------------------------------------
    results = []
    for r in top5.itertuples():
        results.append({
            "name": r.name.title(),
            "predicted_rating": round(r.predicted_rating,2),
            "actual_rating": round(r.rate,1),
            "distance_km": round(r.distance_km,2),
            "success_rate": int(r.success_rate*100),
            "cancellation_rate": int(r.cancellation_rate*100),
            "budget_for_two": int(r._6),
            "dish": selected_dish.title()
        })

    return results
