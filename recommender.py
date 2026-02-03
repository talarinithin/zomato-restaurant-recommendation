import pandas as pd
import joblib
import numpy as np
from difflib import get_close_matches
from math import radians, sin, cos, asin, sqrt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data/processed/final_dataset.csv")
LOC_PATH = os.path.join(BASE_DIR, "data/enriched/restaurant_locations.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models/best_model.pkl")

df = pd.read_csv(DATA_PATH)
loc_df = pd.read_csv(LOC_PATH)
model = joblib.load(MODEL_PATH)

# normalize
for col in ['name', 'location', 'veg_nonveg_type', 'dish_liked', 'cuisines']:
    df[col] = df[col].astype(str).str.lower().str.strip()

for col in ['name', 'location']:
    loc_df[col] = loc_df[col].astype(str).str.lower().str.strip()


def normalize(text):
    return text.lower().strip()


def normalize_dish(dish):
    return dish.replace("biriyani", "biryani").replace("biriani", "biryani")


def is_nonveg_dish_name(dish):
    nonveg_keywords = [
        "chicken", "mutton", "fish", "egg", "biryani",
        "keema", "kebab", "prawn", "seafood", "beef"
    ]
    return any(k in dish.lower() for k in nonveg_keywords)


def extract_dishes(df_loc):
    dishes = []
    dishes += df_loc['dish_liked'].dropna().str.split(',').explode().tolist()
    dishes += df_loc['cuisines'].dropna().str.split(',').explode().tolist()
    dishes = [normalize_dish(d.strip()) for d in dishes if len(d.strip()) > 2]
    return sorted(set(dishes))


def suggest_dishes(user_dish, available_dishes):
    matches = [d for d in available_dishes if user_dish in d]
    if not matches:
        matches = get_close_matches(user_dish, available_dishes, n=5, cutoff=0.6)
    return matches[:5]


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)*2 + cos(lat1)*cos(lat2)*sin(dlon/2)*2
    return 2 * asin(np.sqrt(a)) * 6371