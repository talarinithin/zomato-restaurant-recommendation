import pandas as pd
import numpy as np
from difflib import get_close_matches
from math import radians, sin, cos, asin, sqrt

def normalize(text):
    return text.lower().strip()

def normalize_dish(dish):
    return dish.replace("biriyani", "biryani").replace("biriani", "biryani")

def is_nonveg_dish_name(dish):
    nonveg_keywords = [
        "chicken","mutton","fish","egg","biryani","keema",
        "kebab","prawn","seafood","crab","lamb","beef"
    ]
    dish = dish.lower()
    return any(k in dish for k in nonveg_keywords)

def extract_dishes(df):
    dishes = []
    for col in ['dish_liked', 'cuisines']:
        dishes += (
            df[col]
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

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * asin(sqrt(a)) * 6371
