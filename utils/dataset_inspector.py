import os
import pandas as pd

# =====================================================
# DATASET PATH (ROBUST)
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "data", "processed", "final_dataset.csv")


def load_data():
    df = pd.read_csv(DATASET_PATH)

    # Normalize text columns
    df['location'] = df['location'].str.lower().str.strip()
    df['veg_nonveg_type'] = df['veg_nonveg_type'].str.lower().str.strip()

    return df


def extract_all_dishes(row):
    """
    Priority:
    1. dish_liked (comma-separated dishes)
    2. cuisines (fallback)
    """
    if pd.notna(row['dish_liked']) and str(row['dish_liked']).strip():
        return [d.strip().lower() for d in row['dish_liked'].split(',')]
    elif pd.notna(row['cuisines']) and str(row['cuisines']).strip():
        return [c.strip().lower() for c in row['cuisines'].split(',')]
    else:
        return []


def show_location_dataset(df, location):
    loc_df = df[df['location'] == location.lower()]

    if loc_df.empty:
        print("\n❌ Location not found in dataset")
        return

    print(f"\n📍 LOCATION: {location.title()}")
    print(f"🍴 Total Restaurants: {loc_df['name'].nunique()}")

    rows = []

    for _, row in loc_df.iterrows():
        dishes = extract_all_dishes(row)

        for dish in dishes:
            rows.append({
                "Restaurant Name": row['name'],
                "Dish": dish.title(),
                "Veg / Non-Veg": row['veg_nonveg_type']
            })

    if not rows:
        print("\n❌ No dish information available for this location.")
        return

    table = (
        pd.DataFrame(rows)
        .drop_duplicates()
        .sort_values(by=["Restaurant Name", "Dish"])
        .reset_index(drop=True)
    )

    print("\n📊 RESTAURANTS & ALL AVAILABLE DISHES (DATASET VIEW):")
    print(table.to_string(index=False))


if __name__ == "__main__":
    df = load_data()
    user_location = input("Enter location to inspect: ").strip()
    show_location_dataset(df, user_location)
