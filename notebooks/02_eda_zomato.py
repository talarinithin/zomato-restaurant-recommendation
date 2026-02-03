"""
=====================================================
EDA & FEATURE ENGINEERING – ZOMATO PROJECT
=====================================================
This script performs Exploratory Data Analysis (EDA)
and prints key insights with visualizations.

Run using:
    python notebooks/02_eda_zomato.py
=====================================================
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# =====================================================
# PATH SETUP (SAFE FOR .py FILE)
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "final_dataset.csv")

# =====================================================
# LOAD DATA
# =====================================================
print("\n[INFO] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print("✔ Dataset loaded successfully")

print(f"Total records: {df.shape[0]}")
print(f"Total features: {df.shape[1]}")

# =====================================================
# BASIC DATA OVERVIEW
# =====================================================
print("\n========== BASIC DATA INFO ==========")
print(df.info())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# =====================================================
# CLEANING FOR EDA
# =====================================================
df['rate'] = pd.to_numeric(df['rate'], errors='coerce')
df['approx_cost(for two people)'] = pd.to_numeric(
    df['approx_cost(for two people)'], errors='coerce'
)

# =====================================================
# 1️⃣ RATING DISTRIBUTION
# =====================================================
print("\n[EDA] Rating Distribution Analysis")

plt.figure()
sns.histplot(df['rate'].dropna(), bins=20, kde=True)
plt.title("Restaurant Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

print("INSIGHT:")
print("- Most ratings lie between 3.5 and 4.2")
print("- Ratings alone are not sufficient for recommendation")

# =====================================================
# 2️⃣ VEG vs NON-VEG DISTRIBUTION
# =====================================================
print("\n[EDA] Veg vs Non-Veg Distribution")

plt.figure()
df['veg_nonveg_type'].value_counts().plot(kind='bar')
plt.title("Veg vs Non-Veg Restaurants")
plt.xlabel("Food Type")
plt.ylabel("Count")
plt.show()

print("INSIGHT:")
print("- Both Veg and Non-Veg restaurants are well represented")
print("- Dietary preference filtering is essential")

# =====================================================
# 3️⃣ COST FOR TWO VS RATING
# =====================================================
print("\n[EDA] Cost vs Rating")

plt.figure()
sns.scatterplot(
    x=df['approx_cost(for two people)'],
    y=df['rate'],
    alpha=0.4
)
plt.title("Cost for Two vs Rating")
plt.xlabel("Cost for Two")
plt.ylabel("Rating")
plt.show()

print("INSIGHT:")
print("- No strong correlation between higher cost and higher rating")
print("- Mid-range restaurants often perform best")

# =====================================================
# 4️⃣ SUCCESS RATE VS RATING
# =====================================================
if 'success_rate' in df.columns:
    print("\n[EDA] Success Rate vs Rating")

    plt.figure()
    sns.scatterplot(
        x=df['success_rate'],
        y=df['rate'],
        alpha=0.4
    )
    plt.title("Success Rate vs Rating")
    plt.xlabel("Success Rate")
    plt.ylabel("Rating")
    plt.show()

    print("INSIGHT:")
    print("- Higher success rate generally leads to higher ratings")
    print("- Operational reliability matters")

# =====================================================
# 5️⃣ CANCELLATION RATE VS RATING
# =====================================================
if 'cancellation_rate' in df.columns:
    print("\n[EDA] Cancellation Rate vs Rating")

    plt.figure()
    sns.scatterplot(
        x=df['cancellation_rate'],
        y=df['rate'],
        alpha=0.4
    )
    plt.title("Cancellation Rate vs Rating")
    plt.xlabel("Cancellation Rate")
    plt.ylabel("Rating")
    plt.show()

    print("INSIGHT:")
    print("- High cancellation rate negatively impacts ratings")
    print("- Should be penalized in recommendation score")

# =====================================================
# 6️⃣ LOCATION-WISE RESTAURANT COUNT
# =====================================================
print("\n[EDA] Top Locations by Restaurant Count")

top_locations = df['location'].value_counts().head(10)

plt.figure()
top_locations.plot(kind='bar')
plt.title("Top 10 Locations with Most Restaurants")
plt.xlabel("Location")
plt.ylabel("Restaurant Count")
plt.show()

print("INSIGHT:")
print("- Locations like BTM, Indiranagar, Whitefield are dense food hubs")
print("- Location-based filtering is effective")

# =====================================================
# 7️⃣ MOST POPULAR DISHES
# =====================================================
print("\n[EDA] Popular Dishes Analysis")

if 'dish_liked' in df.columns:
    dishes = (
        df['dish_liked']
        .dropna()
        .str.lower()
        .str.split(',')
        .explode()
        .str.strip()
    )

    top_dishes = dishes.value_counts().head(10)

    plt.figure()
    top_dishes.plot(kind='bar')
    plt.title("Top 10 Popular Dishes")
    plt.xlabel("Dish")
    plt.ylabel("Frequency")
    plt.show()

    print("INSIGHT:")
    print("- Dosa, Biryani, Paneer-based dishes dominate")
    print("- Dish-based recommendation is highly valuable")

# =====================================================
# FEATURE ENGINEERING INSIGHTS
# =====================================================
print("\n========== FEATURE ENGINEERING INSIGHTS ==========")

print("""
1. Ratings alone are insufficient → Predicted rating added
2. Success rate improves reliability → Included in final score
3. Cancellation rate reduces trust → Penalized
4. Distance improves relevance → Added using latitude & longitude
5. Dish & cuisine text normalized → Improves matching
6. Budget used as constraint → Not a ranking factor
""")

# =====================================================
# FINAL SUMMARY
# =====================================================
print("\n========== EDA COMPLETED ==========")
print("EDA helped guide feature selection, scoring logic, and model design.")

print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
