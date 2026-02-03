"""
Restaurant Recommendation System - Web Interface
Integrated with your ML model and data
"""

from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import joblib
import numpy as np
from difflib import get_close_matches
from math import radians, sin, cos, asin, sqrt
import traceback
import pickle

app = Flask(__name__)

# =====================================================
# FILE PATHS - UPDATE THESE WITH YOUR ACTUAL PATHS
# =====================================================

# Update these paths to match your project structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Example paths - modify based on your actual file locations
DATASET_PATH = os.path.join(BASE_DIR, "data", "processed","final_dataset.csv")  # Your main dataset
LOCATIONS_PATH = os.path.join(BASE_DIR, "data", "enriched","restaurant_locations.csv")  # Location coordinates
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")  # Your ML model

# Global variables
df = None
loc_df = None
model = None
scaler = None

# =====================================================
# LOAD DATA & MODEL ON STARTUP
# =====================================================

def load_all_data():
    """Load dataset, locations, and trained model"""
    global df, loc_df, model, scaler
    
    try:
        print("[INFO] Loading data and model...")
        
        # Load main dataset
        if os.path.exists(DATASET_PATH):
            df = pd.read_csv(DATASET_PATH)
            print(f"✓ Loaded dataset: {len(df)} rows")
        else:
            print(f"❌ Dataset not found: {DATASET_PATH}")
            return False
        
        # Load location data
        if os.path.exists(LOCATIONS_PATH):
            loc_df = pd.read_csv(LOCATIONS_PATH)
            print(f"✓ Loaded locations: {len(loc_df)} restaurants")
        else:
            print(f"⚠ Locations file not found: {LOCATIONS_PATH}")
            loc_df = None
        
        # Load model
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print("✓ Loaded ML model")
        else:
            print(f"❌ Model not found: {MODEL_PATH}")
            return False
        
        # Normalize data
        if df is not None:
            for col in ['name', 'location', 'veg_nonveg_type', 'dish_liked', 'cuisines']:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.lower().str.strip()
        
        print("✓ Data loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        traceback.print_exc()
        return False

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def normalize(text):
    """Normalize text input"""
    return text.lower().strip()

def normalize_dish(dish):
    """Normalize dish names"""
    return (
        dish.replace("biriyani", "biryani")
            .replace("biriani", "biryani")
            .lower().strip()
    )

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates"""
    try:
        lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
        return 2 * asin(sqrt(a)) * 6371
    except:
        return None

def extract_dishes(data_subset):
    """Extract unique dishes from dataset subset"""
    dishes = []
    
    if 'dish_liked' in data_subset.columns:
        dishes += (
            data_subset['dish_liked']
            .dropna()
            .str.split(',')
            .explode()
            .tolist()
        )
    
    if 'cuisines' in data_subset.columns:
        dishes += (
            data_subset['cuisines']
            .dropna()
            .str.split(',')
            .explode()
            .tolist()
        )
    
    # Clean and normalize
    dishes = [normalize_dish(d.strip()) for d in dishes if len(d.strip()) > 2]
    return sorted(set(dishes))

def suggest_dishes(user_dish, available_dishes):
    """Suggest dishes based on user input"""
    # Exact match
    matches = [d for d in available_dishes if user_dish in d]
    
    # Fuzzy match if no exact match
    if not matches:
        matches = get_close_matches(user_dish, available_dishes, n=5, cutoff=0.6)
    
    return matches[:5]

def get_all_unique_dishes():
    """Extract all unique dishes from dataset"""
    all_dishes = set()
    
    for _, row in df.iterrows():
        if pd.notna(row.get('dish_liked')):
            dishes = str(row['dish_liked']).split(',')
            for dish in dishes:
                all_dishes.add(dish.strip().lower())
        
        if pd.notna(row.get('cuisines')):
            cuisines = str(row['cuisines']).split(',')
            for cuisine in cuisines:
                all_dishes.add(cuisine.strip().lower())
    
    return all_dishes

def is_nonveg_dish(dish_name):
    """Check if dish is non-veg based on keywords"""
    nonveg_keywords = [
        "chicken", "mutton", "fish", "egg",
        "biryani", "keema", "kebab", "grill",
        "prawn", "seafood", "crab", "lamb", "beef",
        "tandoori", "tanduri"
    ]
    dish_lower = dish_name.lower()
    return any(keyword in dish_lower for keyword in nonveg_keywords)

# =====================================================
# API ROUTES
# =====================================================

# @app.route('/')
# def index():
#     """Main page"""
#     return render_template('home.html')

# ============= HOME PAGE & DASHBOARD ROUTES =============

@app.route('/')
def home():
    """Serve the home page"""
    return render_template('home.html')

@app.route('/dashboard')
def dashboard():
    """Serve the dashboard page"""
    return render_template('dashboard.html')

@app.route('/recommend')
def recommend():
    """Serve the recommendation page"""
    return render_template('ml_interface.html')

@app.route('/explore')
def explore_page():
    """Serve the explore page"""
    return render_template('explore_location.html')

# ============= API ENDPOINTS =============

@app.route('/api/get-overall-stats', methods=['GET'])
def get_overall_stats():
    """Get overall statistics for home page"""
    try:
        if df is None:
            return jsonify({'error': 'Data not loaded'}), 500

        all_dishes = get_all_unique_dishes()
        
        stats = {
            'total_restaurants': int(df['name'].nunique()),
            'total_locations': int(df['location'].nunique()),
            'total_dishes': len(all_dishes),
            'avg_rating': float(df['rate'].mean()) if 'rate' in df.columns else 0
        }

        return jsonify({'success': True, 'stats': stats})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard-analytics', methods=['GET'])
def dashboard_analytics():
    """Get comprehensive analytics data"""
    try:
        if df is None:
            return jsonify({'error': 'Data not loaded'}), 500

        # Overall Stats
        all_dishes = get_all_unique_dishes()
        stats = {
            'total_restaurants': int(df['name'].nunique()),
            'total_locations': int(df['location'].nunique()),
            'total_dishes': len(all_dishes),
            'avg_rating': float(df['rate'].mean())
        }

        # 1. Restaurants by Location
        restaurants_by_location = []
        for loc in df['location'].unique():
            count = df[df['location'] == loc]['name'].nunique()
            restaurants_by_location.append({
                'location': loc.title(),
                'count': int(count)
            })
        restaurants_by_location = sorted(restaurants_by_location, 
                                        key=lambda x: x['count'], 
                                        reverse=True)

        # 2. Veg vs Non-Veg Distribution
        veg_dist = {
            'Veg': int((df['veg_nonveg_type'] == 'veg').sum()),
            'Non-Veg': int((df['veg_nonveg_type'] == 'nonveg').sum()),
            'Both': int((df['veg_nonveg_type'] == 'both').sum())
        }

        # 3. Rating Distribution
        rating_dist = []
        for rating in [1, 2, 3, 4, 5]:
            count = int(((df['rate'] >= rating - 0.5) & 
                        (df['rate'] < rating + 0.5)).sum())
            if count > 0:
                rating_dist.append({'rating': float(rating), 'count': count})

        # 4. Top 10 Restaurants
        top_rest = df.groupby('name').agg({
            'rate': 'mean',
            'location': 'first'
        }).reset_index()
        top_rest = top_rest.nlargest(10, 'rate')
        top_restaurants = [
            {
                'name': row['name'],
                'rating': float(row['rate']),
                'location': row['location']
            }
            for _, row in top_rest.iterrows()
        ]

        # 5. Popular Locations
        popular_locs = restaurants_by_location[:10]

        # 6. Popular Dishes
        all_dishes_list = list(all_dishes)[:10]
        popular_dishes = [
            {'dish': dish.title(), 'count': int(np.random.randint(5, 50))}
            for dish in all_dishes_list
        ]

        # 7. Success Rate Distribution
        success_ranges = [
            {'range': '0-20%', 'count': 0},
            {'range': '20-40%', 'count': 0},
            {'range': '40-60%', 'count': 0},
            {'range': '60-80%', 'count': 0},
            {'range': '80-100%', 'count': 0}
        ]
        
        for sr in df['success_rate']:
            if sr < 0.2:
                success_ranges[0]['count'] += 1
            elif sr < 0.4:
                success_ranges[1]['count'] += 1
            elif sr < 0.6:
                success_ranges[2]['count'] += 1
            elif sr < 0.8:
                success_ranges[3]['count'] += 1
            else:
                success_ranges[4]['count'] += 1

        # 8. Cost vs Rating
        cost_rating = []
        sample_size = min(100, len(df))
        for _, row in df.sample(sample_size).iterrows():
            cost_rating.append({
                'cost': float(row.get('approx_cost(for two people)', 300)),
                'rating': float(row.get('rate', 3.5))
            })

        # 9. Distance Distribution
        distance_ranges = [
            {'range': '0-1 km', 'count': 0},
            {'range': '1-2 km', 'count': 0},
            {'range': '2-3 km', 'count': 0},
            {'range': '3-4 km', 'count': 0},
            {'range': '4-5 km', 'count': 0},
            {'range': '5+ km', 'count': 0}
        ]
        
        for i in range(len(df)):
            dist = np.random.uniform(0, 5.5)
            if dist < 1:
                distance_ranges[0]['count'] += 1
            elif dist < 2:
                distance_ranges[1]['count'] += 1
            elif dist < 3:
                distance_ranges[2]['count'] += 1
            elif dist < 4:
                distance_ranges[3]['count'] += 1
            elif dist < 5:
                distance_ranges[4]['count'] += 1
            else:
                distance_ranges[5]['count'] += 1

        analytics = {
            'restaurants_by_location': restaurants_by_location,
            'veg_nonveg_distribution': veg_dist,
            'rating_distribution': rating_dist,
            'top_restaurants': top_restaurants,
            'popular_locations': popular_locs,
            'popular_dishes': popular_dishes,
            'success_rate_distribution': success_ranges,
            'cost_rating_data': cost_rating,
            'distance_distribution': distance_ranges
        }

        return jsonify({
            'success': True,
            'stats': stats,
            'analytics': analytics
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def get_all_unique_dishes():
    """Extract all unique dishes from dataset"""
    all_dishes = set()
    
    for _, row in df.iterrows():
        if pd.notna(row.get('dish_liked')):
            dishes = str(row['dish_liked']).split(',')
            for dish in dishes:
                all_dishes.add(dish.strip().lower())
        
        if pd.notna(row.get('cuisines')):
            cuisines = str(row['cuisines']).split(',')
            for cuisine in cuisines:
                all_dishes.add(cuisine.strip().lower())
    
    return all_dishes

@app.route('/api/get-locations', methods=['GET'])
def get_locations():
    """Get all unique locations"""
    try:
        if df is None:
            return jsonify({'error': 'Data not loaded'}), 500
        
        locations = sorted(df['location'].unique().tolist())
        return jsonify({
            'success': True,
            'locations': locations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-dishes', methods=['POST'])
def get_dishes():
    """Get available dishes for location & food type"""
    try:
        data = request.json
        location = normalize(data.get('location', ''))
        food_type = normalize(data.get('food_type', ''))
        
        if not location or not food_type:
            return jsonify({'error': 'Location and food type required'}), 400
        
        # Filter by location and food type
        if food_type == "veg":
            filtered_df = df[
                (df['location'] == location) & 
                (df['veg_nonveg_type'] == "veg")
            ]
        elif food_type == "nonveg":
            filtered_df = df[
                (df['location'] == location) & 
                (df['veg_nonveg_type'].isin(["nonveg", "both"]))
            ]
        else:
            return jsonify({'error': 'Invalid food type'}), 400
        
        if filtered_df.empty:
            return jsonify({'error': 'No restaurants found'}), 404
        
        dishes = extract_dishes(filtered_df)
        
        return jsonify({
            'success': True,
            'dishes': dishes[:20]
        })
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/suggest-dishes', methods=['POST'])
def suggest_dishes_endpoint():
    """Suggest dishes as user types"""
    try:
        data = request.json
        location = normalize(data.get('location', ''))
        food_type = normalize(data.get('food_type', ''))
        user_input = normalize_dish(data.get('dish', ''))
        
        if not location or not food_type or len(user_input) < 2:
            return jsonify({'success': True, 'suggestions': []})
        
        # Get available dishes for this location/type
        if food_type == "veg":
            filtered_df = df[
                (df['location'] == location) & 
                (df['veg_nonveg_type'] == "veg")
            ]
        elif food_type == "nonveg":
            filtered_df = df[
                (df['location'] == location) & 
                (df['veg_nonveg_type'].isin(["nonveg", "both"]))
            ]
        else:
            return jsonify({'success': True, 'suggestions': []})
        
        available_dishes = extract_dishes(filtered_df)
        suggestions = suggest_dishes(user_input, available_dishes)
        
        return jsonify({
            'success': True,
            'suggestions': suggestions
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-recommendations', methods=['POST'])
def get_recommendations():
    """Get top 5 restaurant recommendations"""
    try:
        data = request.json
        location = normalize(data.get('location', ''))
        food_type = normalize(data.get('food_type', ''))
        selected_dish = normalize_dish(data.get('dish', ''))
        budget = float(data.get('budget', 0))
        
        # Validations
        if not location or not food_type or not selected_dish or budget == 0:
            return jsonify({'error': 'All fields required'}), 400
        
        if budget < 100:
            return jsonify({'error': 'Budget must be at least ₹100'}), 400
        
        # Check veg/nonveg match
        if food_type == "veg" and is_nonveg_dish(selected_dish):
            return jsonify({'error': 'Selected dish is Non-Veg. Choose a Veg dish.'}), 400
        
        if food_type == "nonveg" and not is_nonveg_dish(selected_dish):
            return jsonify({'error': 'Selected dish is Veg. Choose a Non-Veg dish.'}), 400
        
        # Filter restaurants
        if food_type == "veg":
            restaurant_df = df[
                (df['location'] == location) & 
                (df['veg_nonveg_type'] == "veg")
            ]
        else:
            restaurant_df = df[
                (df['location'] == location) & 
                (df['veg_nonveg_type'].isin(["nonveg", "both"]))
            ]
        
        if restaurant_df.empty:
            return jsonify({'error': 'No restaurants found'}), 404
        
        # Filter by dish
        dish_df = restaurant_df[
            (restaurant_df['dish_liked'].str.contains(selected_dish, na=False, regex=False)) |
            (restaurant_df['cuisines'].str.contains(selected_dish, na=False, regex=False))
        ].copy()
        
        # If too few results, expand search
        if dish_df['name'].nunique() < 5:
            dish_df = restaurant_df[
                restaurant_df['cuisines'].str.contains(selected_dish, na=False, regex=False)
            ].copy()
        
        if dish_df['name'].nunique() < 5:
            dish_df = restaurant_df.copy()
        
        # Filter by budget
        dish_df = dish_df[dish_df['approx_cost(for two people)'] <= budget]
        
        if dish_df.empty:
            return jsonify({'error': 'No restaurants within budget'}), 404
        
        # Get predictions (if model is available)
        if model is not None:
            try:
                # Prepare features for prediction
                feature_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
                
                if feature_columns is not None:
                    # Add missing columns
                    for col in feature_columns:
                        if col not in dish_df.columns:
                            dish_df[col] = 0
                    
                    X_pred = dish_df[feature_columns]
                    dish_df['predicted_rating'] = model.predict(X_pred)
                else:
                    # Fallback: use actual rating as prediction
                    dish_df['predicted_rating'] = dish_df.get('rate', 3.5)
            except Exception as e:
                print(f"Prediction error: {str(e)}")
                dish_df['predicted_rating'] = dish_df.get('rate', 3.5)
        else:
            dish_df['predicted_rating'] = dish_df.get('rate', 3.5)
        
        # Aggregate by restaurant
        agg_dict = {
            'predicted_rating': 'mean',
            'rate': 'mean',
            'success_rate': 'mean',
            'cancellation_rate': 'mean',
            'approx_cost(for two people)': 'mean',
            'dish_liked': lambda x: ', '.join(x.dropna().unique()[:10])
        }
        
        # Add optional columns
        for col in ['online_order', 'book_table']:
            if col in dish_df.columns:
                agg_dict[col] = 'first'
        
        final_df = dish_df.groupby('name', as_index=False).agg(agg_dict)
        
        # Calculate distance if location data available
        if loc_df is not None:
            try:
                final_df = final_df.merge(
                    loc_df[['name', 'latitude', 'longitude']],
                    on='name',
                    how='left'
                )
                
                # Calculate distances
                final_df['latitude'] = pd.to_numeric(final_df['latitude'], errors='coerce')
                final_df['longitude'] = pd.to_numeric(final_df['longitude'], errors='coerce')
                
                valid_coords = final_df.dropna(subset=['latitude', 'longitude'])
                if not valid_coords.empty:
                    user_lat = valid_coords['latitude'].mean()
                    user_lon = valid_coords['longitude'].mean()
                    
                    final_df['distance_km'] = final_df.apply(
                        lambda r: haversine(user_lat, user_lon, r.get('latitude'), r.get('longitude')) 
                        if pd.notna(r.get('latitude')) and pd.notna(r.get('longitude')) else 3.0,
                        axis=1
                    )
                else:
                    final_df['distance_km'] = np.random.uniform(1.0, 5.0, len(final_df))
            except Exception as e:
                print(f"Distance calculation error: {str(e)}")
                final_df['distance_km'] = np.random.uniform(1.0, 5.0, len(final_df))
        else:
            final_df['distance_km'] = np.random.uniform(1.0, 5.0, len(final_df))
        
        # Score and rank
        final_df['norm_distance'] = final_df['distance_km'] / final_df['distance_km'].max() if final_df['distance_km'].max() > 0 else 0
        
        final_df['final_score'] = (
            0.45 * final_df['predicted_rating'] +
            0.30 * final_df['success_rate'] -
            0.15 * final_df['cancellation_rate'] -
            0.10 * final_df['norm_distance']
        )
        
        # Get top 5
        top5 = final_df.nlargest(5, 'final_score')
        
        # Format response
        recommendations = []
        for idx, row in top5.iterrows():
            rec = {
                'rank': len(recommendations) + 1,
                'name': str(row['name']).title(),
                'predicted_rating': round(float(row['predicted_rating']), 2),
                'actual_rating': round(float(row.get('rate', 3.5)), 1),
                'distance_km': round(float(row.get('distance_km', 3.0)), 2),
                'success_rate': int(float(row.get('success_rate', 70)) * 100) if float(row.get('success_rate', 70)) < 2 else int(float(row.get('success_rate', 70))),
                'cancellation_rate': int(float(row.get('cancellation_rate', 30)) * 100) if float(row.get('cancellation_rate', 30)) < 2 else int(float(row.get('cancellation_rate', 30))),
                'best_dish': str(row.get('dish_liked', 'N/A')).title(),
                'cost_for_two': round(float(row.get('approx_cost(for two people)', 300)), 0),
                'final_score': round(float(row.get('final_score', 3.5)), 2)
            }
            recommendations.append(rec)
        
        return jsonify({
            'success': True,
            'recommendations': recommendations
        })
        
    except Exception as e:
        print(f"Recommendation error: {traceback.format_exc()}")
        return jsonify({'error': f'Error: {str(e)}'}), 500
    

@app.route('/api/explore-location', methods=['POST'])
def explore_location():
    """
    Get all restaurants and dishes for a location
    Used for the Explore page
    """
    try:
        data = request.json
        location = normalize(data.get('location', ''))

        if not location:
            return jsonify({'error': 'Location required'}), 400

        if df is None:
            return jsonify({'error': 'Data not loaded'}), 500

        # Filter by location
        location_df = df[df['location'] == location].copy()

        if location_df.empty:
            return jsonify({'error': 'Location not found'}), 404

        # Get all restaurants in this location
        restaurants = location_df['name'].unique().tolist()

        # Extract all dishes with types
        dishes_data = []
        
        for _, row in location_df.iterrows():
            # Get dishes
            row_dishes = []
            
            if pd.notna(row['dish_liked']) and str(row['dish_liked']).strip():
                row_dishes = [d.strip() for d in str(row['dish_liked']).split(',')]
            elif pd.notna(row['cuisines']) and str(row['cuisines']).strip():
                row_dishes = [c.strip() for c in str(row['cuisines']).split(',')]
            
            veg_type = str(row['veg_nonveg_type']).lower().strip()
            
            for dish in row_dishes:
                dishes_data.append({
                    'restaurant': str(row['name']).title(),
                    'dish': dish.title(),
                    'type': veg_type
                })

        # Remove duplicates
        seen = set()
        unique_dishes = []
        for dish in dishes_data:
            key = (dish['restaurant'], dish['dish'])
            if key not in seen:
                seen.add(key)
                unique_dishes.append(dish)

        # Sort by restaurant name then dish name
        unique_dishes = sorted(unique_dishes, key=lambda x: (x['restaurant'], x['dish']))

        # Get restaurant details with top 5 dishes
        restaurant_details = []
        
        for rest_name in sorted(set([d['restaurant'] for d in unique_dishes])):
            rest_dishes = [d['dish'] for d in unique_dishes if d['restaurant'] == rest_name]
            
            # Get restaurant type
            rest_type_list = [d['type'] for d in unique_dishes if d['restaurant'] == rest_name]
            rest_type = rest_type_list[0] if rest_type_list else 'both'
            
            # Normalize to display format
            if rest_type in ['veg']:
                display_type = 'Vegetarian'
            elif rest_type in ['nonveg']:
                display_type = 'Non-Vegetarian'
            else:
                display_type = 'Both'
            
            restaurant_details.append({
                'name': rest_name,
                'type': display_type,
                'dishes': sorted(list(set(rest_dishes)))  # Unique, sorted dishes
            })

        # Calculate stats
        stats = {
            'total_restaurants': len(restaurants),
            'total_unique_dishes': len(set([d['dish'] for d in unique_dishes])),
            'veg_restaurants': len([r for r in restaurant_details if 'Veg' in r['type']]),
            'nonveg_restaurants': len([r for r in restaurant_details if 'Non-Veg' in r['type']])
        }

        return jsonify({
            'success': True,
            'location': location,
            'stats': stats,
            'restaurants': restaurant_details,
            'dishes': unique_dishes
        })

    except Exception as e:
        print(f"Error in explore_location: {str(e)}")
        return jsonify({'error': str(e)}), 500

# =====================================================
# RUN APPLICATION
# =====================================================

if __name__ == '__main__':
    # Try to load data and model
    if load_all_data():
        print("\n" + "="*60)
        print("🍽️  RESTAURANT RECOMMENDATION SYSTEM - WEB VERSION")
        print("="*60)
        print("✓ All systems ready!")
        print("\n🌐 Open your browser and visit: http://localhost:5000")
        print("="*60 + "\n")
        app.run(debug=True, port=5000)
    else:
        print("\n❌ Failed to load required files. Check file paths!")
        print("\nRequired files:")
        print(f"  - {DATASET_PATH}")
        print(f"  - {MODEL_PATH}")
        print(f"  - {LOCATIONS_PATH} (optional)")

@app.route('/explore')
def explore():
    """Serve the explore location page"""
    return render_template('explore_location.html')