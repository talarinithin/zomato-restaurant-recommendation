# 🍽️ Zomato Restaurant Recommendation System

> **AI-Powered Restaurant Discovery for Bangalore**
> 
> Get personalized restaurant recommendations based on your location, cuisine preferences, budget, and favorite dishes using machine learning.

---

## 📊 Dataset Source

This project uses the **Zomato Bangalore Restaurants Dataset** from Kaggle:

🔗 **[Zomato Bangalore Restaurants Dataset](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants)**

**Dataset Overview:**
- 📍 **City**: Bangalore, India
- 🏪 **Total Restaurants**: 500+ restaurants
- 📊 **Total Records**: 51,696 restaurant entries
- 📍 **Locations**: 30+ popular areas (BTM, Marathalli, Indiranagar, Whitefield, etc.)
- 🍽️ **Cuisines**: 100+ different cuisines and dishes
- ⭐ **Ratings**: 1-5 star scale with success and cancellation metrics
- 📈 **Features**: 19 attributes per restaurant

---

## ✨ Key Features

### 🎯 Smart Recommendations
Get top 5 personalized restaurant recommendations based on:
- 📍 **Location** - Choose from 30+ Bangalore areas (BTM, Marathalli, Indiranagar, etc.)
- 🥗 **Food Type** - Vegetarian or Non-Vegetarian
- 🍲 **Favorite Dish** - Biryani, Pizza, Dosa, Curry, and more
- 💰 **Budget** - Set your budget for two people (₹100 - ₹5000+)

### 🔍 Intelligent Dish Matching
- ✅ Fuzzy matching for dish names
- ✅ Automatic spelling correction (e.g., "biriyani" → "biryani")
- ✅ Dish availability validation
- ✅ Cuisine-based suggestions

### 📈 Advanced Scoring Algorithm
Recommendations ranked by:
- ⭐ **Predicted Rating** (45% weight) - ML model prediction
- ✅ **Success Rate** (30% weight) - Order fulfillment rate
- ❌ **Cancellation Rate** (-15% weight) - Lower is better
- 📍 **Distance** (-10% weight) - Proximity to location

### 📊 Interactive Web Dashboard
- 🏠 **Home Page** - Quick statistics and overview
- 📈 **Analytics Dashboard** - Comprehensive insights
- 🎯 **Recommendation Engine** - Interactive recommendation form
- 🗺️ **Location Explorer** - Browse all restaurants and dishes in any area

---

## 🗺️ Supported Bangalore Locations

### Popular Areas Covered:
- **BTM** (Banashankari)
- **Marathalli**
- **Indiranagar**
- **Whitefield**
- **Koramangala**
- **MG Road**
- **Jayanagar**
- **Bellandur**
- **Sarjapur Road**
- **Bannerghatta Road**
- **Vijayanagar**
- **Malleswaram**
- **Ulsoor**
- **Frazer Town**
- **Shivajinagar**
- And 15+ more locations...

---

## 📁 Project Structure

```
ZomatoProj/
│
├── 📊 data/
│   ├── raw/
│   │   └── zomato.csv                    # Original Kaggle dataset
│   ├── processed/
│   │   └── final_dataset.csv             # Cleaned & processed data (51,696 rows)
│   └── enriched/
│       └── restaurant_locations.csv      # Coordinates & locations
│
├── 🤖 models/
│   └── best_model.pkl                    # Trained Random Forest model
│
├── 📓 notebooks/
│   ├── 01_data_cleaning.ipynb            # Data cleaning pipeline
│   └── 02_eda_zomato.py                  # Exploratory analysis
│
├── 🔧 scripts/
│   ├── create_db.py                      # Database setup
│   ├── migrate_csv_to_sqlite.py          # CSV to SQLite migration
│   └── check_db.py                       # Database validation
│
├── 🛠️ services/
│   ├── recommender_service.py            # Recommendation logic
│   └── recommender.py                    # Recommender class
│
├─��� 🔨 utils/
│   ├── dataset_inspector.py              # Data inspection
│   ├── dish_suggester.py                 # Dish suggestions
│   ├── distance_utils.py                 # Distance calculations
│   ├── fetch_locations_osm.py            # Location fetching
│   └── synthetic_data.py                 # Data generation
│
├── 🎨 templates/
│   ├── home.html                         # Home page
│   ├── dashboard.html                    # Analytics dashboard
│   ├── ml_interface.html                 # Recommendation form
│   └── explore_location.html             # Location explorer
│
├── 💾 db/
│   └── zomato.db                         # SQLite database
│
├── 🚀 main.py                            # CLI entry point
├── 🌐 ml_app.py                          # Flask web app
├── 🤖 model_building.py                  # Model training & comparison
├── ⚙️ hyperparameter_tuning.py           # Hyperparameter optimization
├── 🧪 test_model.py                      # Model testing
├── 📋 requirements.txt                   # Dependencies
├── 🔐 .gitignore                         # Git ignore rules
└── 📖 README.md                          # This file
```

---

## 🚀 Complete Installation Guide

### Prerequisites
- 🐍 **Python 3.8 or higher** - [Download Python](https://www.python.org/downloads/)
- 📦 **pip** - Comes with Python
- 🔧 **Git** (optional, for cloning) - [Download Git](https://git-scm.com/)
- 💻 **Terminal/Command Prompt** - Built-in on all systems

---

## 📥 Step-by-Step Installation

### **Step 1️⃣: Clone or Download the Project**

#### Option A: Clone using Git (Recommended)
```bash
# Open terminal/command prompt and run:
git clone https://github.com/yourusername/ZomatoProj.git
cd ZomatoProj
```

#### Option B: Download as ZIP
1. Go to the GitHub repository
2. Click **"Code"** → **"Download ZIP"**
3. Extract the ZIP file
4. Open terminal/command prompt in the extracted folder

```bash
# Navigate to project folder
cd path/to/ZomatoProj
```

---

### **Step 2️⃣: Verify Python Installation**

Check if Python is installed correctly:

```bash
# Check Python version
python --version

# Should show: Python 3.8.x or higher
```

If Python is not recognized, you may need to:
- Reinstall Python and check "Add Python to PATH"
- Use `python3` instead of `python`

---

### **Step 3️⃣: Create Virtual Environment**

A virtual environment isolates project dependencies from your system Python.

#### Windows:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# You should see (venv) in your terminal
```

#### macOS/Linux:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# You should see (venv) in your terminal
```

**What is a Virtual Environment?**
- 🔒 Isolates project dependencies
- 🛡️ Prevents conflicts with other projects
- �� Makes project portable
- ✅ Best practice for Python development

---

### **Step 4️⃣: Upgrade pip (Optional but Recommended)**

```bash
# Upgrade pip to latest version
pip install --upgrade pip

# Verify pip version
pip --version
```

---

### **Step 5️⃣: Install Project Dependencies**

Install all required packages from `requirements.txt`:

```bash
# Install all dependencies
pip install -r requirements.txt

# This will install:
# - pandas (data manipulation)
# - numpy (numerical computing)
# - scikit-learn (machine learning)
# - Flask (web framework)
# - And 20+ other packages
```

**Installation Time**: ~2-5 minutes depending on internet speed

**What's being installed?**
```
✓ Data Science: pandas, numpy, scikit-learn
✓ Web Framework: Flask, Jinja2
✓ Visualization: matplotlib, seaborn, folium
✓ Geospatial: geopy, geographiclib
✓ Database: mysql-connector-python
✓ Utilities: requests, Pillow, joblib
```

---

### **Step 6️⃣: Verify Installation**

Verify all dependencies are installed correctly:

```bash
# Test all major imports
python -c "import pandas, numpy, sklearn, flask; print('✓ All dependencies installed successfully!')"

# Should output: ✓ All dependencies installed successfully!
```

If you see errors, run:
```bash
pip install -r requirements.txt --upgrade
```

---

### **Step 7️⃣: Verify Project Files**

Check that all required files exist:

```bash
# List project structure
dir  # Windows
ls   # macOS/Linux

# You should see:
# ✓ data/
# ✓ models/
# ✓ templates/
# ✓ main.py
# ✓ ml_app.py
# ✓ requirements.txt
```

---

## 🎯 Running the Application

### **Option 1️⃣: Web Interface (Recommended for New Users)**

Start the Flask web application:

```bash
# Make sure virtual environment is activated
# (You should see (venv) in your terminal)

# Run the web app
python ml_app.py

# Output should show:
# * Running on http://127.0.0.1:5000
# * Press CTRL+C to quit
```

**Then open your browser:**
```
🌐 http://localhost:5000
```

**Features available:**
- 🏠 Home page with statistics
- 📊 Analytics dashboard
- 🎯 Recommendation engine
- 🗺️ Location explorer

---

### **Option 2️⃣: Command-Line Interface (CLI)**

For quick recommendations without opening a browser:

```bash
# Make sure virtual environment is activated
python main.py

# Follow the interactive prompts:
# Enter location: BTM
# Veg / Non-Veg: nonveg
# Favourite dish: biryani
# Budget (for two people): 500

# Get top 5 recommendations!
```

---

### **Option 3️⃣: Retrain the Model**

If you want to retrain the model with new data:

```bash
# Train and compare all models
python model_building.py

# Output shows:
# - Model comparison table
# - Best model selected (Random Forest)
# - Model saved to models/best_model.pkl

# Optimize hyperparameters
python hyperparameter_tuning.py

# Output shows:
# - Best parameters found
# - Cross-validated R² score
# - Optimized model saved
```

---

## 📋 Complete Setup Checklist

Use this checklist to verify your setup:

- [ ] Python 3.8+ installed
- [ ] Virtual environment created (`venv` folder exists)
- [ ] Virtual environment activated (see `(venv)` in terminal)
- [ ] `pip install -r requirements.txt` completed successfully
- [ ] All dependencies verified with test import
- [ ] Project files present (data/, models/, templates/, etc.)
- [ ] Can run `python ml_app.py` without errors
- [ ] Can access http://localhost:5000 in browser
- [ ] Can run `python main.py` for CLI interface

---

## 🔄 Deactivating Virtual Environment

When you're done working on the project:

```bash
# Deactivate virtual environment
deactivate

# (venv) should disappear from terminal
```

---

## 🔁 Reactivating Virtual Environment

Next time you work on the project:

```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Then run your application
python ml_app.py
```

---

## 🐛 Common Installation Issues

### Issue 1: "Python is not recognized"
**Solution:**
- Reinstall Python and check "Add Python to PATH"
- Or use `python3` instead of `python`

### Issue 2: "pip is not recognized"
**Solution:**
```bash
# Use Python module to run pip
python -m pip install -r requirements.txt
```

### Issue 3: "Permission denied" (macOS/Linux)
**Solution:**
```bash
# Use sudo (not recommended for virtual env)
# Better: Check virtual environment is activated
source venv/bin/activate
pip install -r requirements.txt
```

### Issue 4: "Module not found" errors
**Solution:**
```bash
# Reinstall all dependencies
pip install -r requirements.txt --upgrade --force-reinstall
```

### Issue 5: Port 5000 already in use
**Solution:**
```bash
# Edit ml_app.py and change port
# Find: app.run(debug=True, port=5000)
# Change to: app.run(debug=True, port=5001)
```

---

## 📚 Project Structure After Installation

```
ZomatoProj/
├── venv/                          # Virtual environment (created)
├── data/
│   ├── raw/
│   │   └── zomato.csv
│   ├── processed/
│   │   └── final_dataset.csv
│   └── enriched/
│       └── restaurant_locations.csv
├── models/
│   └── best_model.pkl
├── templates/
│   ├── home.html
│   ├── dashboard.html
│   ├── ml_interface.html
│   └── explore_location.html
├── main.py
├── ml_app.py
├── model_building.py
├── hyperparameter_tuning.py
├── requirements.txt
└── README.md
```

---

## ✅ Verification Commands

Run these commands to verify everything is working:

```bash
# 1. Check Python version
python --version

# 2. Check virtual environment is active
# (You should see (venv) in terminal)

# 3. Check pip version
pip --version

# 4. List installed packages
pip list

# 5. Test imports
python -c "import pandas, numpy, sklearn, flask; print('✓ All good!')"

# 6. Check project files
ls data/processed/final_dataset.csv
ls models/best_model.pkl

# 7. Run web app
python ml_app.py
# Then visit http://localhost:5000
```

---

## 🎓 Next Steps After Installation

1. **Explore the Web Interface**
   - Open http://localhost:5000
   - Try the Location Explorer
   - Get recommendations

2. **Try the CLI**
   - Run `python main.py`
   - Enter your preferences
   - Get instant recommendations

3. **Explore the Code**
   - Read `main.py` for CLI logic
   - Read `ml_app.py` for web app logic
   - Check `model_building.py` for ML pipeline

4. **Customize the Project**
   - Modify recommendation weights
   - Add new features
   - Retrain the model

---

## 📞 Need Help?

If you encounter issues:

1. **Check the Troubleshooting section** above
2. **Verify all files exist** in the project directory
3. **Ensure virtual environment is activated**
4. **Check Python version** is 3.8 or higher
5. **Reinstall dependencies** if needed

```bash
# Nuclear option - reinstall everything
deactivate
rm -rf venv  # or rmdir venv on Windows
python -m venv venv
# Activate venv
pip install -r requirements.txt
```

---

## 💻 How to Use

### Option 1️⃣: Command-Line Interface (CLI)

Perfect for quick recommendations without opening a browser.

```bash
python main.py
```

**Interactive Steps:**

```
========== RESTAURANT RECOMMENDATION SYSTEM ==========

Enter location: BTM
Veg / Non-Veg: nonveg
Favourite dish: biryani
Budget (for two people): 500

🍽 Related dishes found in this location:
1. Biryani
2. Chicken Biryani
3. Mutton Biryani

Select a dish (number): 1

========== TOP 5 RECOMMENDED RESTAURANTS ==========

#1 🍴 Restaurant Name
⭐ Predicted Rating : 4.25
📊 Actual Rating   : 4.2
📍 Distance        : 2.45 km
✅ Success Rate    : 85%
❌ Cancellation   : 12%
🍽 Best Dish       : Biryani

#2 🍴 Another Restaurant
⭐ Predicted Rating : 4.15
📊 Actual Rating   : 4.1
📍 Distance        : 1.80 km
✅ Success Rate    : 82%
❌ Cancellation   : 15%
🍽 Best Dish       : Chicken Biryani

... and 3 more recommendations
```

---

### Option 2️⃣: Web Interface (Recommended)

Full-featured web application with dashboard and explorer.

```bash
python ml_app.py
```

Then open your browser:
```
🌐 http://localhost:5000
```

#### 🏠 Home Page
- Quick statistics overview
- Total restaurants, locations, dishes
- Average ratings
- Quick access to all features

#### 📊 Dashboard (`/dashboard`)
Comprehensive analytics including:
- 📍 Restaurants by location
- 🥗 Veg vs Non-Veg distribution
- ⭐ Rating distribution
- 🏆 Top 10 restaurants
- 🍽️ Popular dishes
- ✅ Success rate analysis
- 💰 Cost vs Rating correlation
- 📏 Distance distribution

#### 🎯 Recommendation Engine (`/recommend`)
Interactive form to get personalized recommendations:
1. Select location (BTM, Marathalli, etc.)
2. Choose food type (Veg/Non-Veg)
3. Enter favorite dish
4. Set budget
5. Get top 5 recommendations with detailed insights

#### 🗺️ Location Explorer (`/explore`)
**Perfect for new users!** Browse and discover:
- 🏪 All restaurants in a location
- 🍲 All available dishes
- 🥗 Veg/Non-Veg classification
- 📊 Location statistics
- 🔍 Search and filter options

**Example: Explore BTM**
```
Location: BTM
Total Restaurants: 45
Total Unique Dishes: 120
Vegetarian Restaurants: 18
Non-Vegetarian Restaurants: 27

Restaurants:
├── Restaurant A (Non-Veg)
│   ├── Biryani
│   ├── Chicken Curry
│   └── Tandoori Chicken
├── Restaurant B (Veg)
│   ├── Dosa
│   ├── Idli
│   └── Sambar
└── ... and 43 more
```

---

## 🤖 Machine Learning Model

### Model Selection: Random Forest ⭐

After comparing 5 different algorithms on **51,696 restaurant records**, **Random Forest** was selected for its superior performance and reliability:

#### Model Comparison Results

| Algorithm | MAE | RMSE | R² Score |
|-----------|-----|------|----------|
| Linear Regression | 0.176 | 0.230 | 0.665 |
| Decision Tree | 0.112 | 0.223 | 0.684 |
| **Random Forest** ⭐ | **0.122** | **0.191** | **0.769** |
| Gradient Boosting | 0.152 | 0.216 | 0.704 |
| XGBoost | 0.140 | 0.203 | 0.738 |

### Why Random Forest?
✅ **Best R² Score (0.769)** - Explains 76.9% of variance  
✅ **Lowest RMSE (0.191)** - Minimal prediction error  
✅ **Handles non-linear relationships** effectively  
✅ **Reduces overfitting** through ensemble averaging  
✅ **Robust to outliers and noise** in data  
✅ **Fast training and inference** time  
✅ **Interpretable feature importance** analysis  

### Model Architecture

**Algorithm**: Random Forest Regressor  
**Ensemble Size**: 50 decision trees  
**Training Data**: 80% (41,356 samples)  
**Test Data**: 20% (10,340 samples)  

### Input Features (15 features)
- Location (categorical)
- Restaurant type (categorical)
- Veg/Non-Veg type (categorical)
- Cuisines (categorical)
- Online order availability (binary)
- Book table availability (binary)
- Votes (numerical)
- Approx cost for two (numerical)
- Success rate (numerical)
- Cancellation rate (numerical)
- Listed in type (categorical)
- Listed in city (categorical)
- Latitude (numerical)
- Longitude (numerical)
- Distance (numerical)

### Output
- **Target Variable**: Restaurant rating (1-5 scale)
- **Prediction Type**: Continuous regression

### Hyperparameters (Optimized)
```python
n_estimators: 50              # Number of trees in forest
max_depth: 15                 # Maximum tree depth
min_samples_split: 2          # Minimum samples to split node
n_jobs: -1                    # Use all processors
random_state: 42              # Reproducibility
```

### Hyperparameter Tuning
The model was optimized using **GridSearchCV** with 3-fold cross-validation:
- **Tested n_estimators**: [50]
- **Tested max_depth**: [10, 15]
- **Tested min_samples_split**: [2]
- **Best Cross-Validated R²**: 0.772 (77.2%)
- **Best configuration**: max_depth=15, n_estimators=50, min_samples_split=2

---

## 📊 Data Pipeline

### 1️⃣ Data Cleaning
- ✅ Handle missing values
- ✅ Remove duplicates
- ✅ Normalize text fields
- ✅ Standardize data types
- ✅ Fix inconsistencies
- **Result**: 51,696 clean records

### 2️⃣ Exploratory Data Analysis
- 📊 Statistical analysis
- 📈 Distribution analysis
- 🔗 Correlation analysis
- 🎨 Data visualization

### 3️⃣ Feature Engineering
- 🍽️ Extract and normalize dishes
- ✅ Calculate success rates
- ❌ Compute cancellation rates
- 🔤 Encode categorical variables
- 📏 Calculate distances

### 4️⃣ Data Enrichment
- 📍 Fetch restaurant coordinates
- 📏 Calculate distances using Haversine formula
- 🗺️ Merge location data
- 🌍 Geospatial analysis

### 5️⃣ Model Training & Evaluation
- 🤖 Train 5 different algorithms
- 📊 Compare performance metrics
- ⚙️ Tune hyperparameters with GridSearchCV
- 💾 Save best model (Random Forest)

---

## 🔍 Scoring Algorithm

### Multi-Factor Recommendation Score

```
Final Score = (0.45 × Predicted Rating)
            + (0.30 × Success Rate)
            - (0.15 × Cancellation Rate)
            - (0.10 × Normalized Distance)
```

### Weight Explanation
- **45% - Predicted Rating**: Most important factor - ML model's prediction
- **30% - Success Rate**: Reliability of restaurant - order fulfillment
- **15% - Cancellation Rate**: Penalty for unreliable service
- **10% - Distance**: Proximity to user location

---

## 🔌 API Endpoints

### 📄 Pages
| Endpoint | Description |
|----------|-------------|
| `GET /` | Home page with statistics |
| `GET /dashboard` | Analytics dashboard |
| `GET /recommend` | Recommendation interface |
| `GET /explore` | Location explorer |

### 📡 API Endpoints

#### Get Overall Statistics
```
GET /api/get-overall-stats
```
**Response:**
```json
{
  "success": true,
  "stats": {
    "total_restaurants": 500,
    "total_locations": 30,
    "total_dishes": 120,
    "avg_rating": 3.8
  }
}
```

#### Get All Locations
```
GET /api/get-locations
```
**Response:**
```json
{
  "success": true,
  "locations": ["BTM", "Marathalli", "Indiranagar", ...]
}
```

#### Get Dishes for Location
```
POST /api/get-dishes
Content-Type: application/json

{
  "location": "BTM",
  "food_type": "nonveg"
}
```

#### Get Recommendations
```
POST /api/get-recommendations
Content-Type: application/json

{
  "location": "BTM",
  "food_type": "nonveg",
  "dish": "biryani",
  "budget": 500
}
```

**Response:**
```json
{
  "success": true,
  "recommendations": [
    {
      "rank": 1,
      "name": "Restaurant Name",
      "predicted_rating": 4.25,
      "actual_rating": 4.2,
      "distance_km": 2.45,
      "success_rate": 85,
      "cancellation_rate": 12,
      "best_dish": "Biryani",
      "cost_for_two": 450,
      "final_score": 4.18
    }
  ]
}
```

#### Explore Location
```
POST /api/explore-location
Content-Type: application/json

{
  "location": "BTM"
}
```

**Response:**
```json
{
  "success": true,
  "location": "BTM",
  "stats": {
    "total_restaurants": 45,
    "total_unique_dishes": 120,
    "veg_restaurants": 18,
    "nonveg_restaurants": 27
  },
  "restaurants": [
    {
      "name": "Restaurant A",
      "type": "Non-Vegetarian",
      "dishes": ["Biryani", "Chicken Curry", "Tandoori Chicken"]
    }
  ]
}
```

---

## 🛠️ Technologies Used

### 🐍 Data Science & ML
- **pandas** (3.0.0) - Data manipulation and analysis
- **numpy** (2.4.1) - Numerical computing
- **scikit-learn** (1.8.0) - Machine learning algorithms
- **joblib** (1.5.3) - Model serialization and persistence

### 🌐 Web Framework
- **Flask** (3.1.2) - Web application framework
- **Jinja2** (3.1.6) - Template engine
- **flask-cors** (6.0.2) - CORS support

### 📊 Visualization & Analysis
- **matplotlib** (3.10.8) - Plotting library
- **seaborn** (0.13.2) - Statistical visualization
- **folium** (0.20.0) - Interactive maps

### 🗺️ Geospatial
- **geopy** (2.4.1) - Geolocation services
- **geographiclib** (2.1) - Geographic calculations

### 💾 Database
- **SQLite** - Local database
- **mysql-connector-python** (9.5.0) - MySQL support

### 📦 Utilities
- **requests** (2.32.5) - HTTP requests
- **Pillow** (12.1.0) - Image processing
- **python-dateutil** (2.9.0) - Date utilities

---

## 📈 Example Workflows

### Workflow 1️⃣: New User Exploring BTM

```
1. Open http://localhost:5000
2. Click "Explore Location"
3. Select "BTM"
4. Browse all 45 restaurants
5. See 120+ available dishes
6. Discover new cuisines
7. Get recommendations for favorite dish
```

### Workflow 2️⃣: Quick Recommendation via CLI

```bash
python main.py
# Enter: Marathalli
# Enter: veg
# Enter: dosa
# Enter: 300
# Get top 5 veg restaurants serving dosa
```

### Workflow 3️⃣: Budget-Conscious Search

```
1. Open Recommendation Engine
2. Location: Indiranagar
3. Food Type: Non-Veg
4. Dish: Biryani
5. Budget: ₹400
6. Get restaurants within budget
```

### Workflow 4️⃣: Analytics Deep Dive

```
1. Open Dashboard
2. View restaurants by location
3. Analyze veg/non-veg distribution
4. Check rating distribution
5. See cost vs rating correlation
6. Explore distance patterns
```

---

## 🧪 Running Model Building

You can retrain the model with your own data:

### Train and Compare Models
```bash
python model_building.py
```

**Output:**
- Trains 5 different algorithms
- Compares performance metrics
- Selects best model (Random Forest)
- Saves model to `models/best_model.pkl`

### Optimize Hyperparameters
```bash
python hyperparameter_tuning.py
```

**Output:**
- Runs GridSearchCV with 3-fold cross-validation
- Tests different hyperparameter combinations
- Finds optimal configuration
- Saves optimized model

---

## 🐛 Troubleshooting

### ❌ "Data not loaded" Error
**Solution**: Ensure files exist:
```
✓ data/processed/final_dataset.csv
✓ data/enriched/restaurant_locations.csv
✓ models/best_model.pkl
```

### ❌ Port 5000 Already in Use
**Solution**: Change port in `ml_app.py`:
```python
app.run(debug=True, port=5001)
```

### ❌ Missing Dependencies
**Solution**: Reinstall requirements:
```bash
pip install -r requirements.txt --upgrade
```

### ❌ Location Not Found
**Solution**: Check available locations:
```bash
python -c "import pandas as pd; df = pd.read_csv('data/processed/final_dataset.csv'); print(df['location'].unique())"
```

---

## 📝 Example Locations & Dishes

### BTM (Banashankari)
- 🍲 Biryani
- 🥘 Curry
- 🍕 Pizza
- 🥗 Salads

### Marathalli
- 🍜 Noodles
- 🥟 Dumplings
- 🍱 Asian Cuisine
- 🍔 Burgers

### Indiranagar
- 🥙 North Indian
- 🍛 South Indian
- 🍝 Italian
- 🌮 Mexican

### Whitefield
- 🍱 Japanese
- 🥡 Chinese
- 🍕 Continental
- 🥗 Healthy Options

---

## 🎯 Use Cases

### 👨‍💼 Business Professional
- Quick lunch recommendations near office
- Budget-conscious options
- High success rate restaurants

### 👨‍👩‍👧‍👦 Family Outing
- Family-friendly restaurants
- Variety of cuisines
- Good ratings and reviews

### 🎓 Student
- Budget-friendly options (₹100-300)
- Popular student areas
- Quick service restaurants

### 🎉 Special Occasion
- Premium restaurants
- High-rated establishments
- Specific cuisine preferences

---

## 📞 Support & Documentation

### Getting Help
1. 📖 Check README sections
2. 🔍 Review code comments
3. 📊 Check API documentation
4. 🐛 Review troubleshooting section

### Project Information
- 📊 **Dataset**: Kaggle Zomato Bangalore
- 🏙️ **City**: Bangalore, India
- 🏪 **Restaurants**: 500+
- 📍 **Locations**: 30+
- 🍽️ **Dishes**: 100+
- 📈 **Records**: 51,696
- 🤖 **Model**: Random Forest (50 trees)
- 📊 **R² Score**: 0.769 (76.9% accuracy)

---

## 📄 License

This project is open source and available for educational and commercial use.

---

## 🙏 Acknowledgments

- 📊 **Dataset**: [Kaggle - Zomato Bangalore Restaurants](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants)
- 🤖 **ML Libraries**: scikit-learn, pandas, numpy
- 🌐 **Web Framework**: Flask
- 🗺️ **Geospatial**: geopy, folium

---

## 🚀 Getting Started Now

### Quick Start Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Option A: CLI Interface
python main.py

# 2. Option B: Web Interface
python ml_app.py
# Then open http://localhost:5000

# 3. Option C: Retrain Model
python model_building.py
python hyperparameter_tuning.py
```

---

<div align="center">

### 🍽️ **Discover Your Next Favorite Restaurant in Bangalore!** 🍽️

**Made with ❤️ for Bangalore Food Lovers**

**Model Performance**: R² = 0.769 | MAE = 0.122 | RMSE = 0.191

[⬆ Back to Top](#-zomato-restaurant-recommendation-system)

</div>

