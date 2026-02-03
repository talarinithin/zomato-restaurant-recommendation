🍽️ Zomato Restaurant Recommendation System
Machine Learning + Flask Web Application
An end-to-end restaurant recommendation system inspired by Zomato that predicts ratings and recommends the top 5 restaurants based on user preferences, distance, and restaurant performance metrics.
📌 Project Overview
This project uses Machine Learning, Data Analysis, and a Flask web interface to help users find the best restaurants based on:
📍 Location
🥗 Veg / 🍗 Non-Veg preference
🍽 Favourite dish (with similarity & spelling correction)
⭐ Predicted restaurant rating
📊 Success rate & cancellation rate
🗺 Distance (map-based visualization)
The system works in real-time and explains why each restaurant is recommended.
🚀 Key Features
🔹 Recommendation System
Dish similarity matching (e.g., dosa → masala dosa)
Auto-correction for spelling mistakes
Veg / Non-Veg validation logic
ML-based rating prediction
Distance-aware ranking
Top-5 personalized restaurant recommendations
🔹 Flask Web Application
The UI contains 3 main pages (buttons):
🍴 Get Recommendations
User form (location, food type, dish, budget)
Dish suggestion step
Final recommendations
Map visualization (Veg vs Non-Veg)
📍 Explore Restaurants
View all restaurants available in a selected location
Shows dishes, cuisines, and food type
📊 Analytics Dashboard
Total restaurants
Veg vs Non-Veg distribution
Top locations
Average ratings per location
Success vs cancellation insights
🗂 Project Structure
Copy code

ZomatoProj/
│
├── app.py                         # Flask web application
├── main.py                        # CLI-based recommendation logic
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
│
├── data/
│   ├── raw/                       # Original Kaggle dataset
│   ├── processed/
│   │   └── final_dataset.csv      # Cleaned & engineered dataset
│   └── enriched/
│       └── restaurant_locations.csv # Latitude & longitude data
│
├── models/
│   └── best_model.pkl             # Trained ML pipeline
│
├── notebooks/
│   └── 02_eda_zomato.py            # EDA & feature engineering
│
├── templates/
│   ├── index.html                 # Landing page
│   ├── recommend.html             # Recommendation form
│   ├── select_dish.html           # Dish selection
│   ├── results.html               # Results + map
│   ├── locations.html             # Location explorer
│   └── dashboard.html             # Analytics dashboard
│
├── static/
│   ├── css/
│   │   └── style.css              # UI styling
│
└── utils/
    ├── fetch_locations_osm.py      # OpenStreetMap geocoding
    └── dataset_inspector.py        # Dataset exploration tool
📊 Dataset Source
Kaggle – Zomato Bangalore Restaurants Dataset
🔗 https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants
Dataset Contains
Restaurant name & location
Cuisines & dishes
Veg / Non-Veg type
Ratings & votes
Approx cost for two
Engineered success & cancellation rates
🔍 Exploratory Data Analysis (EDA)
EDA was performed to:
Understand restaurant distribution across locations
Analyze Veg vs Non-Veg trends
Study success rate vs ratings
Identify important predictive features
📁 File:
Copy code

notebooks/02_eda_zomato.py
Key Insights
Higher success rate → higher rating
Distance affects user preference
Some cuisines perform consistently well
🧠 Machine Learning Model
🔹 Model Details
Target: Restaurant Rating Prediction
Pipeline:
Data preprocessing (ColumnTransformer)
Feature scaling & encoding
Regression model
Hyperparameter tuning using GridSearchCV
🔹 Saved Model
Copy code

models/best_model.pkl
The same trained model is used in:
main.py (CLI)
app.py (Flask UI)
🗺 Map Visualization
Implemented using Folium
Interactive map shown on results page
Marker colors:
🟢 Green → Veg restaurants
🔴 Red → Non-Veg restaurants
Distance calculated using Haversine formula
Fallback logic when coordinates are missing
⚙️ Installation & Setup
1️⃣ Create Virtual Environment
Copy code
Bash
python -m venv zomato_env
2️⃣ Activate Environment
Windows
Copy code
Bash
zomato_env\Scripts\activate
Linux / macOS
Copy code
Bash
source zomato_env/bin/activate
3️⃣ Install Requirements
Copy code
Bash
pip install -r requirements.txt
▶️ How to Run
🔹 Run Flask Web App
Copy code
Bash
python app.py
Open in browser:
Copy code

http://127.0.0.1:5000/
🔹 Run CLI Version (Optional)
Copy code
Bash
python main.py
✅ Validation & Business Rules
Veg user selecting non-veg dish → ❌ error message
Non-veg user selecting veg dish → ❌ prompt to select non-veg
Dish spelling correction & similarity search
Cuisine-based fallback if dish not found
Distance fallback when location data missing
🧰 Technologies Used
Python
Pandas, NumPy
Scikit-Learn
Flask
Folium
HTML, CSS
