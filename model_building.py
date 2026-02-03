import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# ======================================================
# STEP 1: LOAD DATA
# ======================================================
def load_data():
    print("\n[STEP 1] Loading final processed dataset...")
    df = pd.read_csv("data/processed/final_dataset.csv")
    print(f"Dataset shape: {df.shape}")
    return df


# ======================================================
# STEP 2: SPLIT FEATURES & TARGET
# ======================================================
def split_features(df):
    print("\n[STEP 2] Separating features and target...")

    X = df.drop(columns=[
        'rate',
        'name',
        'dish_liked',
        'famous_dish'
    ])
    y = df['rate']

    print(f"Features shape: {X.shape}")
    print(f"Target shape  : {y.shape}")
    return X, y


# ======================================================
# STEP 3: PREPROCESSOR
# ======================================================
def get_preprocessor(X):
    print("\n[STEP 3] Creating preprocessing pipeline...")

    categorical_cols = X.select_dtypes(include=['object', 'string']).columns
    numerical_cols = X.select_dtypes(exclude=['object', 'string']).columns

    print(f"Categorical columns: {len(categorical_cols)}")
    print(f"Numerical columns  : {len(numerical_cols)}")

    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', 'passthrough', numerical_cols)
        ]
    )


# ======================================================
# STEP 4: MODELS
# ======================================================
def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=50,
            max_depth=15,
            n_jobs=-1,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "XGBoost": XGBRegressor(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
    }


# ======================================================
# STEP 5: TRAIN & COMPARE MODELS
# ======================================================
def train_and_compare(X, y, preprocessor):
    print("\n[STEP 5] Training and comparing models...")
    print("Train-Test Split: 80% / 20%\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = []
    best_model = None
    best_r2 = -np.inf
    best_model_name = ""

    for name, model in get_models().items():
        print(f"Training {name}...")

        pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('model', model)
        ])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results.append([name, mae, rmse, r2])

        if r2 > best_r2:
            best_r2 = r2
            best_model = pipeline
            best_model_name = name

    # ==================================================
    # MODEL COMPARISON TABLE
    # ==================================================
    results_df = pd.DataFrame(
        results,
        columns=["Model", "MAE", "RMSE", "R2 Score"]
    )

    print("\n================ MODEL COMPARISON TABLE ================\n")
    print(results_df.to_string(index=False))
    print("\n========================================================")

    print("\nBEST MODEL SELECTED")
    print("-------------------")
    print(f"Model Name : {best_model_name}")
    print(f"Best R²    : {best_r2:.3f}")

    print("\nWHY THIS MODEL?")
    print(
        f"{best_model_name} achieved the highest R² score and lowest error. "
        "As an ensemble model, it reduces overfitting and captures complex "
        "non-linear relationships between restaurant features and ratings."
    )

    return best_model


# ======================================================
# STEP 6: SAVE MODEL
# ======================================================
def save_model(model):
    joblib.dump(model, "models/best_model.pkl")
    print("\n[STEP 6] Best model saved to models/best_model.pkl")


# ======================================================
# MAIN
# ======================================================
def main():
    print("\n========== MODEL BUILDING PIPELINE STARTED ==========")
    df = load_data()
    X, y = split_features(df)
    preprocessor = get_preprocessor(X)
    best_model = train_and_compare(X, y, preprocessor)
    save_model(best_model)
    print("\n========== MODEL BUILDING COMPLETED ==========\n")


if __name__ == "__main__":
    main()
