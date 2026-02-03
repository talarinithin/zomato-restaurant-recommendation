from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

from model_building import load_data, split_features, get_preprocessor


def run_hyperparameter_tuning():
    print("\n========== HYPERPARAMETER TUNING STARTED ==========\n")

    df = load_data()
    X, y = split_features(df)
    preprocessor = get_preprocessor(X)

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', RandomForestRegressor(random_state=42))
    ])

    param_grid = {
        'model__n_estimators': [50],
        'model__max_depth': [10, 15],
        'model__min_samples_split': [2]
    }

    print("GridSearchCV Configuration:")
    print("Model           : Random Forest")
    print("Cross-validation: 3-fold")
    print("Parameter Grid  :", param_grid)

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring='r2',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("\nBEST PARAMETERS FOUND")
    for k, v in grid.best_params_.items():
        print(f"{k} : {v}")

    print(f"\nBest Cross-Validated R²: {grid.best_score_:.3f}")

    joblib.dump(grid.best_estimator_, "models/best_model.pkl")
    print("\nFinal optimized model saved successfully")

    print("\n========== HYPERPARAMETER TUNING COMPLETED ==========\n")


if __name__ == "__main__":
    run_hyperparameter_tuning()

