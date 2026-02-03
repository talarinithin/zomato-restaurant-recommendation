import joblib

model = joblib.load("models/best_model.pkl")
print("Model loaded successfully!")
print("Model type:", type(model))
