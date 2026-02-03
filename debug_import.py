
import sys
import traceback

try:
    print("Attempting to import services.recommender_service...")
    import services.recommender_service
    print("Import successful!")
except Exception:
    traceback.print_exc()
