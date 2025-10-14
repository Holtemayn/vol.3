
import joblib
from app.core.config import settings

_MODEL = None

def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = joblib.load(settings.MODEL_PATH)
    return _MODEL
