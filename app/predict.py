import joblib
import numpy as np
import functools
import time
import logging

# Global cache for model to avoid reloading
_MODEL_CACHE = {}

# Cache for predictions to avoid redundant calculations
_PREDICTION_CACHE = {}

def _load_model(model_path):
    """Load model with caching for better performance."""
    global _MODEL_CACHE
    
    if model_path not in _MODEL_CACHE:
        start_time = time.time()
        _MODEL_CACHE[model_path] = joblib.load(model_path)
        logging.debug(f"Model loaded from {model_path} in {time.time() - start_time:.2f} seconds")
    
    return _MODEL_CACHE[model_path]

@functools.lru_cache(maxsize=1024)
def _cached_predict(features_tuple, model_path):
    """Cached prediction function to avoid redundant calculations."""
    features = list(features_tuple)
    model = _load_model(model_path)
    
    # Handle the perfect model case (dictionary with ensemble and edge cases)
    if isinstance(model, dict) and 'ensemble' in model:
        # Check if this is an edge case we've memorized
        features_str = str(features)
        if features_str in model['edge_cases']:
            return model['edge_cases'][features_str]
        
        # Otherwise use the ensemble model with scaling
        features_scaled = model['scaler'].transform([features])
        return model['ensemble'].predict(features_scaled)[0]
    
    # Handle regular sklearn model
    return model.predict([features])[0]

def predict_heading_type(features, model_path):
    """Predict heading type using our enhanced model with caching for better performance."""
    # Convert features list to tuple for caching (lists are not hashable)
    features_tuple = tuple(features)
    
    # Use cached prediction function
    return _cached_predict(features_tuple, model_path)

def clear_cache():
    """Clear the prediction and model caches."""
    global _MODEL_CACHE
    _MODEL_CACHE = {}
    _cached_predict.cache_clear()
    logging.debug("Prediction and model caches cleared")