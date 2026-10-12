import random
import time

def generate_belief_state():
    """Simulates the Bayesian POMDP trust/cognitive-load estimator"""
    return {
        "belief_state": {
            "trust_mean": round(random.uniform(0.1, 0.9), 2),
            "trust_variance": round(random.uniform(0.01, 0.2), 2),
            "cognitive_load": round(random.uniform(0.1, 0.9), 2),
            "clc_raw": round(random.uniform(0.3, 0.6), 2),
            "emotion_label": random.choice(["frustrated", "focused", "fatigued"]),
            "emotion_confidence": round(random.uniform(0.4, 0.9), 2)
        },
        "task_context": {
            "current_action": "optimize routing protocol",
            "risk_level": random.choice(["low", "med", "high"])
        },
        "interaction_history": []
    }

# In a real system, this would stream state via WebSockets or Kafka 
# into the Middleware layer.
