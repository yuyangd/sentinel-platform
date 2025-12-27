import ray
from ray import serve
from fastapi import FastAPI

app = FastAPI()

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 0.5, "num_gpus": 0})
@serve.ingress(app)
class SentinelModel:
    def __init__(self):
        # In a real scenario, we load the model here (e.g., from S3/MLflow)
        print("Initializing Sentinel Model...")
        self.model_name = "text-classifier-v1"

    @app.post("/predict")
    def predict(self, data: dict):
        # Mock Inference Logic
        text = data.get("text", "")
        # Simulate a prediction
        sentiment = "positive" if "great" in text.lower() else "neutral"
        return {
            "model": self.model_name,
            "prediction": sentiment,
            "confidence": 0.95
        }

# This part connects to the running Ray cluster
# In KubeRay, the submission script usually handles the connection context automatically
# or we submit it via the Ray Job API.
deployment = SentinelModel.bind()
