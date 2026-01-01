import torch
import torch.nn as nn
import lightning as L
import os
from ray import serve
from ray.train import Checkpoint
from starlette.requests import Request
from typing import Dict

# --- 1. The Model Definition (MUST match training exactly) ---
class MatrixFactorization(L.LightningModule):
    def __init__(self, num_users=1000, num_items=2000, embedding_dim=32, learning_rate=0.1):
        super().__init__()
        # We set default sizes slightly larger to accommodate MovieLens 100k
        # In prod, you'd save these params in the checkpoint config
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user_id, item_id):
        user_vector = self.user_embedding(user_id)
        item_vector = self.item_embedding(item_id)
        return (user_vector * item_vector).sum(1)

# --- 2. The Serving Deployment ---
@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0}
)
class MovieRecommender:
    def __init__(self):
        print("📥 Downloading Model from S3...")
        
        # PASTE YOUR S3 PATH HERE (The folder containing 'checkpoint.ckpt')
        # I constructed this from your logs:
        s3_uri = "s3://training-artifacts-du-yuyang/recommend-models/tuning_run/TorchTrainer_2025-12-31_20-53-21/TorchTrainer_cce85_00000_0_batch_size=512,embedding_dim=32,lr=0.1000_2025-12-31_20-53-25/checkpoint_000009"
        
        # Ray automatically handles the S3 download
        checkpoint = Checkpoint(s3_uri)
        
        with checkpoint.as_directory() as checkpoint_dir:
            # We look for the lightning checkpoint file inside the folder
            ckpt_path = os.path.join(checkpoint_dir, "checkpoint.ckpt")
            print(f"Loading weights from {ckpt_path}")
            
            # Load the Model
            # Note: strict=False allows us to ignore minor mismatch in graph wrapper
            self.model = MatrixFactorization.load_from_checkpoint(ckpt_path, strict=False)
            self.model.eval() # Set to inference mode (freezes dropout, etc)

    async def __call__(self, http_request: Request) -> Dict:
        """
        Expects JSON input:
        {
            "user_id": 42,
            "movie_ids": [100, 200, 300]
        }
        """
        data = await http_request.json()
        user_id = int(data["user_id"])
        movie_ids = data["movie_ids"]
        
        # Convert to Tensors
        user_tensor = torch.tensor([user_id] * len(movie_ids))
        movie_tensor = torch.tensor(movie_ids)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(user_tensor, movie_tensor)
        
        # Return nice JSON
        return {
            "user_id": user_id,
            "predictions": [
                {"movie_id": mid, "predicted_rating": float(score)} 
                for mid, score in zip(movie_ids, predictions)
            ]
        }

# --- 3. The Entrypoint ---
# This binds the deployment to the "sentinel" app
entrypoint = MovieRecommender.bind()
