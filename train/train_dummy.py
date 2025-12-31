import ray
from ray import train
from ray.train import ScalingConfig, Checkpoint
from ray.train.torch import TorchTrainer
import torch
import torch.nn as nn
import tempfile
import os
import time

# 1. Define the Neural Network (Simple Dummy Model)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# 2. Define the Training Loop (The "Worker" Logic)
def train_func(config):
    # Initialize Model
    model = SimpleModel()
    
    # Simulate Data Loading & Training
    # In a real interview, you'd mention "Streaming Datasets" here
    for epoch in range(10):
        # Simulate computation work (burning CPU to show up on Grafana!)
        start = time.time()
        while time.time() - start < 5: # Burn 5 seconds per epoch
            _ = [x**2 for x in range(10000)]
            
        loss = 0.1 * (10 - epoch) # Fake loss decreasing
        
        # Report metrics to Ray (and thus Prometheus)
        train.report({"loss": loss, "epoch": epoch})
        
        # Save Checkpoint (every 5 epochs)
        if epoch % 5 == 0:
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                torch.save(model.state_dict(), os.path.join(temp_checkpoint_dir, "model.pt"))
                
                # Create a Ray Checkpoint object
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                
                # Report with checkpoint (This triggers the upload to S3)
                train.report({"loss": loss}, checkpoint=checkpoint)

    print("Training Complete!")

# 3. The Orchestration (The "Driver" Logic)
if __name__ == "__main__":
    ray.init()
    
    # YOUR BUCKET HERE
    # Ray will automatically sync checkpoints to this S3 path
    storage_path = "s3://training-artifacts-du-yuyang/runs/"

    print(f"Submitting job. Artifacts will be saved to: {storage_path}")

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        scaling_config=ScalingConfig(num_workers=2, use_gpu=False), # We use 2 workers to test distributed logic
        run_config=train.RunConfig(
            storage_path=storage_path,
            name="sentinel-finetune-v1"
        )
    )

    result = trainer.fit()
    print(f"Best Checkpoint saved at: {result.checkpoint}")