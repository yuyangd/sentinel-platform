import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import requests
import zipfile
import io
import os
from ray.train.lightning import RayDDPStrategy, RayTrainReportCallback
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig


# --- 1. The Model (Pure PyTorch Lightning - No Ray logic here) ---
class MatrixFactorization(L.LightningModule):
    def __init__(self, num_users, num_items, embedding_dim=20):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user_id, item_id):
        user_vector = self.user_embedding(user_id)
        item_vector = self.item_embedding(item_id)
        return (user_vector * item_vector).sum(1)

    def training_step(self, batch, batch_idx):
        user_id, item_id, rating = batch
        prediction = self(user_id, item_id)
        loss = nn.functional.mse_loss(prediction, rating)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=0.01)

# --- 2. The Data (Cloud-Native: Downloads itself) ---
class MovieLensDataset(Dataset):
    def __init__(self):
        self.data_dir = os.path.join(os.getcwd(), "data")
        self.download_data_if_needed()
        self.dataframe = self.load_data()
        self.num_users = self.dataframe['user_id'].max() + 1
        self.num_items = self.dataframe['item_id'].max() + 1

    def download_data_if_needed(self):
        os.makedirs(self.data_dir, exist_ok=True)
        file_path = os.path.join(self.data_dir, "u.data")
        if not os.path.exists(file_path):
            print("Downloading MovieLens 100k data...")
            url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
            r = requests.get(url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(self.data_dir)
            # Move internal file to expected location if needed, or just read from extracted path
            # The zip extracts a folder 'ml-100k'.
            
    def load_data(self):
        # Path inside the extracted zip
        data_path = os.path.join(self.data_dir, "ml-100k/u.data")
        columns = ['user_id', 'item_id', 'rating', 'timestamp']
        df = pd.read_csv(data_path, sep='\t', names=columns)
        # Fix indexing to start from 0
        df['user_id'] -= 1
        df['item_id'] -= 1
        return df

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        return (
            torch.tensor(row['user_id'], dtype=torch.long),
            torch.tensor(row['item_id'], dtype=torch.long),
            torch.tensor(row['rating'], dtype=torch.float)
        )

# --- 3. The Execution Engine (Ray Train Logic) ---
def train_func_per_worker(config):
# Prepare Data
    dataset = MovieLensDataset()
    train_loader = DataLoader(dataset, batch_size=512, shuffle=False)
    
    # Prepare Model
    model = MatrixFactorization(dataset.num_users, dataset.num_items)
    
    # Keep your existing Lightning Checkpoint logic (controls "Best" vs "Last")
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=None,  
        save_top_k=1,
        monitor="train_loss",
        mode="min",
        filename="sentinel-model-{epoch}-{step}" # Optional: Naming
    )
    
    # Prepare Trainer
    trainer = L.Trainer(
        max_epochs=5,
        devices="auto",
        accelerator="cpu",
        strategy=RayDDPStrategy(),
        enable_progress_bar=False,
        use_distributed_sampler=False,
        # 2. ADD THE RAY CALLBACK HERE
        callbacks=[checkpoint_callback, RayTrainReportCallback()], 
    )
    
    # Train
    trainer.fit(model, train_dataloaders=train_loader)

if __name__ == "__main__":
    print("🚀 Submitting Job to Ray Cluster...")
    
    # Define Storage Path (S3 bucket)
    storage_path = "s3://training-artifacts-du-yuyang/recommend-models/"
    
    # Define Resources: 1 Worker for local testing
    # For production on Ray cluster, increase to num_workers=2+
    scaling_config = ScalingConfig(
        num_workers=4, 
        use_gpu=False,
        resources_per_worker={"CPU": 1} 
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        scaling_config=scaling_config,
        run_config=RunConfig(
            name="sentinel_training_run",
            storage_path=storage_path,  # <--- Save checkpoints to S3
        )
    )
    
    result = trainer.fit()
    print(f"✅ Training Complete!")
    print(f"📦 Checkpoint saved at: {result.checkpoint}")
    print(f"📊 Results: {result.metrics}")
