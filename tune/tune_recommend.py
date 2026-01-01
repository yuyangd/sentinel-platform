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
# --- NEW IMPORTS FOR TUNING ---
from ray import tune
from ray.train.lightning import RayDDPStrategy, RayTrainReportCallback
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig
from ray.tune.schedulers import ASHAScheduler

# --- 1. The Model (Modified to accept dynamic config) ---
class MatrixFactorization(L.LightningModule):
    # Added learning_rate to init so we can tune it
    def __init__(self, num_users, num_items, embedding_dim, learning_rate):
        super().__init__()
        self.save_hyperparameters() # Lightning best practice
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.learning_rate = learning_rate

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
        # Use the tuned learning rate
        return optim.SGD(self.parameters(), lr=self.learning_rate)

# --- 2. The Data (Same as before) ---
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
            
    def load_data(self):
        data_path = os.path.join(self.data_dir, "ml-100k/u.data")
        columns = ['user_id', 'item_id', 'rating', 'timestamp']
        df = pd.read_csv(data_path, sep='\t', names=columns)
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

# --- 3. The Execution Engine (Modified for Tuning) ---
def train_func_per_worker(config):
    # Retrieve hyperparameters from Ray Tune's config dictionary
    lr = config["lr"]
    emb_dim = config["embedding_dim"]
    batch_size = config["batch_size"]
    
    dataset = MovieLensDataset()
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Pass the config values into the model
    model = MatrixFactorization(dataset.num_users, dataset.num_items, emb_dim, lr)
    
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=None, save_top_k=1, monitor="train_loss", mode="min"
    )
    
    trainer = L.Trainer(
        max_epochs=10, # Give it time to converge
        devices="auto",
        accelerator="cpu",
        strategy=RayDDPStrategy(),
        enable_progress_bar=False,
        use_distributed_sampler=False,
        callbacks=[checkpoint_callback, RayTrainReportCallback()],
    )
    
    trainer.fit(model, train_dataloaders=train_loader)

if __name__ == "__main__":
    print("🧪 Starting Hyperparameter Tuning...")
    
    storage_path = "s3://training-artifacts-du-yuyang/recommend-models/tuning_run"

    # B. Define the Search Space (The "Menu")
    # Ray will mix and match these values.
    param_space = {
        "train_loop_config": {
            "lr": tune.grid_search([0.1, 0.01, 0.001, 0.0001]), # Try all 4
            "embedding_dim": tune.choice([16, 32, 64]),         # Pick random
            "batch_size": tune.choice([256, 512, 1024]),
        }
    }

    # C. Define the Scheduler (The "Terminator")
    # ASHA will monitor trials. If a trial (e.g., lr=0.1) has awful loss at Epoch 1, 
    # ASHA will kill it immediately to save money/time.
    scheduler = ASHAScheduler(
        max_t=10, 
        grace_period=2, 
        metric="train_loss", 
        mode="min"
    )

    # A. Define the Trainer (The "Recipe")
    # This is a template. Ray Tune will clone this for every trial.
    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        scaling_config=ScalingConfig(
            num_workers=1,                 # Lower worker count to reduce CPU bundles
            use_gpu=False,
            resources_per_worker={"CPU": 1},
            trainer_resources={"CPU": 0},  # Avoid reserving an extra CPU for trainer actor
        ),
        run_config=RunConfig(storage_path=storage_path) # All trials save to S3
    )

    # D. Launch the Tuner (The "Manager")
    tuner = tune.Tuner(
        trainer,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            num_samples=2, # Number of random samples to try (combined with grid_search)
            scheduler=scheduler,
        ),
    )
    
    results = tuner.fit()
    
    # E. The Victory Lap
    best_result = results.get_best_result(metric="train_loss", mode="min")
    print(f"🏆 Best Result Found!")
    print(f"   Loss: {best_result.metrics['train_loss']}")
    print(f"   Config: {best_result.config['train_loop_config']}")
    print(f"   Checkpoint: {best_result.checkpoint}")
