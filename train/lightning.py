import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
import pandas as pd
from torch.utils.data import DataLoader, Dataset


class MatrixFactorization(L.LightningModule):
    def __init__(self, num_users, num_items, embedding_dim):
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


class MovieLensDataset(Dataset):
    def __init__(self, dataframe=None):
        self.dataframe = dataframe or self.load_movielens_data()
        self.num_users = self.dataframe['user_id'].nunique()
        self.num_items = self.dataframe['item_id'].nunique()
    
    @staticmethod
    def load_movielens_data(path_u='data/ml-100k/u.data'):
        column_names = ['user_id', 'item_id', 'rating', 'timestamp']
        df = pd.read_csv(path_u, sep='\t', names=column_names)
        df['user_id'] -= 1
        df['item_id'] -= 1
        return df

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        user_id = self.dataframe.iloc[idx, 0]
        item_id = self.dataframe.iloc[idx, 1]
        rating = self.dataframe.iloc[idx, 2]
        return (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(item_id, dtype=torch.long),
            torch.tensor(rating, dtype=torch.float)
        )


if __name__ == '__main__':
    dataset = MovieLensDataset()
    train_data = DataLoader(dataset, batch_size=512, shuffle=True)

    model = MatrixFactorization(dataset.num_users, dataset.num_items, embedding_dim=20)
    trainer = L.Trainer(max_epochs=10)
    trainer.fit(model, train_data)
