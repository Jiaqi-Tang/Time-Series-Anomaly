import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
import torch.optim as optim

from src.models.preprocessing import standardize_ts, create_sliding_windows, np_to_dataloader
from src.visualization.plots import plot_training_accuracy


# Custom LSTM-AE class
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, latent_dim=32, num_layers=1):
        super().__init__()

        self.encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                               num_layers=num_layers, batch_first=True)
        self.latent = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.LSTM(input_size=latent_dim, hidden_size=hidden_dim,
                               num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # Encode
        enc_out, _ = self.encoder(x)
        latent = self.latent(
            enc_out[:, -1, :]).unsqueeze(1).repeat(1, x.size(1), 1)

        # Decode
        dec_out, _ = self.decoder(latent)
        out = self.output_layer(dec_out)
        return out


# For training LSTM
def train_LSTMAE(model: LSTMAutoencoder, dataloader: DataLoader, criterion, optimizer, epochs=100):
    epoch_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0

        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, x)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / batch_count
        epoch_losses.append(avg_loss)
    return epoch_losses


# For testing / evaluating LSTM-AE
def eval_LSTMAE(model, test_data: Tensor, idx: int = 0):
    model.eval()
    with torch.no_grad():
        recon = model(test_data)
    return torch.mean((recon - test_data) ** 2, dim=(1, 2))


# Wrapper function for LSTM-AE model
def model_LSTMAE(ts: pd.Series, window_size=20, batch_size=16, hidden_dim=16, latent_dim=16, num_layers=1,
                 learning_rate=1e-2, epochs=50, plot_accuracy=False):
    g = torch.Generator()
    g.manual_seed(13)

    windows = create_sliding_windows(
        standardize_ts(ts), window_size=window_size)
    data_loader = np_to_dataloader(windows, batch_size=batch_size, generator=g)

    input_dim = next(iter(data_loader))[0].shape[-1]

    model = LSTMAutoencoder(input_dim=input_dim,
                            hidden_dim=hidden_dim, latent_dim=latent_dim, num_layers=num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = train_LSTMAE(
        model, data_loader, criterion=criterion, optimizer=optimizer, epochs=epochs)
    if plot_accuracy:
        plot_training_accuracy(losses, title="Taring loss of LSTM-AE")

    reconstruction_error = eval_LSTMAE(model, test_data=torch.tensor(
        windows, dtype=torch.float32))
    return reconstruction_error, model
