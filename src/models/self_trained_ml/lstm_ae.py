import torch
import torch.nn as nn

from torch import Tensor
from torch.utils.data import DataLoader


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


def train_LSTMAE(model: LSTMAutoencoder, dataloader: DataLoader, criterion, optimizer, epochs=100):
    for epoch in range(epochs):
        model.train()
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, x)
            loss.backward()
            optimizer.step()


def eval_LTSMAE_MSE(model: LSTMAutoencoder, test_data: Tensor):
    model.eval()
    with torch.no_grad():
        recon = model(test_data)
        return torch.mean((recon - test_data) ** 2, dim=(1, 2))
