import torch.nn as nn
import torch

class VAE(nn.Module):
    def __init__(self, config):
        super(VAE, self).__init__()
        self.hidden_dim = config.hidden_dim
        self.latent_dim = config.latent_dim
        self.dropout = config.dropout

        # mnist (28x28)
        # Gaussian encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU()
        )

        # Q phi parameter
        
        self.FC_mean = nn.Linear(self.hidden_dim//2, self.latent_dim)
        self.FC_log_var = nn.Linear(self.hidden_dim//2, self.latent_dim)

        # Bernoulli decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 784),
            nn.Sigmoid()
        )

    def reparameterizantion(self, mean, log_var):

        # log var -> std
        std = torch.exp(0.5*log_var)
        z = torch.randn_like(mean)*std + mean
        return z, mean, log_var


    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        encoded_variable = self.encoder(x)
        mean = self.FC_mean(encoded_variable)
        log_var = self.FC_log_var(encoded_variable)

        z, mean, log_var = self.reparameterizantion(mean, log_var)

        output = self.decoder(z)
        output = output.view(batch_size, -1, 28, 28)

        return output, mean, log_var