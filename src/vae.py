import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam


class ConvVAE(pl.LightningModule):
    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 128,
        hidden_dims: list[int] = [32, 64, 128, 256],
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim

        # Calculate the dimension after all convolutions
        # assuming input is 64x64 and each conv reduces size by half (stride=2)
        # For n convolutional layers: 64 -> 32 -> 16 -> 8 -> 4 (etc)
        self.last_encoder_size = 64 // (2 ** len(hidden_dims))  # Size after last conv layer
        self.last_encoder_channels = hidden_dims[-1]  # Channels after last conv
        
        print(f"Debug - Network structure: input_size=64, hidden_dims={hidden_dims}")
        print(f"Debug - Last encoder size: {self.last_encoder_size}, channels: {self.last_encoder_channels}")

        # Encoder
        encoder_layers = []
        in_channels = input_channels

        for h_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                ]
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space projections
        # Calculate flattened size: channels × height × width
        self.flatten_size = hidden_dims[-1] * self.last_encoder_size * self.last_encoder_size
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_var = nn.Linear(self.flatten_size, latent_dim)

        # Decoder first layer to convert from latent space back to conv feature maps
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)

        # Decoder
        decoder_layers = []
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            decoder_layers.extend(
                [
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                ]
            )

        decoder_layers.extend(
            [
                nn.ConvTranspose2d(
                    hidden_dims[-1],
                    hidden_dims[-1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.LeakyReLU(),
                nn.Conv2d(hidden_dims[-1], input_channels, kernel_size=3, padding=1),
                nn.Sigmoid(),
            ]
        )

        self.decoder = nn.Sequential(*decoder_layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # First project from latent space to flattened convolutional features
        z = self.decoder_input(z)
        # Reshape to match the encoder's output shape before flattening
        z = z.view(-1, self.last_encoder_channels, self.last_encoder_size, self.last_encoder_size)
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def loss_function(self, recon_x, x, mu, log_var):
        """
        Computes the VAE loss function.
        The loss function consists of two components:

        1. Binary Cross Entropy (BCE) - Reconstruction loss that measures how
            well the decoder reconstructs the input image from the latent space.
            Lower values indicate better reconstruction quality.
        2. Kullback-Leibler Divergence (KLD) - Regularization term that encourages
            the latent space distribution to approximate a standard normal distribution.
            It prevents the model from simply memorizing the training data.

        The total loss is the sum of these two components, balancing reconstruction
        quality with the regularization of the latent space.
        """
        # Get batch size for normalization
        batch_size = x.size(0)
        
        # Use mean reduction to normalize by the number of elements
        BCE = F.binary_cross_entropy(recon_x, x, reduction="mean")
        
        # Normalize KLD by batch size and number of elements per sample
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size
        
        # Optionally scale KLD to balance with BCE (beta-VAE)
        beta = 1.0  # Adjust this value to control KLD weight
        
        # Combine the losses
        return BCE + beta * KLD

    def training_step(self, batch, batch_idx):
        x = batch["tensor"]
        # Print shape for debugging
        if batch_idx == 0:
            print(f"Training input tensor shape: {x.shape}, dim: {x.dim()}")
        
        recon_x, mu, log_var = self(x)
        loss = self.loss_function(recon_x, x, mu, log_var)
        # Only log on the last batch to reduce overhead
        if batch_idx % 50 == 0:
            self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["tensor"]
        # Print shape for debugging
        if batch_idx == 0:
            print(f"Validation input tensor shape: {x.shape}, dim: {x.dim()}")
        
        recon_x, mu, log_var = self(x)
        loss = self.loss_function(recon_x, x, mu, log_var)
        # Only log once per validation to reduce overhead
        if batch_idx == 0:
            self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)
