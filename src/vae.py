from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class ConvAutoencoder_Simple(pl.LightningModule):
    """
    A simple convolutional autoencoder model. Proven to reconstruct images well.
    """

    def __init__(
        self,
        image_pixel_size: int = 64,
        learning_rate: float = 1e-4,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.save_hyperparameters()
        # Calculate the flattened size after convolutions
        self.flatten_size = (image_pixel_size // 4) * (image_pixel_size // 4) * 128

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Halve the spatial dimensions
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Halve the spatial dimensions again
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Latent space projections
        self.fc_mu = nn.Linear(self.flatten_size, self.hparams.latent_dim)
        self.fc_var = nn.Linear(self.flatten_size, self.hparams.latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.hparams.latent_dim, self.flatten_size),
            nn.ReLU(),
            nn.Unflatten(1, (128, image_pixel_size // 4, image_pixel_size // 4)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Precise upsampling
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Precise upsampling
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),  # Final convolution to get right channels
            nn.Sigmoid(),  # Ensure output is between 0 and 1
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        # In a simple autoencoder, we just use the mean for reconstruction (no sampling)
        return self.decode(mu), mu, log_var

    def loss_function(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor = None,
        log_var: torch.Tensor = None,
    ) -> torch.Tensor:
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        self.log("recon_loss", recon_loss, on_step=False, on_epoch=True)
        return recon_loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x = batch["tensor"]
        recon_x, mu, log_var = self(x)
        loss = self.loss_function(recon_x, x)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x = batch["tensor"]
        recon_x, mu, log_var = self(x)
        loss = self.loss_function(recon_x, x)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1000,
            three_phase=False,
            verbose=False,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run prediction on input data

        Args:
            x: Input tensor of shape [batch_size, channels, height, width] representing the image data.

        Returns:
            Reconstructed images tensor
        """
        self.eval()
        recon_x, _, _ = self(x)
        return recon_x

    @classmethod
    def load_model(cls, checkpoint_path, device="cpu"):
        """
        Load a model from a checkpoint

        Args:
            checkpoint_path: Path to the checkpoint file
            device: Device to load the model to

        Returns:
            Loaded model in eval mode
        """
        model = cls.load_from_checkpoint(checkpoint_path)
        model.to(device)
        model.eval()
        return model


class ConvVAE_Simple(pl.LightningModule):
    """
    A simple convolutional Variational Autoencoder model.
    """

    def __init__(
        self,
        image_pixel_size: int = 64,
        learning_rate: float = 1e-4,
        latent_dim: int = 128,
        kl_weight: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        # Calculate the flattened size after convolutions
        self.flatten_size = (image_pixel_size // 4) * (image_pixel_size // 4) * 128

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Halve the spatial dimensions
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Halve the spatial dimensions again
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Latent space projections
        self.fc_mu = nn.Linear(self.flatten_size, self.hparams.latent_dim)
        self.fc_var = nn.Linear(self.flatten_size, self.hparams.latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.hparams.latent_dim, self.flatten_size),
            nn.ReLU(),
            nn.Unflatten(1, (128, image_pixel_size // 4, image_pixel_size // 4)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Precise upsampling
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Precise upsampling
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),  # Final convolution to get right channels
            nn.Sigmoid(),  # Ensure output is between 0 and 1
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var

    def loss_function(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        # Combine losses
        total_loss = recon_loss + self.hparams.kl_weight * kl_loss
        # Log individual components
        self.log("recon_loss", recon_loss, on_step=False, on_epoch=True)
        self.log("kl_loss", kl_loss, on_step=False, on_epoch=True)
        self.log("total_loss", total_loss, on_step=False, on_epoch=True)
        return total_loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x = batch["tensor"]
        recon_x, mu, log_var = self.forward(x)
        loss = self.loss_function(recon_x, x, mu, log_var)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x = batch["tensor"]
        recon_x, mu, log_var = self.forward(x)
        loss = self.loss_function(recon_x, x, mu, log_var)
        self.log("val_loss", loss, prog_bar=True)
        self.log("mu_mean", mu.mean(), prog_bar=True)
        self.log("log_var_mean", log_var.mean(), prog_bar=True)
        return loss

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1000,
            three_phase=False,
            verbose=False,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run prediction on input data with sampling from the latent space

        Args:
            x: Input tensor of shape [batch_size, channels, height, width] representing the image data.

        Returns:
            Tuple of (reconstructed images, mean, log variance)
        """
        self.eval()
        return self.forward(x)

    @classmethod
    def load_model(cls, checkpoint_path: Path, device: str = "cpu") -> "ConvVAE_Simple":
        """
        Load a model from a checkpoint

        Args:
            checkpoint_path: Path to the checkpoint file
            device: Device to load the model to

        Returns:
            Loaded model in eval mode
        """
        model = cls.load_from_checkpoint(checkpoint_path)
        model.to(device)
        model.eval()
        return model
