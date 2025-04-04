from tqdm import tqdm
from PIL import Image
from pathlib import Path

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
import wandb

from src.dataset import load_and_preprocess_dataset
from src.utils import tensor_to_pil_image, pil_image_concat


######## BARE TORCH CONV AUTOENCODER FROM A PREVIOUS PROJECT #########


def new_model(image_pixel_size: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        # Encoder
        torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),  # Halve the spatial dimensions
        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),  # Halve the spatial dimensions again
        torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        # Flatten for latent space
        torch.nn.Flatten(),
        torch.nn.Linear(
            (image_pixel_size // 4) * (image_pixel_size // 4) * 128, 128
        ),  # Latent representation
        # Decoder
        torch.nn.Linear(128, (image_pixel_size // 4) * (image_pixel_size // 4) * 128),
        torch.nn.ReLU(),
        # Reshape for convolutions
        torch.nn.Unflatten(1, (128, image_pixel_size // 4, image_pixel_size // 4)),
        torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Precise upsampling
        torch.nn.ReLU(),
        torch.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Precise upsampling
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 3, kernel_size=3, padding=1),  # Final convolution to get right channels
    )


def prepare_input(input_tensor: torch.Tensor, device: str) -> torch.Tensor:
    """Format input tensor based on model type.

    Args:
        input_tensor: Input tensor of shape (batch_size, channels, height, width)
        model_type: Type of model being used

    Returns:
        Formatted input tensor ready for model
    """
    input_tensor: torch.Tensor = input_tensor.unsqueeze(0).to(device)
    return input_tensor


def prepare_output(
    output_tensor: torch.Tensor,
    image_pixel_size: int,
    remove_batch: bool = False,
) -> torch.Tensor:
    """Format output tensor based on model type.

    Args:
        output_tensor: Output tensor from model
        model_type: Type of model being used
        image_pixel_size: Size of the image in pixels
        remove_batch: Whether to remove the batch dimension

    Returns:
        Formatted output tensor in shape (batch_size, channels, height, width) or (channels, height, width) if remove_batch=True
    """
    outputs = output_tensor.view(output_tensor.size(0), 3, image_pixel_size, image_pixel_size)
    # Common output formatting
    outputs = outputs.clamp(0, 1)  # Ensure values are between 0 and 1
    outputs = outputs.cpu()
    if remove_batch:
        outputs = outputs.squeeze(0)  # Remove batch dimension
    return outputs


def train_predict(
    model: torch.nn.Sequential,
    input_tensor_batch: torch.Tensor,
    no_grad: bool = False,
) -> torch.Tensor:
    """
    Predict output of model from input tensor batch.

    Args:
        model: Model to predict from
        input_tensor_batch: Input tensor batch
        no_grad: Whether to run in no_grad mode

    Returns:
        Predicted output tensor batch
    """
    if no_grad:
        with torch.no_grad():
            outputs: torch.Tensor = model(input_tensor_batch)
    else:
        outputs: torch.Tensor = model(input_tensor_batch)
    return outputs


def predict(
    model_path: str,
    input_tensor: torch.Tensor,
    image_pixel_size: int,
    device: str,
) -> torch.Tensor:
    # Load model
    model: torch.nn.Sequential = new_model(image_pixel_size=image_pixel_size)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    # Predict
    input_tensor = prepare_input(input_tensor, device)
    with torch.no_grad():
        outputs: torch.Tensor = model(input_tensor)
        outputs = prepare_output(outputs, image_pixel_size, remove_batch=True)
    return outputs


def predict_from_dataset_index(
    model_path: str,
    ds: torch.utils.data.Dataset,
    ds_index: int,
    image_pixel_size: int,
    device: str,
) -> torch.Tensor:
    input_tensor: torch.Tensor = ds[ds_index]["tensor"]
    output_tensor: torch.Tensor = predict(
        model_path=model_path,
        input_tensor=input_tensor,
        image_pixel_size=image_pixel_size,
        device=device,
    )
    return output_tensor


def train(
    image_pixel_size: int,
    batch_size: int,
    num_epochs: int,
    device: str,
    overfit: bool = False,
    wandb_project: str = "pixelart-autoencoder",
    num_viz_images: int = 5,
):
    config_dict = {k: v for k, v in locals().items()}

    # Initialize wandb
    print("Initializing wandb...")
    run: wandb.Run = wandb.init(
        project=wandb_project,
        config=config_dict,
    )

    # Load and preprocess the datasets
    print("Loading and preprocessing datasets...")
    train_ds: torch.utils.data.Dataset = load_and_preprocess_dataset(
        path="tkarr/sprite_caption_dataset",
        split="train",
        image_pixel_size=image_pixel_size,
    )
    val_ds: torch.utils.data.Dataset = load_and_preprocess_dataset(
        path="tkarr/sprite_caption_dataset",
        split="valid",
        image_pixel_size=image_pixel_size,
    )
    if overfit:
        train_ds = train_ds.select(range(100))
        val_ds = train_ds

    print("Creating data loaders...")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print("Instantiating model...")
    model = new_model(image_pixel_size=image_pixel_size)
    model = model.to(device)

    print("Defining loss function and optimizer...")
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW

    print("Training the model...")
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Training
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Training epoch {epoch+1}"):
            optimizer.zero_grad()
            inputs = batch["tensor"].to(device)
            outputs: torch.Tensor = train_predict(model, inputs, no_grad=False)
            loss: torch.Tensor = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["tensor"].to(device)
                outputs: torch.Tensor = train_predict(model, inputs, no_grad=True)
                loss: torch.Tensor = criterion(outputs, inputs)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader)

        # Log sample input and reconstruction images side by side in a single grid
        imgs: list[Image.Image] = []
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            viz_batch = next(iter(val_loader))
            viz_inputs: torch.Tensor = viz_batch["tensor"][:num_viz_images].to(device)
            viz_outputs: torch.Tensor = train_predict(model, viz_inputs, no_grad=True)
            # Create pairs of original and reconstructed images
            for idx in range(num_viz_images):
                # Convert tensors to PIL images and append
                imgs.append(tensor_to_pil_image(viz_inputs[idx]))
                imgs.append(tensor_to_pil_image(viz_outputs[idx]))
            run.log(
                {
                    "comparisons": wandb.Image(
                        pil_image_concat(imgs),
                        mode="RGB",
                        caption="Left: Original, Right: Reconstruction",
                    )
                }
            )

        # Log metrics and images to wandb
        run.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            # Log best model to wandb
            print(f"Model saved at epoch {epoch+1} with validation loss: {val_loss:.4f}")

    # Log details
    print("Training complete!")
    print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print("Config:")
    for k, v in config_dict.items():
        print(f"  {k}: {v}")

    # Close wandb run
    run.finish()


######## VIBE CODED CONV THAT DIDN'T WORK #########


class ConvVAE(pl.LightningModule):
    def __init__(
        self,
        input_channels: int,
        latent_dim: int,
        hidden_dims: list[int],
        learning_rate: float,
        kl_weight: float,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

        # Save the original hidden dims before we modify them
        self.hidden_dims = hidden_dims.copy()

        # Calculate the dimension after all convolutions
        # We'll use stride=1 and max pooling for downsampling to preserve more information
        self.last_encoder_size = 64 // (2 ** len(hidden_dims))
        self.last_encoder_channels = hidden_dims[-1]

        # Encoder
        encoder_layers = []
        in_channels = input_channels
        self.skip_connections = []

        for i, h_dim in enumerate(hidden_dims):
            # Main convolution path with stride=1
            encoder_layers.extend(
                [
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                    # Extra conv layer for more expressiveness
                    nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                    # Max pooling for downsampling (preserves edges better than strided conv)
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space projections
        self.flatten_size = hidden_dims[-1] * self.last_encoder_size * self.last_encoder_size
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_var = nn.Linear(self.flatten_size, latent_dim)

        # Decoder first layer
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)

        # Decoder
        decoder_layers = []
        reversed_dims = hidden_dims.copy()
        reversed_dims.reverse()

        for i in range(len(reversed_dims) - 1):
            # Upsample + conv instead of transposed conv for better quality
            decoder_layers.extend(
                [
                    nn.Upsample(
                        scale_factor=2, mode="nearest"
                    ),  # Nearest neighbor upsampling for pixel art
                    nn.Conv2d(
                        reversed_dims[i], reversed_dims[i + 1], kernel_size=3, stride=1, padding=1
                    ),
                    nn.BatchNorm2d(reversed_dims[i + 1]),
                    nn.LeakyReLU(0.2, inplace=True),
                    # Extra conv layer
                    nn.Conv2d(
                        reversed_dims[i + 1],
                        reversed_dims[i + 1],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.BatchNorm2d(reversed_dims[i + 1]),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )

        # Final decoder block
        decoder_layers.extend(
            [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(reversed_dims[-1], reversed_dims[-1], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(reversed_dims[-1]),
                nn.LeakyReLU(0.2, inplace=True),
                # Extra conv layers for better detail
                nn.Conv2d(reversed_dims[-1], reversed_dims[-1], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(reversed_dims[-1]),
                nn.LeakyReLU(0.2, inplace=True),
                # Final conv to RGB
                nn.Conv2d(reversed_dims[-1], input_channels, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid(),
            ]
        )

        self.decoder = nn.Sequential(*decoder_layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
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

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std: torch.Tensor = torch.exp(0.5 * log_var)
        eps: torch.Tensor = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # First project from latent space to flattened convolutional features
        z = self.decoder_input(z)
        # Reshape to match the encoder's output shape before flattening
        z = z.view(-1, self.last_encoder_channels, self.last_encoder_size, self.last_encoder_size)
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def loss_function(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """
        Loss function specifically designed for pixel art with:
        1. MSE for overall reconstruction
        2. Sharpness penalty to prevent blur
        3. Color preservation term
        4. Very small KL weight
        """
        # Basic MSE reconstruction loss
        mse_loss = F.mse_loss(recon_x, x, reduction="mean")

        # Sharpness penalty using Laplacian filter
        laplacian_kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=x.device
        ).view(1, 1, 3, 3)

        # Apply Laplacian to both original and reconstructed
        x_edges = F.conv2d(x, laplacian_kernel.expand(3, -1, -1, -1), padding=1, groups=3)
        recon_edges = F.conv2d(recon_x, laplacian_kernel.expand(3, -1, -1, -1), padding=1, groups=3)
        sharpness_loss = F.mse_loss(recon_edges, x_edges, reduction="mean")

        # Color preservation term
        x_colors = x.view(x.size(0), 3, -1).mean(dim=2)  # Average color per channel
        recon_colors = recon_x.view(recon_x.size(0), 3, -1).mean(dim=2)
        color_loss = F.mse_loss(recon_colors, x_colors, reduction="mean")

        # KL Divergence term (very small weight)
        KLD = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))
        beta = 0.00001 * self.kl_weight  # Even smaller KL impact

        # Combine losses with weights
        total_loss = (
            mse_loss  # Base reconstruction
            + 0.5 * sharpness_loss  # Encourage sharp edges
            + 0.2 * color_loss  # Preserve colors
            + beta * KLD  # Very small KL term
        )

        # Log components
        self.log("mse_loss", mse_loss, on_step=False, on_epoch=True)
        self.log("sharpness_loss", sharpness_loss, on_step=False, on_epoch=True)
        self.log("color_loss", color_loss, on_step=False, on_epoch=True)
        self.log("kld_loss", KLD, on_step=False, on_epoch=True)
        self.log("kl_weight", beta, on_step=False, on_epoch=True)

        return total_loss

    def loss_function_alt(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the VAE loss function with a modified reconstruction loss
        that is better suited for pixel art images.
        """
        # Modified reconstruction loss with MSE and BCE combined
        # MSE helps with capturing color details and gradients
        MSE = F.mse_loss(recon_x, x, reduction="mean")

        # BCE helps with sharp pixel edges and discrete color transitions
        # Ensure inputs are properly bounded between 0 and 1 for numerical stability
        recon_x_clamped = torch.clamp(recon_x, 1e-6, 1.0 - 1e-6)
        BCE = F.binary_cross_entropy(recon_x_clamped, x, reduction="mean")

        # Combine reconstruction losses with adjusted weighting
        # Higher BCE weight for pixel art as it handles sharp edges better
        recon_loss = 0.3 * MSE + 0.7 * BCE

        # KL Divergence term (Kullback-Leibler Divergence)
        # This encourages the latent distribution to be close to a standard normal distribution
        KLD = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))

        # Scale KLD with beta parameter for better disentanglement (beta-VAE)
        # Lower values prioritize reconstruction quality over latent space regularization
        beta = self.kl_weight

        # Log individual loss components for monitoring
        self.log("mse_loss", MSE, on_step=False, on_epoch=True)
        self.log("bce_loss", BCE, on_step=False, on_epoch=True)
        self.log("recon_loss", recon_loss, on_step=False, on_epoch=True)
        self.log("kld_loss", KLD, on_step=False, on_epoch=True)
        self.log("kl_weight", beta, on_step=False, on_epoch=True)

        # Combine the losses - for pixel art we prioritize reconstruction
        total_loss = recon_loss + beta * KLD
        return total_loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x = batch["tensor"]
        # Debug print to check input values
        print(f"Training input min: {x.min()}, max: {x.max()}, mean: {x.mean()}")
        recon_x, mu, log_var = self(x)
        # Debug print to check output values
        print(f"Training output min: {recon_x.min()}, max: {recon_x.max()}, mean: {recon_x.mean()}")
        loss = self.loss_function(recon_x, x, mu, log_var)
        # Only log on the last batch to reduce overhead
        if batch_idx % 50 == 0:
            self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x = batch["tensor"]
        recon_x, mu, log_var = self(x)
        loss = self.loss_function(recon_x, x, mu, log_var)
        # Only log once per validation to reduce overhead
        if batch_idx == 0:
            self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> dict:
        """
        Configure optimizers and learning rate schedulers for training.
        Uses AdamW with weight decay and a OneCycleLR scheduler for better convergence.
        """
        # Use AdamW instead of Adam for better weight regularization
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.01,  # Add weight decay to prevent overfitting
            betas=(0.9, 0.999),
        )

        # Use a OneCycleLR scheduler for faster convergence
        # This scheduler starts with low learning rate, increases to the max_lr,
        # then decreases again which helps with both faster convergence and generalization
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.3,  # Spend 30% of training time in the increasing phase
            div_factor=25,  # Start with lr/25
            final_div_factor=1000,  # End with lr/1000
            three_phase=False,  # Use two-phase schedule
            verbose=False,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update at each training step
            },
        }

    @classmethod
    def load_model(cls, filepath: Path, device: str = "cpu") -> "ConvVAE":
        """
        Load a model with its weights and hyperparameters

        Args:
            filepath: Path to the saved model
            device: Device to load the model on

        Returns:
            Loaded ConvVAE model
        """
        checkpoint = torch.load(filepath, map_location=device)

        # Handle different checkpoint formats
        if "hyper_parameters" in checkpoint:
            # PyTorch Lightning checkpoint format
            print("Using PyTorch Lightning checkpoint format")
            hyper_params = checkpoint["hyper_parameters"]
            print(f"Hyperparameters: {hyper_params}")

            # Create a model with the parameters from the checkpoint
            model = cls(
                input_channels=hyper_params.get("input_channels", 3),
                latent_dim=hyper_params.get("latent_dim", 128),
                hidden_dims=list(hyper_params.get("hidden_dims", [32, 64, 128, 256])),
                learning_rate=hyper_params.get("learning_rate", 1e-4),
                kl_weight=hyper_params.get("kl_weight", 1.0),
            )

            print(f"Created model with parameters: hidden_dims={model.hparams.hidden_dims}")

            # Check for parameter mismatch
            checkpoint_state_dict = checkpoint["state_dict"]

            # For checkpoints saved with model prefix
            if all(k.startswith("model.") for k in checkpoint_state_dict.keys()):
                state_dict = {k.replace("model.", ""): v for k, v in checkpoint_state_dict.items()}
            else:
                state_dict = checkpoint_state_dict

            # Handle case where model was saved with reversed hidden_dims
            # Try loading the state dict as is
            try:
                model.load_state_dict(state_dict)
                print("Successfully loaded state dict!")
            except RuntimeError as e:
                print(f"Error loading state dict, trying to match checkpoint weights...")
                # Try to match the checkpoint weights - analyze the weights to determine format
                # Get a reference shape from encoder weights in checkpoint
                encoder_weights = [k for k in state_dict.keys() if k.startswith("encoder.0.weight")]
                if encoder_weights:
                    sample_weight = state_dict[encoder_weights[0]]
                    sample_shape = sample_weight.shape
                    print(f"Sample encoder weight shape: {sample_shape}")

                    # Check if the hidden_dims in checkpoint match the state_dict
                    if sample_shape[0] == 32:  # First layer output channels is 32
                        expected_hidden_dims = [32, 64, 128, 256]
                        print(
                            f"Detected checkpoint was trained with hidden_dims={expected_hidden_dims}"
                        )

                        # Create a new model with the expected hidden_dims
                        model = cls(
                            input_channels=hyper_params.get("input_channels", 3),
                            latent_dim=hyper_params.get("latent_dim", 128),
                            hidden_dims=expected_hidden_dims,
                            learning_rate=hyper_params.get("learning_rate", 1e-4),
                            kl_weight=hyper_params.get("kl_weight", 1.0),
                        )

                        try:
                            model.load_state_dict(state_dict)
                            print("Successfully loaded state dict after adjusting hidden_dims!")
                        except Exception as e2:
                            print(f"Still failed to load state dict: {e2}")
                            print("Creating model with default parameters")
                            model = cls()
                    else:
                        print(f"Unexpected weight shape, creating model with default parameters")
                        model = cls()
                else:
                    print("Could not find encoder weights, creating model with default parameters")
                    model = cls()

        elif "state_dict" in checkpoint and "hparams" in checkpoint:
            # Our custom save format
            print("Using custom save format")
            model = cls(**checkpoint["hparams"])
            model.load_state_dict(checkpoint["state_dict"])
        else:
            # Assume it's just a state dict
            print("No recognized format, using default parameters")
            model = cls()  # Create with default parameters
            try:
                model.load_state_dict(checkpoint)
            except Exception as e:
                print(f"Failed to load state dict: {e}")

        model.eval()  # Set to evaluation mode
        return model

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run prediction on input data

        Args:
            x: Input tensor of shape [batch_size, channels, height, width] representing the image data.
                This should be the raw image tensor (normalized between 0-1), not the encoded latent representation.
                The method will handle the encoding and decoding process internally.

        Returns:
            Reconstructed images, mean and log variance
        """
        self.eval()  # Ensure model is in evaluation mode
        reconstructed, mu, log_var = self(x)
        return reconstructed, mu, log_var
