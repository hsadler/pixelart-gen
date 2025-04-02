from fire import Fire
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.vae import ConvVAE1
from src.dataset import load_and_preprocess_dataset


def train():
    wandb_logger = WandbLogger(
        project="pixelart-autoencoder",
        name="vae-experiment-test",
        log_model=False,  # Logs model checkpoints
        save_dir="./logs",  # Where to store logs locally
        tags=["vae-test"],  # Add custom tags for organizing runs
    )
    # Load your dataset
    train_dl: DataLoader = load_and_preprocess_dataset(
        path="tkarr/sprite_caption_dataset",
        split="train",
        image_pixel_size=64,
    )
    val_dl: DataLoader = load_and_preprocess_dataset(
        path="tkarr/sprite_caption_dataset",
        split="valid",
        image_pixel_size=64,
    )
    # Create the model
    model = ConvVAE1(
        input_channels=3,
        latent_dim=128,
        hidden_dims=[32, 64, 128, 256],
        learning_rate=1e-4,
    )
    # Create the trainer
    trainer = Trainer(
        max_epochs=3,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=100,
        logger=wandb_logger,
    )
    # Train the model
    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    Fire(
        {
            "train": train,
        }
    )
