from fire import Fire
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.vae import ConvVAE
from src.dataset import load_and_preprocess_dataset


def train():
    # Instantiate the logger
    wandb_logger = WandbLogger(
        # project="pixelart-autoencoder",
        project="pixelart-autoencoder-testing",
        log_model=False,
        save_dir="./logs",  # Where to store logs locally
        tags=["vae-test"],  # Add custom tags for organizing runs
    )
    # Define batch size and other hyperparameters
    batch_size = 64
    # Load your dataset
    train_dl = load_and_preprocess_dataset(
        path="tkarr/sprite_caption_dataset",
        split="train",
        image_pixel_size=64,
        batch_size=batch_size,
        shuffle=True,
    )
    val_dl = load_and_preprocess_dataset(
        path="tkarr/sprite_caption_dataset",
        split="valid",
        image_pixel_size=64,
        batch_size=batch_size,
        shuffle=False,
    )    
    # Instantiate the model
    model = ConvVAE(
        input_channels=3,
        latent_dim=128,
        hidden_dims=[32, 64, 128, 256],
        # hidden_dims=[16, 32, 64],
        learning_rate=0.0001,
    )
    # Instantiate the trainer
    trainer = Trainer(
        max_epochs=100,
        accelerator="auto",
        devices="auto",
        logger=wandb_logger,
        log_every_n_steps=100,
        check_val_every_n_epoch=1,  # Only validate once per epoch
        # limit_train_batches=0.5,  # Use only half of training data per epoch
        # limit_val_batches=0.3,  # Use only 30% of validation data
    )
    # Train the model
    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    Fire(
        {
            "train": train,
        }
    )
