from PIL import Image
from pathlib import Path
import logging

from fire import Fire
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback
import wandb
from datasets import Dataset, load_dataset

from src.vae import ConvAutoencoder_Simple, ConvVAE_Simple
from src.dataset import to_dataloaders_for_training, cleanup_dataloader
from src.utils import tensor_batch_to_pil_images, pil_image_concat, interpolate_latents

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# wandb config
WANDB_PROJECT: str = "pixelart-autoencoder-testing"
LOG_IMAGES_EVERY_N_STEPS: int = 100
# training data config
TRAIN_BATCH_SIZE: int = 128
VAL_BATCH_SIZE: int = 128
# model config
IMAGE_PIXEL_SIZE: int = 64
TRAIN_EPOCHS: int = 100
LATENT_DIM: int = 1024  # options: 64, 128, 256, 512, 1024
KL_WEIGHT: float = 0.0
LEARNING_RATE: float = 5e-4
# device config
DEVICE: str = "mps"
# checkpoint config
MODEL_CHECKPOINT: Path = Path("checkpoints/vae-best.ckpt")
GENERATE_SAMPLES_DIR: Path = Path("generated_samples")


class ImageLoggerCallback(Callback):
    def __init__(self, val_samples: torch.Tensor, every_n_epochs=10):
        super().__init__()
        self.val_samples = val_samples
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        # Only log every n epochs
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            # Get a batch from validation
            val_imgs = self.val_samples.to(device=pl_module.device)
            with torch.no_grad():
                recon_imgs, _, _ = pl_module(val_imgs)
            # Log sample input and reconstruction images side by side in a single grid
            imgs: list[Image.Image] = []
            for idx in range(len(val_imgs)):
                # Convert tensors to PIL images and append
                imgs.append(tensor_batch_to_pil_images(val_imgs)[idx])
                imgs.append(tensor_batch_to_pil_images(recon_imgs)[idx])
            pl_module.logger.experiment.log(
                {
                    "comparisons": wandb.Image(
                        pil_image_concat(imgs),
                        mode="RGB",
                        caption="Left: Original, Right: Reconstruction",
                    )
                }
            )
            interpolated = interpolate_latents(
                pl_module.encode,
                pl_module.decode,
                self.val_samples[0].to(pl_module.device),
                self.val_samples[1].to(pl_module.device),
                steps=20,
            )
            # Log sample input and reconstruction images side by side in a single grid
            pl_module.logger.experiment.log(
                {
                    "interpolations": wandb.Image(
                        pil_image_concat(tensor_batch_to_pil_images(interpolated)),
                    ),
                }
            )


def train():
    # Instantiate the logger
    wandb_logger: WandbLogger = WandbLogger(
        project=WANDB_PROJECT,
        log_model=False,
        save_dir="./logs",
    )
    # Load dataset splits
    train_dataset: Dataset = load_dataset("tkarr/sprite_caption_dataset", split="train")
    val_dataset: Dataset = load_dataset("tkarr/sprite_caption_dataset", split="valid")
    train_dl, val_dl = to_dataloaders_for_training(
        train_dataset,
        val_dataset,
        image_pixel_size=IMAGE_PIXEL_SIZE,
        batch_size=TRAIN_BATCH_SIZE,
    )
    # Instantiate the model
    model: ConvVAE_Simple = ConvVAE_Simple(
        image_pixel_size=IMAGE_PIXEL_SIZE,
        learning_rate=LEARNING_RATE,
        latent_dim=LATENT_DIM,
        kl_weight=KL_WEIGHT,
    )
    # Setup model checkpointing
    checkpoint_callback_best = ModelCheckpoint(
        dirpath="checkpoints",
        filename="vae-best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,  # Save the best model
        verbose=True,
        save_last=True,  # Also save the last model
    )
    # Add learning rate monitor
    lr_monitor: LearningRateMonitor = LearningRateMonitor(logging_interval="epoch")
    # Add early stopping
    early_stop_callback: EarlyStopping = EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=20,  # Wait for 20 epochs without improvement
        verbose=True,
        mode="min",
    )
    # Initialize validation samples for image logging
    # Get a small batch for visualization
    val_samples_for_logging = None
    for batch in val_dl:
        val_samples_for_logging = batch["tensor"][:10]
        break
    # Setup image logger callback
    image_logger = ImageLoggerCallback(
        val_samples_for_logging,
        every_n_epochs=LOG_IMAGES_EVERY_N_STEPS,
    )
    # Instantiate the trainer with callbacks
    trainer: Trainer = Trainer(
        max_epochs=TRAIN_EPOCHS,
        accelerator=DEVICE,
        devices="auto",
        logger=wandb_logger,
        log_every_n_steps=100,
        check_val_every_n_epoch=1,  # Only validate once per epoch
        limit_val_batches=1.0,  # Use at least 1 batch for validation
        callbacks=[
            checkpoint_callback_best,
            lr_monitor,
            early_stop_callback,
            image_logger,
        ],
        gradient_clip_val=1.0,  # Add gradient clipping to prevent exploding gradients
    )
    # Log configuration parameters to WandB
    config = {
        "model_class": model.__class__.__name__,
        "epochs": TRAIN_EPOCHS,
        "image_size": IMAGE_PIXEL_SIZE,
        "batch_size": TRAIN_BATCH_SIZE,
        "latent_dim": LATENT_DIM,
        "learning_rate": LEARNING_RATE,
        "kl_weight": KL_WEIGHT,
    }
    wandb_logger.experiment.config.update(config)
    # Train the model
    trainer.fit(model, train_dl, val_dl)


def train_overfit(num_samples: int = 10, epochs: int = 100):
    """
    Train the model on a small subset of the training data to verify it can overfit.
    This is a good sanity check to ensure the model has enough capacity to learn.

    Args:
        num_samples: Number of training samples to use
        epochs: Number of epochs to train for
    """
    logger.info(f"Overfitting on {num_samples} samples for {epochs} epochs")
    # Instantiate the logger
    wandb_logger: WandbLogger = WandbLogger(
        project="pixelart-autoencoder-testing",
        log_model=False,
        save_dir="./logs",
        tags=["vae-overfit-test"],
    )
    
    # Load dataset splits
    train_dataset: Dataset = load_dataset("tkarr/sprite_caption_dataset", split="train")
    val_dataset: Dataset = load_dataset("tkarr/sprite_caption_dataset", split="valid")
    full_train_dl, full_val_dl = to_dataloaders_for_training(
        train_dataset,
        val_dataset,
        image_pixel_size=IMAGE_PIXEL_SIZE,
        batch_size=num_samples,
    )
    # Get the first batch and create a fixed dataset from it
    for batch in full_train_dl:
        train_samples = batch
        break
    cleanup_dataloader(full_train_dl)
    cleanup_dataloader(full_val_dl)

    # Use this batch for both training and validation
    class FixedDataset(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples["tensor"])

        def __getitem__(self, idx):
            return {"tensor": self.samples["tensor"][idx]}
    fixed_dataset = FixedDataset(train_samples)
    train_dl = DataLoader(fixed_dataset, batch_size=num_samples, shuffle=True)
    val_dl = DataLoader(fixed_dataset, batch_size=num_samples, shuffle=False)
    
    # Instantiate model
    model = ConvVAE_Simple(
        image_pixel_size=IMAGE_PIXEL_SIZE,
        learning_rate=LEARNING_RATE,
        latent_dim=LATENT_DIM,
        kl_weight=0.0,
    )
    # Add learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    # Instantiate the trainer with callbacks
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        logger=wandb_logger,
        log_every_n_steps=10,
        callbacks=[lr_monitor],
        gradient_clip_val=1.0,
    )
    # Train the model
    trainer.fit(model, train_dl, val_dl)
    # Generate reconstructions of the training samples
    with torch.no_grad():
        train_tensors = train_samples["tensor"].to(model.device)
        # Get reconstructed images
        recon_tensors, _, _ = model.predict(train_tensors)
        # Save reconstructions
        for i in range(min(num_samples, 5)):  # Save up to 5 examples
            input_img = tensor_batch_to_pil_images(train_tensors[i : i + 1])[0]
            recon_img = tensor_batch_to_pil_images(recon_tensors[i : i + 1])[0]
            comparison = pil_image_concat([input_img, recon_img])
            GENERATE_SAMPLES_DIR.mkdir(exist_ok=True)
            comparison.save(GENERATE_SAMPLES_DIR / f"overfit_sample_{i}.png")
    
    # Clean up dataloaders
    cleanup_dataloader(train_dl)
    cleanup_dataloader(val_dl)


def predict_from_dataset_index(
    split: str, index: int, device: str = "cpu", deterministic: bool = True
) -> None:
    train_dataset: Dataset = load_dataset("tkarr/sprite_caption_dataset", split="train")
    val_dataset: Dataset = load_dataset("tkarr/sprite_caption_dataset", split="valid")
    train_dl, val_dl = to_dataloaders_for_training(
        train_dataset,
        val_dataset,
        image_pixel_size=IMAGE_PIXEL_SIZE,
        batch_size=1,
    )
    dl = train_dl if split == "train" else val_dl
    # Get the first batch from the data loader
    for batch in dl:
        input_tensor_batch: torch.Tensor = batch["tensor"]
        break
    input_tensor_batch = input_tensor_batch.to(device)
    logger.info(f"input_tensor_batch.shape: {input_tensor_batch.shape}")
    # Load the model
    model = ConvAutoencoder_Simple.load_model(MODEL_CHECKPOINT, device=device)
    if deterministic:
        # Use deterministic encoding and decoding for clearer images
        logger.info("Using deterministic encoding (mu only without sampling)")
        with torch.no_grad():
            # Get the mean representation without sampling
            mu, log_var = model.encode(input_tensor_batch)
            # Decode directly from mu without adding random noise
            reconstructed_tensor_batch = model.decode(mu)
    else:
        # Use the standard VAE sampling with reparameterization
        logger.info("Using stochastic sampling with reparameterization")
        reconstructed_tensor_batch = model.predict(input_tensor_batch)
    logger.info(f"reconstructed.shape: {reconstructed_tensor_batch.shape}")

    # Get latent representation for logging
    with torch.no_grad():
        mu, log_var = model.encode(input_tensor_batch)

    logger.info(f"mu.shape: {mu.shape}")
    logger.info(f"log_var.shape: {log_var.shape}")
    # Save the input and reconstructed images
    input_img: Image.Image = tensor_batch_to_pil_images(input_tensor_batch)[0]
    reconstructed_image: Image.Image = tensor_batch_to_pil_images(reconstructed_tensor_batch)[0]
    side_by_side_image = pil_image_concat([input_img, reconstructed_image])
    GENERATE_SAMPLES_DIR.mkdir(exist_ok=True)
    side_by_side_image.save(GENERATE_SAMPLES_DIR / f"output_{split}_{index}.png")
    cleanup_dataloader(dl)


if __name__ == "__main__":
    Fire(
        {
            "train": train,
            "train_overfit": train_overfit,
            "predict_from_dataset_index": predict_from_dataset_index,
        }
    )
