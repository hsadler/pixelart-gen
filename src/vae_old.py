from enum import Enum
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import DataLoader
import wandb

from src.dataset import load_and_preprocess_dataset
from src.utils import tensor_to_pil_image, pil_image_concat

class ModelType(Enum):
    SIMPLE = "simple"
    LOW_CAPACITY = "low_capacity"
    HIGH_CAPACITY = "high_capacity"
    VERY_HIGH_CAPACITY = "very_high_capacity"
    BATCH_NORM = "batch_norm"
    LEAKY_RELU = "leaky_relu"
    BATCH_NORM_LEAKY = "batch_norm_leaky"
    CONV = "conv"


def new_model(model_type: ModelType, image_pixel_size: int) -> torch.nn.Sequential:
    # simple model for testing
    simple_model: torch.nn.Sequential = torch.nn.Sequential(
        # Encoder
        torch.nn.Flatten(),
        torch.nn.Linear(3 * image_pixel_size * image_pixel_size, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 64),  # Latent representation
        torch.nn.ReLU(),
        # Decoder
        torch.nn.Linear(64, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 3 * image_pixel_size * image_pixel_size),
    )
    low_capacity_model: torch.nn.Sequential = torch.nn.Sequential(
        # Encoder
        torch.nn.Flatten(),
        torch.nn.Linear(3 * image_pixel_size * image_pixel_size, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),  # This creates the latent representation
        torch.nn.ReLU(),
        # Decoder
        torch.nn.Linear(128, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 3 * image_pixel_size * image_pixel_size),
    )
    high_capacity_model: torch.nn.Sequential = torch.nn.Sequential(
        # Encoder
        torch.nn.Flatten(),
        torch.nn.Linear(3 * image_pixel_size * image_pixel_size, 1024),  # Increased capacity
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 512),  # Added another layer
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),  # Latent representation
        torch.nn.ReLU(),
        # Decoder (mirror the encoder)
        torch.nn.Linear(128, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 3 * image_pixel_size * image_pixel_size),
    )
    very_high_capacity_model: torch.nn.Sequential = torch.nn.Sequential(
        # Encoder
        torch.nn.Flatten(),
        torch.nn.Linear(3 * image_pixel_size * image_pixel_size, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),  # Latent representation
        torch.nn.ReLU(),
        # Decoder (mirror the encoder)
        torch.nn.Linear(128, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 3 * image_pixel_size * image_pixel_size),
    )
    batch_norm_model: torch.nn.Sequential = torch.nn.Sequential(
        # Encoder
        torch.nn.Flatten(),
        torch.nn.Linear(3 * image_pixel_size * image_pixel_size, 1024),
        torch.nn.BatchNorm1d(1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 512),
        torch.nn.BatchNorm1d(512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.BatchNorm1d(256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),  # Latent representation
        torch.nn.BatchNorm1d(128),
        torch.nn.ReLU(),
        # Decoder (mirror the encoder)
        torch.nn.Linear(128, 256),
        torch.nn.BatchNorm1d(256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 512),
        torch.nn.BatchNorm1d(512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 1024),
        torch.nn.BatchNorm1d(1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 3 * image_pixel_size * image_pixel_size),
    )
    leaky_relu_model: torch.nn.Sequential = torch.nn.Sequential(
        # Encoder
        torch.nn.Flatten(),
        torch.nn.Linear(3 * image_pixel_size * image_pixel_size, 1024),
        torch.nn.LeakyReLU(0.2),
        torch.nn.Linear(1024, 512),
        torch.nn.LeakyReLU(0.2),
        torch.nn.Linear(512, 256),
        torch.nn.LeakyReLU(0.2),
        torch.nn.Linear(256, 128),  # Latent representation
        torch.nn.LeakyReLU(0.2),
        # Decoder (mirror the encoder)
        torch.nn.Linear(128, 256),
        torch.nn.LeakyReLU(0.2),
        torch.nn.Linear(256, 512),
        torch.nn.LeakyReLU(0.2),
        torch.nn.Linear(512, 1024),
        torch.nn.LeakyReLU(0.2),
        torch.nn.Linear(1024, 3 * image_pixel_size * image_pixel_size),
    )
    batch_norm_leaky_model: torch.nn.Sequential = torch.nn.Sequential(
        # Encoder
        torch.nn.Flatten(),
        torch.nn.Linear(3 * image_pixel_size * image_pixel_size, 1024),
        torch.nn.BatchNorm1d(1024),
        torch.nn.LeakyReLU(0.2),
        torch.nn.Linear(1024, 512),
        torch.nn.BatchNorm1d(512),
        torch.nn.LeakyReLU(0.2),
        torch.nn.Linear(512, 256),
        torch.nn.BatchNorm1d(256),
        torch.nn.LeakyReLU(0.2),
        torch.nn.Linear(256, 128),  # Latent representation
        torch.nn.BatchNorm1d(128),
        torch.nn.LeakyReLU(0.2),
        # Decoder (mirror the encoder)
        torch.nn.Linear(128, 256),
        torch.nn.BatchNorm1d(256),
        torch.nn.LeakyReLU(0.2),
        torch.nn.Linear(256, 512),
        torch.nn.BatchNorm1d(512),
        torch.nn.LeakyReLU(0.2),
        torch.nn.Linear(512, 1024),
        torch.nn.BatchNorm1d(1024),
        torch.nn.LeakyReLU(0.2),
        torch.nn.Linear(1024, 3 * image_pixel_size * image_pixel_size),
    )
    conv_model: torch.nn.Sequential = torch.nn.Sequential(
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
    return {
        ModelType.SIMPLE: simple_model,
        ModelType.HIGH_CAPACITY: high_capacity_model,
        ModelType.LOW_CAPACITY: low_capacity_model,
        ModelType.VERY_HIGH_CAPACITY: very_high_capacity_model,
        ModelType.BATCH_NORM: batch_norm_model,
        ModelType.LEAKY_RELU: leaky_relu_model,
        ModelType.BATCH_NORM_LEAKY: batch_norm_leaky_model,
        ModelType.CONV: conv_model,
    }[model_type]


def prepare_input(input_tensor: torch.Tensor, model_type: ModelType, device: str) -> torch.Tensor:
    """Format input tensor based on model type.

    Args:
        input_tensor: Input tensor of shape (batch_size, channels, height, width)
        model_type: Type of model being used

    Returns:
        Formatted input tensor ready for model
    """
    input_tensor: torch.Tensor = input_tensor.unsqueeze(0).to(device)
    if model_type == ModelType.CONV:
        # Conv model expects 4D tensor (batch_size, channels, height, width)
        return input_tensor
    else:
        # Other models expect flattened input
        return input_tensor.view(input_tensor.size(0), -1)


def prepare_output(
    output_tensor: torch.Tensor,
    model_type: ModelType,
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
    # Reshape based on model type
    if model_type == ModelType.CONV:
        # Conv model output needs to be reshaped from flattened to image dimensions
        outputs = output_tensor.view(output_tensor.size(0), 3, image_pixel_size, image_pixel_size)
    else:
        # Other models output is already in correct shape
        outputs = output_tensor

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
    model_type: ModelType,
    model_path: str,
    input_tensor: torch.Tensor,
    image_pixel_size: int,
    device: str,
) -> torch.Tensor:
    # Load model
    model: torch.nn.Sequential = new_model(model_type=model_type, image_pixel_size=image_pixel_size)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    # Predict
    input_tensor = prepare_input(input_tensor, model_type, device)
    with torch.no_grad():
        outputs: torch.Tensor = model(input_tensor)
        outputs = prepare_output(outputs, model_type, image_pixel_size, remove_batch=True)
    return outputs


def predict_from_dataset_index(
    model_type: ModelType,
    model_path: str,
    ds: torch.utils.data.Dataset,
    ds_index: int,
    image_pixel_size: int,
    device: str,
) -> torch.Tensor:
    input_tensor: torch.Tensor = ds[ds_index]["tensor"]
    output_tensor: torch.Tensor = predict(
        model_type=model_type,
        model_path=model_path,
        input_tensor=input_tensor,
        image_pixel_size=image_pixel_size,
        device=device,
    )
    return output_tensor

def train(
    model_type: str,
    loss_type: str,
    optimizer_type: str,
    learning_rate: float,
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
    model = new_model(model_type=ModelType(model_type), image_pixel_size=image_pixel_size)
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
