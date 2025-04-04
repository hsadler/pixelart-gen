from PIL import Image

from datasets import load_dataset
from torchvision import transforms as v2
import torch
from torch.utils.data import DataLoader


# Define a collate function to ensure proper tensor shape
def _collate_fn(batch):
    tensors = [item["tensor"] for item in batch]
    # If tensors are already batched from preprocessing (N, C, H, W)
    if tensors[0].dim() == 4:
        # Concatenate the first dimension
        return {"tensor": torch.cat(tensors, dim=0)}
    # If tensors are individual images (C, H, W)
    elif tensors[0].dim() == 3:
        # Stack them to form a batch
        return {"tensor": torch.stack(tensors, dim=0)}
    else:
        raise ValueError(f"Unexpected tensor dimension: {tensors[0].dim()}")


def load_and_preprocess_dataset(
    path: str,
    split: str,
    image_pixel_size: int,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    ds = load_dataset(path, split=split)
    transform = v2.Compose(
        [
            v2.Lambda(lambda x: x.convert("RGB")),
            # TODO maybe convert to HSL?
            # v2.Lambda(lambda x: x.convert("HSL")),
            v2.Lambda(lambda x: _scale_image_by_pixel_size(x, image_pixel_size)),
            v2.CenterCrop((image_pixel_size, image_pixel_size)),
            v2.ToTensor(),
        ]
    )

    # Define a preprocess function to convert images to tensors
    def preprocess(examples):
        tensors = [transform(image) for image in examples["image"]]
        # Debug print to check tensor values
        print(f"Tensor min: {tensors[0].min()}, max: {tensors[0].max()}, mean: {tensors[0].mean()}")
        return {"tensor": torch.stack(tensors)}

    ds = ds.map(preprocess, batched=True)
    ds.set_format(type="torch", columns=["tensor"])

    # Create and return the DataLoader with our collate function
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_fn,
        num_workers=9,  # for multiprocessing
        persistent_workers=True,  # Keep workers alive between batches
    )


def _scale_image_by_pixel_size(image: Image.Image, pixel_size: int) -> Image.Image:
    original_width, original_height = image.size
    if original_width <= original_height:
        # Scale by width
        scale_factor = original_width / pixel_size
        new_width = pixel_size
        new_height = int(original_height / scale_factor)
    else:
        # Scale by height
        scale_factor = original_height / pixel_size
        new_height = pixel_size
        new_width = int(original_width / scale_factor)
    scaled_image = image.resize((new_width, new_height), resample=Image.Resampling.NEAREST)
    return scaled_image


def cleanup_dataloader(dataloader: DataLoader) -> None:
    """
    Properly clean up resources used by a DataLoader with multiple workers.

    Args:
        dataloader: The DataLoader to clean up
    """
    # Close the dataloader
    dataloader._iterator = None

    # If there are workers, clean them up
    if hasattr(dataloader, "_worker_pids"):
        # Kill the worker processes
        for worker_id in dataloader._worker_pids.keys():
            if dataloader._worker_pids[worker_id] is not None:
                dataloader._workers[worker_id].terminate()

        # Clear the worker references
        dataloader._workers = {}
        dataloader._worker_pids = {}

    # Force garbage collection
    import gc

    gc.collect()

    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
