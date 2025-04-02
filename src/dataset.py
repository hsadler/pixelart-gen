from datasets import load_dataset
from torchvision import transforms as v2
import torch
from torch.utils.data import DataLoader
from PIL import Image


# Define a collate function to ensure proper tensor shape
def collate_fn(batch):
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

    def preprocess(examples):
        tensors = [transform(image) for image in examples["image"]]
        return {"tensor": torch.stack(tensors)}

    ds = ds.map(preprocess, batched=True)
    ds.set_format(type="torch", columns=["tensor"])
    
    # Create and return the DataLoader with our collate function
    return DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_fn,
        num_workers=4,
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
