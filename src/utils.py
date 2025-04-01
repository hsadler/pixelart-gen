from PIL import Image
import torch
from torchvision.transforms import ToPILImage


def tensor_to_pil_image(image_tensor: torch.Tensor) -> Image.Image:
    if not (image_tensor.dim() == 3 and image_tensor.shape[0] == 3):
        raise ValueError(
            f"Tensor is not a valid image tensor. Expected 3D tensor with shape [3, height, width], "
            f"but got tensor with shape {image_tensor.shape} and {image_tensor.dim()} dimensions"
        )
    # Ensure values are in [0, 1] range before conversion
    image_tensor = image_tensor.clamp(0, 1)
    return ToPILImage(mode="RGB")(image_tensor)


def pil_image_concat(images: list[Image.Image]) -> Image.Image:
    new_image = Image.new("RGB", (images[0].width * len(images), images[0].height))
    for i, img in enumerate(images):
        new_image.paste(img, (i * img.width, 0))
    return new_image
