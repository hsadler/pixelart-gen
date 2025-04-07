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


def tensor_batch_to_pil_images(image_tensors: torch.Tensor) -> list[Image.Image]:
    if image_tensors.dim() != 4:
        raise ValueError(
            f"Expected 4D tensor with shape [batch_size, 3, height, width], "
            f"but got tensor with shape {image_tensors.shape}"
        )
    images = []
    for image_tensor in image_tensors:
        # Remove batch dimension if present and convert to PIL image
        if image_tensor.dim() > 3:
            image_tensor = image_tensor.squeeze(0)
        images.append(tensor_to_pil_image(image_tensor))
    return images


def pil_image_concat(images: list[Image.Image]) -> Image.Image:
    new_image = Image.new("RGB", (images[0].width * len(images), images[0].height))
    for i, img in enumerate(images):
        new_image.paste(img, (i * img.width, 0))
    return new_image


def interpolate_latents(
    vae_encoder: torch.nn.Module,
    vae_decoder: torch.nn.Module,
    img1: torch.Tensor,
    img2: torch.Tensor,
    steps=10,
) -> torch.Tensor:
    mu1, _ = vae_encoder(img1.unsqueeze(0))
    mu2, _ = vae_encoder(img2.unsqueeze(0))
    # Use the mean vectors for interpolation
    z1 = mu1
    z2 = mu2
    interpolated = []
    for alpha in torch.linspace(0, 1, steps):
        z = (1 - alpha) * z1 + alpha * z2
        img: torch.Tensor = vae_decoder(z)
        interpolated.append(img.squeeze())
    return torch.stack(interpolated)
