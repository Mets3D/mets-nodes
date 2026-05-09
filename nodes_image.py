import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import Tensor, is_tensor
from nodes import PreviewImage

import asyncio

_RESOLUTIONS: list[str] = ["High (1280x720 Pixel Count)", "Low (480x854 Pixel Count)"]
_PIXEL_COUNTS: dict[str, int] = {
    "High (1280x720 Pixel Count)": 1280 * 720,
    "Low (480x854 Pixel Count)":   480 * 854,
}


def _rescale_to_pixel_count(image: Tensor, resolution: str) -> Tensor:
    """Scale image to match the target pixel count while preserving aspect ratio.
    Output dimensions are snapped to multiples of 8 for VAE compatibility."""
    _, orig_h, orig_w, _ = image.shape
    target_pixels = _PIXEL_COUNTS[resolution]
    scale = math.sqrt(target_pixels / (orig_w * orig_h))
    target_w = (round(orig_w * scale) // 8) * 8
    target_h = (round(orig_h * scale) // 8) * 8
    if target_w == orig_w and target_h == orig_h:
        return image
    return F.interpolate(
        image.permute(0, 3, 1, 2),
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=False,
    ).permute(0, 2, 3, 1)


class RescaleToPixelCount:
    NAME = "Rescale to Pixel Count"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to rescale."}),
                "resolution": (_RESOLUTIONS, {
                    "default": _RESOLUTIONS[0],
                    "tooltip": "Target pixel count. The image is rescaled to match this pixel budget while preserving aspect ratio.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "rescale"
    CATEGORY = "Met's Nodes/Image"

    def rescale(self, image: Tensor, resolution: str) -> tuple[Tensor]:
        return (_rescale_to_pixel_count(image, resolution),)

class AdjustImageNode(PreviewImage):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0}),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    NAME = "Adjust Image"
    RETURN_NAMES = ("Image",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "adjust_image"
    CATEGORY = "Met's Nodes"

    def adjust_image(self, image: Tensor, brightness=1.0, contrast=1.0, saturation=1.0, prompt=None, extra_pnginfo=None):
        # Ensure image is float tensor
        if not is_tensor(image):
            raise ValueError("Input must be a torch.Tensor")

        new_image = image.clone()

        # Brightness
        if brightness != 1.0:
            new_image = new_image * brightness

        # Contrast
        if contrast != 1.0:
            new_image = (new_image - 0.5) * contrast + 0.5
        
        # Saturation
        if saturation != 1.0:
            # image: [1, H, W, 3] -> [3, H, W]
            new_image = new_image[0].permute(2, 0, 1)
            lum = 0.299*new_image[0] + 0.587*new_image[1] + 0.114*new_image[2]  # H x W
            lum = lum.unsqueeze(0).repeat(3,1,1)
            new_image = lum + (new_image - lum) * saturation
            # back to [1, H, W, 3]
            new_image = new_image.permute(1, 2, 0).unsqueeze(0)

        # Clamp to valid range
        new_image = new_image.clamp(0.0, 1.0)

        # Get data for preview
        res = super().save_images(new_image, filename_prefix="AdjustImage-", prompt=prompt, extra_pnginfo=extra_pnginfo)
        ui_image = res['ui']['images']
        
        # Return preview data + node outputs
        return {
            "ui": {"images": ui_image},
            "result": (new_image,),
        }


_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"})


class LoadImageFromDirectory:
    NAME = "Load Image From Directory"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "directory": ("STRING", {
                    "default": "",
                    "tooltip": "Path to the folder containing images.",
                }),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": "increment",
                    "tooltip": "Alphabetical index of the image to load. Increments automatically after each generation.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "STRING", "INT")
    RETURN_NAMES = ("image", "filename", "total", "directory", "index")
    OUTPUT_TOOLTIPS = (
        "The loaded image.",
        "Filename of the loaded image (without directory path).",
        "Total number of images found in the directory.",
        "The directory path, passed through from the input.",
        "The index, passed through from the input.",
    )
    FUNCTION = "load"
    CATEGORY = "Met's Nodes/Batch Video"

    def load(self, directory: str, index: int) -> tuple:
        path = Path(directory)
        if not path.is_dir():
            raise ValueError(f"Directory not found: {directory}")

        files = sorted(f for f in path.iterdir() if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS)
        if not files:
            raise ValueError(f"No images found in: {directory}")

        file = files[index % len(files)]
        tensor = torch.from_numpy(np.array(ImageOps.exif_transpose(Image.open(file)).convert("RGB")).astype(np.float32) / 255.0).unsqueeze(0)
        return (tensor, file.name, len(files), directory, index)


def _register_routes() -> None:
    try:
        from aiohttp import web
        from server import PromptServer

        @PromptServer.instance.routes.get("/mets/preview_image")
        async def preview_image(request: web.Request) -> web.StreamResponse:
            directory = request.query.get("directory", "")
            try:
                index = int(request.query.get("index", "0"))
            except ValueError:
                return web.Response(status=400)
            path = Path(directory)
            if not path.is_dir():
                return web.Response(status=404)
            files = sorted(f for f in path.iterdir() if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS)
            if not files:
                return web.Response(status=404)
            import mimetypes
            file = files[index % len(files)]
            mime = mimetypes.guess_type(str(file))[0] or "application/octet-stream"
            return web.Response(
                body=file.read_bytes(),
                content_type=mime,
                headers={"X-Filename": file.name, "X-Total": str(len(files))},
            )

        @PromptServer.instance.routes.get("/mets/browse_directory")
        async def browse_directory(_request: web.Request) -> web.Response:
            def _pick() -> str:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                path = filedialog.askdirectory(parent=root)
                root.destroy()
                return path or ""

            path = await asyncio.get_running_loop().run_in_executor(None, _pick)
            return web.json_response({"path": path})

    except Exception:
        pass


_register_routes()
