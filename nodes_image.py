from torch import Tensor, is_tensor
from nodes import PreviewImage

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
