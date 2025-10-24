from torch import Tensor, is_tensor

class AdjustImageNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0}),
            }
        }

    NAME = "Adjust Image"
    RETURN_NAMES = ("Image",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "adjust_image"
    CATEGORY = "Met's Nodes"

    def adjust_image(self, image: Tensor, brightness=1.0, contrast=1.0, saturation=1.0):
        # Ensure image is float tensor
        if not is_tensor(image):
            raise ValueError("Input must be a torch.Tensor")
        
        # Brightness
        image = image * brightness
        
        # Contrast
        image = (image - 0.5) * contrast + 0.5
        
        # Saturation
        # image: [1, H, W, 3] -> [3, H, W]
        image = image[0].permute(2, 0, 1)
        lum = 0.299*image[0] + 0.587*image[1] + 0.114*image[2]  # H x W
        lum = lum.unsqueeze(0).repeat(3,1,1)
        image = lum + (image - lum) * saturation
        # back to [1, H, W, 3]
        image = image.permute(1, 2, 0).unsqueeze(0)

        # Clamp to valid range
        image = image.clamp(0.0, 1.0)
        
        return (image,)
