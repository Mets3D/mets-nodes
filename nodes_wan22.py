import math
import re
from typing import Any, TypeAlias

import torch.nn.functional as F
from torch import Tensor

import folder_paths
from nodes import CLIPLoader, CLIPTextEncode, KSamplerAdvanced, LoraLoader, UNETLoader, VAEDecode, VAELoader
from comfy_extras.nodes_wan import WanImageToVideo

_LORA_TAG_RE: re.Pattern[str] = re.compile(r'<lora:([^:>]+)(?::([-+]?[0-9]*\.?[0-9]+)(?::([-+]?[0-9]*\.?[0-9]+))?)?>')

_RESOLUTIONS: list[str] = ["High (1280x720 Pixel Count)", "Low (480x854 Pixel Count)"]
_PIXEL_COUNTS: dict[str, int] = {
    "High (1280x720 Pixel Count)": 1280 * 720,
    "Low (480x854 Pixel Count)":   480 * 854,
}

# ComfyUI MODEL is an opaque runtime type
_Model: TypeAlias = Any

# Our custom tuple types (matching WAN_DUAL_* ComfyUI type strings)
WanDualLoRA:     TypeAlias = tuple[str, str, str]          # (high_lora, low_lora, keyword)
WanDualModel:    TypeAlias = tuple[str, str]               # (high_model, low_model)
WanLoRAResolved: TypeAlias = tuple[str, str, float, float] # (high_lora, low_lora, high_weight, low_weight)


def _rescale_to_pixel_count(image: Tensor, resolution_mode: str) -> Tensor:
    """Scale image to match the target pixel count while preserving aspect ratio.
    Output dimensions are snapped to multiples of 8 for VAE compatibility."""
    _, orig_h, orig_w, _ = image.shape
    target_pixels = _PIXEL_COUNTS[resolution_mode]
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


def _load_unet(filename: str) -> _Model:
    """Load a UNet or GGUF diffusion model, routing by folder membership rather than file extension.
    Applies SageAttention if KJNodes is installed."""
    import nodes as _nodes

    if filename.endswith(".gguf"):
        print(f"[WAN Render] Loading as GGUF UNet: {filename}")
        gguf_loader_cls = _nodes.NODE_CLASS_MAPPINGS.get("UnetLoaderGGUF")
        if gguf_loader_cls is None:
            raise RuntimeError(
                f"'{filename}' is a UNet folder but the ComfyUI-GGUF custom node is not installed. "
                "Please install it from: https://github.com/city96/ComfyUI-GGUF"
            )
        model, = gguf_loader_cls().load_unet(unet_name=filename)
    else:
        print(f"[WAN Render] Loading as diffusion model: {filename}")
        model, = UNETLoader().load_unet(unet_name=filename, weight_dtype="default")

    sage_cls = _nodes.NODE_CLASS_MAPPINGS.get("PathchSageAttentionKJ")
    if sage_cls is not None:
        print("[WAN Render] KJNodes found — applying SageAttention patch")
        model, = sage_cls().patch(model=model, sage_attention="auto")

    return model


def _parse_lora_tags(
    prompt: str,
    dual_lora_data: list[WanDualLoRA] | None,
) -> tuple[str, list[WanLoRAResolved]]:
    """Extract <lora:keyword[:high_weight[:low_weight]]> tags from the prompt, resolve them
    against the LoRA stack, and return the stripped prompt alongside resolved
    (high_lora, low_lora, high_weight, low_weight) tuples. Raises ValueError for unknown keywords."""
    lora_lookup: dict[str, tuple[str, str]] = {keyword: (high, low) for high, low, keyword in (dual_lora_data or [])}
    resolved: list[WanLoRAResolved] = []
    for match in _LORA_TAG_RE.finditer(prompt):
        name = match.group(1)
        high_weight = float(match.group(2)) if match.group(2) is not None else 1.0
        low_weight  = float(match.group(3)) if match.group(3) is not None else high_weight
        if name not in lora_lookup:
            available = ', '.join(f'"{k}"' for k in lora_lookup) or 'none'
            raise ValueError(f"LoRA '{name}' not found in LoRA stack. Available: {available}")
        high_lora, low_lora = lora_lookup[name]
        resolved.append((high_lora, low_lora, high_weight, low_weight))
    clean_prompt = _LORA_TAG_RE.sub('', prompt).strip()
    return clean_prompt, resolved


class Wan22DualLoRA:
    NAME = "WAN 2.2 Dual LoRA"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "high_noise_lora": (loras, {"tooltip": "LoRA applied during the high-noise (early) sampling pass."}),
                "low_noise_lora": (loras, {"tooltip": "LoRA applied during the low-noise (late) sampling pass."}),
                "keyword": ("STRING", {"default": "", "tooltip": "Activation word used to reference this LoRA pair in prompts, e.g. <lora:keyword:1.0>."}),
            }
        }

    RETURN_TYPES = ("WAN_DUAL_LORA",)
    RETURN_NAMES = ("dual_lora",)
    OUTPUT_TOOLTIPS = ("Bundled LoRA pair and keyword, ready to be passed into a WAN 2.2 LoRA Stacker.",)
    FUNCTION = "combine"
    CATEGORY = "Met's Nodes/WAN 2.2"

    def combine(self, high_noise_lora: str, low_noise_lora: str, keyword: str) -> tuple[WanDualLoRA]:
        return ((high_noise_lora, low_noise_lora, keyword),)


class Wan22DualModel:
    NAME = "WAN 2.2 Dual Model"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        models = folder_paths.get_filename_list("diffusion_models") + folder_paths.get_filename_list("unet_gguf") + folder_paths.get_filename_list("unet")
        return {
            "required": {
                "high_noise_model": (models, {"tooltip": "Diffusion model used during the high-noise (early) sampling pass.",}),
                "low_noise_model": (models, {"tooltip": "Diffusion model used during the low-noise (late) sampling pass.",}),
            }
        }

    RETURN_TYPES = ("WAN_DUAL_MODEL",)
    RETURN_NAMES = ("dual_model",)
    OUTPUT_TOOLTIPS = ("Bundled model pair, ready to be passed into a WAN 2.2 Render node.",)
    FUNCTION = "combine"
    CATEGORY = "Met's Nodes/WAN 2.2"

    def combine(self, high_noise_model: str, low_noise_model: str) -> tuple[WanDualModel]:
        return ((high_noise_model, low_noise_model),)


class Wan22LoRAStacker:
    NAME = "WAN 2.2 LoRA Stacker"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {"required": {}, "optional": {}}

    RETURN_TYPES = ("WAN_DUAL_LORA_LIST",)
    RETURN_NAMES = ("dual_lora_data",)
    OUTPUT_TOOLTIPS = ("Ordered list of all connected Dual LoRA bundles, ready to be passed into a WAN 2.2 Render node.",)
    FUNCTION = "stack"
    CATEGORY = "Met's Nodes/WAN 2.2"

    def stack(self, **kwargs: WanDualLoRA) -> tuple[list[WanDualLoRA]]:
        loras = [v for _, v in sorted(kwargs.items(), key=lambda x: int(x[0].rsplit('_', 1)[-1])) if v is not None]
        return (loras,)


class Wan22Render:
    NAME = "WAN 2.2 Render"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "start_image": ("IMAGE", {
                    "tooltip": "Starting frame for the image-to-video generation. Will be rescaled to match the chosen resolution.",
                }),
                "dual_model": ("WAN_DUAL_MODEL", {
                    "tooltip": "High- and low-noise model pair from a WAN 2.2 Dual Model node.",
                }),
                "clip": (folder_paths.get_filename_list("text_encoders"), {
                    "default": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                    "tooltip": "WAN T5 text encoder used to encode the prompts.",
                }),
                "vae": (folder_paths.get_filename_list("vae"), {
                    "default": "wan_2.1_vae.safetensors",
                    "tooltip": "VAE used to encode the start image and decode the final latent.",
                }),

                "resolution": (_RESOLUTIONS, {
                    "default": _RESOLUTIONS[0],
                    "tooltip": "Target pixel count. The start image is rescaled to match this pixel budget while preserving aspect ratio.",
                }),
    
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Positive prompt. Supports <lora:keyword>, <lora:keyword:weight>, and <lora:keyword:high_weight:low_weight> syntax.",
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Negative prompt describing what to avoid in the generation.",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": "increment",
                    "tooltip": "Noise seed. Both sampling passes use the same seed.",
                }),


                "steps": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Total number of sampling steps across both passes.",
                }),
                "switch_factor": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Fraction of steps to run with the high-noise model before switching to the low-noise model. 0.5 = halfway.",
                }),
                "duration_seconds": ("FLOAT", {
                    "default": 5.0,
                    "min": 1,
                    "max": 60.0,
                    "step": 0.1,
                    "tooltip": "Video length in seconds. Frame count is snapped to the nearest value valid for WAN (multiples of 4 + 1).",
                }),
                "fps": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 120,
                    "tooltip": "Frames per second. Controls how many frames are generated and is passed through to downstream nodes.",
                }),
            },
            "optional": {
                "dual_lora_data": ("WAN_DUAL_LORA_LIST", {"tooltip": "Optional stack of Dual LoRA bundles. LoRAs are activated via <lora:keyword> tags in the prompt."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("frames", "fps")
    OUTPUT_TOOLTIPS = (
        "Decoded video frames as a batch of images.",
        "Frames per second, passed through from the input for use by downstream nodes.",
    )
    FUNCTION = "render"
    CATEGORY = "Met's Nodes/WAN 2.2"

    def render(
        self,
        dual_model:      WanDualModel,
        clip:            str,
        vae:             str,
        prompt:          str,
        negative_prompt: str,
        seed:            int,
        start_image:     Tensor,
        resolution:      str,
        steps:           int,
        switch_factor:   float,
        duration_seconds:        float,
        fps:             int,
        dual_lora_data:      list[WanDualLoRA] | None = None,
    ) -> tuple[Tensor, int]:
        start_image = _rescale_to_pixel_count(start_image, resolution)
        _, img_h, img_w, _ = start_image.shape
        print(f"[WAN Render] Resolution: {img_w}x{img_h}")

        print(f"[WAN Render] Loading CLIP: {clip}")
        loaded_clip, = CLIPLoader().load_clip(clip_name=clip, type="wan")
        print(f"[WAN Render] Loading VAE: {vae}")
        loaded_vae,  = VAELoader().load_vae(vae_name=vae)

        high_noise_filename, low_noise_filename = dual_model
        print(f"[WAN Render] Loading high-noise model: {high_noise_filename}")
        high_model = _load_unet(high_noise_filename)
        print(f"[WAN Render] Loading low-noise model: {low_noise_filename}")
        low_model  = _load_unet(low_noise_filename)

        clean_prompt, loras = _parse_lora_tags(prompt, dual_lora_data)
        if loras:
            print(f"[WAN Render] Applying {len(loras)} LoRA(s)")
            for high_lora_name, low_lora_name, high_weight, low_weight in loras:
                print(f"[WAN Render]   {high_lora_name} (high={high_weight}) / {low_lora_name} (low={low_weight})")
                high_model, loaded_clip = LoraLoader().load_lora(high_model, loaded_clip, high_lora_name, high_weight, high_weight)
                low_model,  _           = LoraLoader().load_lora(low_model,  loaded_clip, low_lora_name,  low_weight,  0.0)

        print("[WAN Render] Encoding prompts")
        positive_cond, = CLIPTextEncode().encode(clip=loaded_clip, text=clean_prompt)
        negative_cond, = CLIPTextEncode().encode(clip=loaded_clip, text=negative_prompt)

        frame_count = round(duration_seconds * fps)
        length = ((frame_count - 1) // 4) * 4 + 1
        print(f"[WAN Render] Video: {length} frames ({duration_seconds}s @ {fps}fps)")

        i2v_result = WanImageToVideo.execute(
            positive=positive_cond,
            negative=negative_cond,
            vae=loaded_vae,
            width=int(img_w),
            height=int(img_h),
            length=length,
            batch_size=1,
            start_image=start_image,
        )
        positive_cond, negative_cond, latent = i2v_result.args

        switch_step = round(steps * switch_factor)
        print(f"[WAN Render] Sampling: steps={steps}, switch at step {switch_step} (factor={switch_factor}), seed={seed}")

        print(f"[WAN Render] High-noise pass: steps 0-{switch_step}")
        latent, = KSamplerAdvanced().sample(
            model=high_model,
            add_noise="enable",
            noise_seed=seed,
            steps=steps,
            cfg=1.0,
            sampler_name="uni_pc",
            scheduler="simple",
            positive=positive_cond,
            negative=negative_cond,
            latent_image=latent,
            start_at_step=0,
            end_at_step=switch_step,
            return_with_leftover_noise="enable",
        )

        print(f"[WAN Render] Low-noise pass: steps {switch_step}-{steps}")
        latent, = KSamplerAdvanced().sample(
            model=low_model,
            add_noise="disable",
            noise_seed=seed,
            steps=steps,
            cfg=1.0,
            sampler_name="uni_pc",
            scheduler="simple",
            positive=positive_cond,
            negative=negative_cond,
            latent_image=latent,
            start_at_step=switch_step,
            end_at_step=steps,
            return_with_leftover_noise="disable",
        )

        print(f"[WAN Render] Decoding {length} frames")
        frames, = VAEDecode().decode(vae=loaded_vae, samples=latent)
        print(f"[WAN Render] Done. Output shape: {frames.shape}")
        return (frames, fps)
