import math
import re

import torch.nn.functional as F

import folder_paths
from nodes import CLIPLoader, CLIPTextEncode, KSamplerAdvanced, LoraLoader, UNETLoader, VAEDecode, VAELoader
from comfy_extras.nodes_wan import WanImageToVideo

_LORA_TAG_RE = re.compile(r'<lora:([^:>]+):([-+]?[0-9]*\.?[0-9]+)(?::([-+]?[0-9]*\.?[0-9]+))?>')

_RESOLUTIONS = ["High (1280x720 Pixel Count)", "Low (480x854 Pixel Count)"]
_PIXEL_COUNTS = {
    "High (1280x720 Pixel Count)": 1280 * 720,
    "Low (480x854 Pixel Count)":   480 * 854,
}


def _rescale_to_pixel_count(image, resolution_mode):
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


_UNET_FILENAMES = set(folder_paths.get_filename_list("unet") + folder_paths.get_filename_list("unet_gguf"))


def _load_unet(filename):
    import nodes as _nodes

    if filename in _UNET_FILENAMES:
        print(f"[WAN Render] Loading as GGUF UNet: {filename}")
        gguf_loader_cls = _nodes.NODE_CLASS_MAPPINGS.get("UnetLoaderGGUF")
        if gguf_loader_cls is None:
            raise RuntimeError(
                f"'{filename}' is from a UNet folder but the ComfyUI-GGUF custom node is not installed. "
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


def _parse_lora_tags(prompt, lora_stack):
    lora_lookup = {keyword: (high, low) for high, low, keyword in (lora_stack or [])}
    resolved = []
    for match in _LORA_TAG_RE.finditer(prompt):
        name = match.group(1)
        high_weight = float(match.group(2))
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
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "high_noise_lora": (loras,),
                "low_noise_lora": (loras,),
                "keyword": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("WAN_DUAL_LORA",)
    RETURN_NAMES = ("dual_lora",)
    FUNCTION = "combine"
    CATEGORY = "Met's Nodes/WAN 2.2"

    def combine(self, high_noise_lora, low_noise_lora, keyword):
        return ((high_noise_lora, low_noise_lora, keyword),)


class Wan22DualModel:
    NAME = "WAN 2.2 Dual Model"

    @classmethod
    def INPUT_TYPES(cls):
        models = folder_paths.get_filename_list("diffusion_models") + folder_paths.get_filename_list("unet_gguf") + folder_paths.get_filename_list("unet")
        return {
            "required": {
                "high_noise_model": (models,),
                "low_noise_model":  (models,),
            }
        }

    RETURN_TYPES = ("WAN_DUAL_MODEL",)
    RETURN_NAMES = ("dual_model",)
    FUNCTION = "combine"
    CATEGORY = "Met's Nodes/WAN 2.2"

    def combine(self, high_noise_model, low_noise_model):
        return ((high_noise_model, low_noise_model),)


class Wan22LoRAStacker:
    NAME = "WAN 2.2 LoRA Stacker"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {}}

    RETURN_TYPES = ("WAN_DUAL_LORA_LIST",)
    RETURN_NAMES = ("dual_loras",)
    FUNCTION = "stack"
    CATEGORY = "Met's Nodes/WAN 2.2"

    def stack(self, **kwargs):
        loras = [v for _, v in sorted(kwargs.items(), key=lambda x: int(x[0].rsplit('_', 1)[-1])) if v is not None]
        return (loras,)


class Wan22Render:
    NAME = "WAN 2.2 Render"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dual_model":    ("WAN_DUAL_MODEL",),
                "clip":          (folder_paths.get_filename_list("text_encoders"),),
                "vae":           (folder_paths.get_filename_list("vae"),),
                "prompt":          ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed":          ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "start_image":   ("IMAGE",),
                "resolution":    (_RESOLUTIONS, {"default": _RESOLUTIONS[0]}),
                "steps":         ("INT", {"default": 4, "min": 1, "max": 100}),
                "switch_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "duration":      ("FLOAT", {"default": 5.0, "min": 1, "max": 60.0, "step": 0.1}),
                "fps":           ("INT", {"default": 16, "min": 1, "max": 120}),
            },
            "optional": {
                "lora_stack": ("WAN_DUAL_LORA_LIST",),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("frames", "fps")
    FUNCTION = "render"
    CATEGORY = "Met's Nodes/WAN 2.2"

    def render(self, dual_model, clip, vae, prompt, negative_prompt, seed, start_image, resolution, steps, switch_factor, duration, fps, lora_stack=None):
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

        clean_prompt, loras = _parse_lora_tags(prompt, lora_stack)
        if loras:
            print(f"[WAN Render] Applying {len(loras)} LoRA(s)")
            for high_lora_name, low_lora_name, high_weight, low_weight in loras:
                print(f"[WAN Render]   {high_lora_name} (high={high_weight}) / {low_lora_name} (low={low_weight})")
                high_model, loaded_clip = LoraLoader().load_lora(high_model, loaded_clip, high_lora_name, high_weight, high_weight)
                low_model,  _           = LoraLoader().load_lora(low_model,  loaded_clip, low_lora_name,  low_weight,  0.0)

        print("[WAN Render] Encoding prompts")
        positive_cond, = CLIPTextEncode().encode(clip=loaded_clip, text=clean_prompt)
        negative_cond, = CLIPTextEncode().encode(clip=loaded_clip, text=negative_prompt)

        frame_count = round(duration * fps)
        length = ((frame_count - 1) // 4) * 4 + 1
        print(f"[WAN Render] Video: {length} frames ({duration}s @ {fps}fps)")

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

