import re
from pathlib import Path

import comfy
import folder_paths
from nodes import CheckpointLoaderSimple, KSampler, CLIPTextEncode, VAEEncode, VAEDecode, EmptyImage, LatentBlend, EmptyLatentImage, ImageScaleBy, NODE_CLASS_MAPPINGS

from .met_context import LoRA_Config, MetCheckpointPreset
from .mega_prompt import (
    unroll_tag_stack, move_neg_tags, override_width_height, apply_checkpoint_prompt, extract_lora_tags, reorder_prompt, 
    remove_excluded_tags, remove_all_tag_syntax, extract_face_prompts
)
from .regex_nodes import randomize_prompt, tidy_prompt
from .model_downloader import DownloadCivitaiModel

# TODO: 
# - Rename Metxyz classes to have better names, eg. MetCheckpointPreset->CheckpointConfig, PrepareCheckpoint->ConfigureCheckpoint
# - Rename their "identifier" input to "alias", although I really don't want to recreate all of them in my workflow (could just search and replace in the .json though...)
# - Add prompt processing to RenderPass node

# Remove a bunch of old technology, like ChainReplace, MegaPrompt, etc.

RE_LORA_TAGS = re.compile(r"<lora:.*?>")

class RenderPass:
    NAME = "Render Pass"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "data": ("RENDER_PASS_DATA", {"tooltip": "Various data collected, to be used by a render pass."}),
                "checkpoint_name": (folder_paths.get_filename_list("checkpoints"),),
                "noise": ("FLOAT", {"tooltip": "Amount of noise to add to the image before starting sampling", "default": 1.0, "min": 0.01, "max": 1.0}),
                "image": ("IMAGE", {"tooltip": "Starting image. Even a black image is useful, to specify the render resolution, and to help generate a darker image if noise is less than 1.0"}),
                "scale": ("FLOAT", {"tooltip": "Amount to scale input image by before adding noise and re-sampling", "default": 1.0, "min":0.01, "max": 10}),
                "prompt_pos": ("STRING", {"multiline": False, "tooltip": "The base positive prompt."}),
                "prompt_neg": ("STRING", {"multiline": False, "tooltip": "The base negative prompt."}),
                "is_prompt_additive": ("BOOLEAN", {"tooltip": "Whether this prompt should be added onto any prompt information already contained in the data input. If not, it will replace the prompt instead.", "default": True}),
            },
        }

    RETURN_NAMES = ("Data", "Image", "Prompt_Pos", "Prompt_Neg")
    RETURN_TYPES = ("RENDER_PASS_DATA","IMAGE", "STRING", "STRING")
    FUNCTION = "render_pass_execute"
    CATEGORY = "MetsNodes"
    DESCRIPTION="""Render an image with the data provided."""
    OUTPUT_NODE = True

    def render_pass_execute(self, data, checkpoint_name, noise, image=None, scale=1.0, prompt_pos="", prompt_neg="", is_prompt_additive=True):
        # We want to make a copy of the passed data, otherwise Comfy might send us back 
        # the dict that we modified in a previous run (eg. if it was a failed run)
        data = data.copy()

        tag_stack = data.get("tag_stack", {})
        checkpoint_datas = data.get("checkpoint_datas", {})
        prompt_seed = data.get("prompt_seed", 1)
        noise_seed = data.get("noise_seed", 1)
        pass_index = data.get("pass_index", 0) + 1
        data['pass_index'] = pass_index

        ckpt_config: MetCheckpointPreset|None = next((cp for identifier, cp in checkpoint_datas.items() if cp.path==checkpoint_name), None)
        if not ckpt_config:
             raise Exception(f"Checkpoint config not found for: {checkpoint_name}\nYou need to provide it by plugging a Prepare Checkpoint node into a Prepare Render Pass Node, and then plugging that into this node.")

        prompt_pos = unroll_tag_stack(prompt_pos, tag_stack)
        prompt_neg = unroll_tag_stack(prompt_neg, tag_stack)

        # Randomize using {blue|red|green} syntax.
        # NOTE: Currently not done for the negative prompt, I don't think it would be useful.
        prompt_pos = randomize_prompt(prompt_pos, prompt_seed)

        if image==None:
            image = data.get('last_image', None)
        if image==None:
            image = EmptyImage().generate(1024, 1024)[0]

        # Apply aspect ratio override.
        real_width, real_height = image.shape[2], image.shape[1]
        width, height, prompt_pos = override_width_height(prompt_pos, real_width, real_height)
        if width != real_width:
            image = image.permute(0, 2, 1, 3).contiguous()

        if image == None:
            image = EmptyImage().generate(1024, 1024)[0]

        prompt_pos, face_pos = extract_face_prompts(prompt_pos)
        prompt_neg, face_neg = extract_face_prompts(prompt_neg)

        if is_prompt_additive and pass_index != 1:
            prev_pos, prev_neg = data.get("prompt_pos", ""), data.get("prompt_neg", "")
            prev_face_pos, prev_face_neg = data.get("prompt_face_pos", ""), data.get("prompt_face_neg", "")
            if prev_pos:
                prompt_pos += ",\n"+prev_pos
                face_pos += ",\n"+prev_face_pos
            if prev_neg:
                prompt_neg += ",\n"+prev_neg
                face_neg += ",\n"+prev_face_neg

        # Store the prompt in its current state of processing to be sent on to subsequent render passes.
        # NOTE: We want to do this before removing anything from the prompts, 
        # and before adding the checkpoint's quality prompt,
        # since each render pass will just remove whatever it wants to remove.
        # However, we want to do it AFTER the prompt has been unrolled and randomized, so subsequent render passes 
        # don't get totally unrelated prompts.
        data["prompt_pos"] = prompt_pos
        data["prompt_neg"] = prompt_neg
        data["prompt_face_pos"] = face_pos
        data["prompt_face_neg"] = face_neg

        # Apply the checkpoint's associated +/- prompts.
        prompt_pos, prompt_neg = apply_checkpoint_prompt(prompt_pos, prompt_neg, ckpt_config)

        # Remove contents of tags which are marked for removal using exclamation mark syntax: <!tag>
        # Useful when a prompt wants to signify that it's not compatible with something.
        # Eg., to easily prompt a blink, mark descriptions of <eye>eyes</eye>, then use "blink <!eye>" in prompt.
        # NOTE: This must come AFTER apply_checkpoint_prompt(), otherwise it will remove the <!modelprompt> 
        # keyword before it has a chance to trigger.
        prompt_pos = remove_excluded_tags(prompt_pos)

        # Move contents of <neg> tags to negative prompt, 
        # and remove exact matches of negative prompt words from the positive prompt.
        prompt_pos, prompt_neg = move_neg_tags(prompt_pos, prompt_neg)

        # Move contents of <!> tags to beginning of prompt.
        prompt_pos = reorder_prompt(prompt_pos)

        prompt_pos = remove_all_tag_syntax(tidy_prompt(prompt_pos))
        prompt_neg = remove_all_tag_syntax(tidy_prompt(prompt_neg))

        # Scale image, if requested.
        if scale != 1.0:
            upscale_method = 'area' if scale < 1.0 else 'lanczos'
            image = ImageScaleBy().upscale(image, upscale_method, scale)[0]

        api_token = data.get("civitai_api_key", "")
        lora_data = data.get("lora_datas", {})

        prompt_pos, lora_data = ensure_required_loras(prompt_pos, lora_data, api_token)

        model, vae, clip, final_image = render(ckpt_config, prompt_pos, prompt_neg, image, noise_seed, noise, pass_index=pass_index, lora_data=lora_data)
        data['last_image'] = final_image
        data['last_model'] = model
        data['last_vae'] = vae
        data['last_clip'] = clip
        data['last_checkpoint_config'] = ckpt_config
        return (data, final_image, prompt_pos, prompt_neg)

class RenderPass_Face:
    NAME = "Face Render Pass"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "data": ("RENDER_PASS_DATA", {"tooltip": "Various data collected, to be used by a render pass."}),
                "image": ("IMAGE", {"tooltip": "Starting image. Not really necessary since the data will also pass along the image, but useful for when you want to bypass this node."}),
                "noise": ("FLOAT", {"tooltip": "Amount of noise to add to the image before starting sampling", "default": 0.32, "min": 0.01, "max": 1.0}),
                "iterations": ("INT", {"tooltip": "Number of times to execute", "default": 1, "min": 1, "max": 5}),
                "prompt_pos": ("STRING", {"multiline": False, "tooltip": "Positive prompt for the faces."}),
                "prompt_neg": ("STRING", {"multiline": False, "tooltip": "Negative prompt for the faces."}),
            },
        }

    RETURN_NAMES = ("Data", "Image")
    RETURN_TYPES = ("RENDER_PASS_DATA","IMAGE")
    FUNCTION = "face_pass_execute"
    CATEGORY = "MetsNodes"
    DESCRIPTION="""Simplified wrapper for the Impact Pack's FaceDetailer node."""
    OUTPUT_NODE = True

    def face_pass_execute(self, data, image, noise, iterations, prompt_pos, prompt_neg):
        try:
            from impact.impact_pack import FaceDetailer
        except ModuleNotFoundError:
            raise Exception("FaceDetailer node must be installed (from Impact Pack).")
        model = data.get('last_model', None)
        vae = data.get('last_vae', None)
        clip = data.get('last_clip', None)
        image = image if image!=None else data.get('last_image', None)
        if model==None or vae==None or clip==None or image==None:
            raise Exception("Missing data for Face Pass node.\nYou must plug in a Render Pass node's data output into this node's data input.")
        ckpt_config: MetCheckpointPreset|None = data.get('last_checkpoint_config', None)
        if not ckpt_config:
            raise Exception("Checkpoint config not provided.\nYou must plug in a Render Pass node's data output into this node's data input.")
        if 'UltralyticsDetectorProvider' not in NODE_CLASS_MAPPINGS:
            raise Exception(f"UltralyticsDetectorProvider node must be installed (from Impact Subpack)")
        DetectorProvider = NODE_CLASS_MAPPINGS['UltralyticsDetectorProvider']
        bbox_detector, segm_detector = DetectorProvider().doit('bbox/face_yolov8m.pt')

        seed = data.get("noise_seed", 0)+10
        steps = ckpt_config.steps
        cfg = ckpt_config.cfg
        sampler_name = ckpt_config.sampler
        scheduler = ckpt_config.scheduler

        prompt_pos += ", "+data.get("prompt_face_pos", "")
        prompt_neg += ", "+data.get("prompt_face_neg", "")

        clip_encoder = CLIPTextEncode()
        positive = clip_encoder.encode(clip, prompt_pos)[0]
        negative = clip_encoder.encode(clip, prompt_neg)[0]

        results = FaceDetailer().doit(
            image, model, clip, vae, guide_size=512, guide_size_for=True, max_size=1024, seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
             positive=positive, negative=negative, denoise=noise, feather=5, noise_mask=True, force_inpaint=True,
             bbox_threshold=0.5, bbox_dilation=10, bbox_crop_factor=3.0,
             sam_detection_hint='center-1', sam_dilation=0, sam_threshold=0.93, sam_bbox_expansion=0, sam_mask_hint_threshold=0.7,
             sam_mask_hint_use_negative='False', drop_size=10, bbox_detector=bbox_detector, wildcard="", cycle=iterations,
        )
        return (data, results[0])

def render(checkpoint_config: MetCheckpointPreset, prompt_pos, prompt_neg, start_image=None, noise_seed=0, noise_strength=1.0, pass_index=1, lora_data: dict[str, float]={}):
    steps = checkpoint_config.steps
    cfg = checkpoint_config.cfg
    sampler_name = checkpoint_config.sampler
    scheduler = checkpoint_config.scheduler

    model, clip, vae = CheckpointLoaderSimple().load_checkpoint(checkpoint_config.path)
    model, clip = apply_loras(model, clip, lora_data)

    clip_encoder = CLIPTextEncode()
    pos_encoded = clip_encoder.encode(clip, prompt_pos)[0]
    neg_encoded = clip_encoder.encode(clip, prompt_neg)[0]

    if start_image == None:
        if pass_index > 1:
            raise Exception(f"No image for pass {pass_index}.")
        start_image = EmptyImage().generate(1024, 1024)[0]

    if noise_strength == 1:
        w, h = start_image.shape[2], start_image.shape[1]
        in_latent = EmptyLatentImage().generate(w, h)[0]
    else:
        in_latent = VAEEncode().encode(vae, start_image)[0]

    out_latent = KSampler().sample(model, noise_seed+pass_index, steps, cfg, sampler_name, scheduler, pos_encoded, neg_encoded, in_latent, noise_strength)[0]

    out_image = VAEDecode().decode(vae, out_latent)[0]

    return model, vae, clip, out_image

def apply_loras(model, clip, lora_data: dict[str, int]):
    if not lora_data:
        return model, clip
    for lora_name, weight in lora_data.items():
        print(f"Applying LoRA: {(lora_name, weight)}")

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, weight, weight)

    return model_lora, clip_lora

def ensure_required_loras(prompt: str, lora_configs: dict[str, LoRA_Config], api_token: str):
    # Extract lora tags.
    prompt_clean, lora_tags = extract_lora_tags(prompt)

    founds = re.findall(RE_LORA_TAGS, prompt)
    lora_files = folder_paths.get_filename_list("loras")

    lora_weights = {}
    for f in founds:
        tag = f[1:-1]
        pak = tag.split(":")
        type = pak[0]
        if type != 'lora':
            continue
        name = None
        if len(pak) > 1 and len(pak[1]) > 0:
            name = pak[1]
        else:
            continue
        weight = _clip_weight = 0
        try:
            if len(pak) > 2 and len(pak[2]) > 0:
                weight = float(pak[2])
                _clip_weight = weight
            if len(pak) > 3 and len(pak[3]) > 0:
                _clip_weight = float(pak[3])
        except ValueError:
            continue
        if weight == 0:
            continue
        if name == None:
            continue
        for lora_file in lora_files:
            if Path(lora_file).name.startswith(name) or lora_file.startswith(name):
                lora_weights[lora_file] = weight
                break
        else:
            lora_config = lora_configs.get(name.lower())
            if not lora_config:
                raise Exception(f"Missing LoRA: {name}\nEither remove it from the prompt, or download it manually, or use a Prepare Lora node to provide information about where to download the LoRA, so it can be automatically downloaded if it is missing.")
            DownloadCivitaiModel().download_model(api_token, lora_config.civitai_url, lora_config.subdir, lora_config.name_noext, lora_config.version)

    return prompt_clean, lora_weights


class RenderPass_Prepare:
    NAME = "Prepare Render Pass"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "checkpoint_datas": ("CHECKPOINT_DATAS",),
                "lora_datas": ("LORA_DATA",),
                "civitai_api_key": ("STRING",),
                "tag_stack": ("TAG_STACK",),
                "noise_seed": ("INT", {"control_after_generate": True, "tooltip": "Image noise seed"}),
                "prompt_seed": ("INT", {"control_after_generate": True, "tooltip": "Seed to use for prompt randomization when the prompt uses {red|green|blue} syntax"}),
            },
        }

    RETURN_NAMES = ("Data", )
    RETURN_TYPES = ("RENDER_PASS_DATA",)
    FUNCTION = "render_pass_prepare"
    CATEGORY = "MetsNodes"
    DESCRIPTION="""Render an image with the data provided."""

    def render_pass_prepare(self, checkpoint_datas={}, lora_datas={}, civitai_api_key="", tag_stack={}, noise_seed=0, prompt_seed=0):
        combined_data = {
            'checkpoint_datas': checkpoint_datas,
            'lora_datas': lora_datas,
            'civitai_api_key': civitai_api_key,
            'tag_stack': tag_stack,
            'noise_seed': noise_seed,
            'prompt_seed': prompt_seed
        }
        return (combined_data, )