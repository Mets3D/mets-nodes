import re, json, hashlib
from pathlib import Path

from .dataclasses import LoRAConfig, CheckpointConfig
from .nodes_prompt_tags import randomize_prompt, tidy_prompt, extract_tag_from_text
from .nodes_downloader import DownloadCivitaiModel
import time
from collections import OrderedDict

import comfy
import folder_paths
from nodes import (
    CheckpointLoaderSimple, KSampler, CLIPTextEncode, 
    VAEEncode, VAEDecode, EmptyImage, EmptyLatentImage, 
    ImageScaleBy, NODE_CLASS_MAPPINGS
)

from torch import Tensor

NEG_TAG = "neg"
FACE_TAG = "face"
TOP_TAG = "!"
FORCE_NO_CP_PROMPT = "<!modelprompt>"
FORCE_PORTRAIT = "<ratio:portrait>"
FORCE_LANDSCAPE = "<ratio:landscape>"
FORCE_SQUARE = "<ratio:square>"
CONTEXT_PREFIX = "context_"

# NOTE: BE CAREFUL WITH REGEX!! Complex Regular Expressions on complex prompts can turn into what's known as a Runaway Regex, and require near-infinite calculation!
# KEEP THESE SIMPLE, and then do simple string operations.
RE_CONTEXT_TAGS = re.compile(rf"(?s)<{CONTEXT_PREFIX}.*?>.*?<\/{CONTEXT_PREFIX}.*?>")
RE_TAG_NAMES = re.compile(r"<\/((?:\w|\s|!)+)>")
RE_EXCLUDE = re.compile(r"<!((?:\w|\s)+)>")
RE_LORA = re.compile(r"<lora.*?>")

MODEL_CACHE = OrderedDict() # filepath : loaded model, max 5 to avoid unnecessary re-loading but not overwhelm memory.

# TODO: 
# - Rename Metxyz classes to have better names, eg. CheckpointConfig->CheckpointConfig, PrepareCheckpoint->ConfigureCheckpoint
# - Rename their "identifier" input to "alias", although I really don't want to recreate all of them in my workflow (could just search and replace in the .json though...)
# - Add prompt processing to RenderPass node

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

        ckpt_config: CheckpointConfig|None = next((cp for identifier, cp in checkpoint_datas.items() if cp.path==checkpoint_name), None)
        if not ckpt_config:
             raise Exception(f"Checkpoint config not found for: {checkpoint_name}\nYou need to provide it by plugging a Prepare Checkpoint node into a Prepare Render Pass Node, and then plugging that into this node.")

        prompt_pos = tidy_prompt(prompt_pos)
        prompt_neg = tidy_prompt(prompt_neg)
        prompt_pos = unroll_tag_stack(prompt_pos, tag_stack)
        prompt_neg = unroll_tag_stack(prompt_neg, tag_stack)

        # Randomize using {blue|red|green} syntax.
        # NOTE: Currently not done for the negative prompt, I don't think it would be useful.
        prompt_pos = randomize_prompt(prompt_pos, prompt_seed)

        if image==None:
            image = data.get('last_image', None)
        if image==None:
            image = EmptyImage().generate(1024, 1024)[0]

        # Apply aspect ratio override. (<ratio:portrait/landscape/square>)
        # This feature may discard or rotate the image, and is meant to be used with a blank input image.
        ret = override_width_height(prompt_pos, image)
        prompt_pos, image = override_width_height(prompt_pos, image)

        if image == None:
            image = EmptyImage().generate(1024, 1024)[0]

        # Apply the checkpoint's associated +/- prompts.
        prompt_pos, prompt_neg = apply_checkpoint_prompt(prompt_pos, prompt_neg, ckpt_config)

        # Remove contents of tags which are marked for removal using exclamation mark syntax: <!tag>
        # Useful when a prompt wants to signify that it's not compatible with something.
        # Eg., to easily prompt a blink, mark descriptions of <eye>eyes</eye>, then use "blink <!eye>" in prompt.
        # NOTE: This must come AFTER apply_checkpoint_prompt(), otherwise it will remove the <!modelprompt> 
        # keyword before it has a chance to trigger.
        prompt_pos = remove_excluded_tags(prompt_pos)

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
        data["prompt_pos"] = prompt_pos
        data["prompt_neg"] = prompt_neg
        data["prompt_face_pos"] = face_pos
        data["prompt_face_neg"] = face_neg

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

    RETURN_NAMES = ("Data", "Image", "Prompt_Pos", "Prompt_Neg")
    RETURN_TYPES = ("RENDER_PASS_DATA","IMAGE", "STRING", "STRING")
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
        ckpt_config: CheckpointConfig|None = data.get('last_checkpoint_config', None)
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
        prompt_pos = remove_all_tag_syntax(tidy_prompt(prompt_pos))
        prompt_neg = remove_all_tag_syntax(tidy_prompt(prompt_neg))

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
        return (data, results[0], prompt_pos, prompt_neg)

### String functions ###
def unroll_tag_stack(prompt: str, tag_stack: dict[str, str]) -> str:
    tag_names = list(tag_stack.keys())
    def present_tags(prompt):
        return {tag for tag in tag_names if f'<{tag}>' in prompt}
    def unroll_tag(prompt, tag):
        return prompt.replace(f'<{tag}>', tag_stack[tag])

    tags_to_unroll = present_tags(prompt)
    while tags_to_unroll:
        for tag in tags_to_unroll:
            prompt = unroll_tag(prompt, tag)
        tags_to_unroll = present_tags(prompt)

    return prompt

def move_neg_tags(positive: str, negative: str) -> tuple[str, str]:
    """We support <neg>Moving this from positive to negative prompt</neg> and also excluding negative keywords from the positive prompt."""
    # Extract negative tags.
    positive, neg_tag_contents = extract_tag_from_text(positive, NEG_TAG, remove_content=True)
    negative += ", " + neg_tag_contents
    for neg_word in negative.split(","):
        neg_word = neg_word.strip()
        if not neg_word:
            continue
        if neg_word in positive:
            positive = re.sub(rf"(,\s*|^)\(?{neg_word}(:?.*\))?", ", ", positive)
    return positive, negative

def override_width_height(prompt, image: Tensor) -> tuple[str, Tensor]:
    width, height = image.shape[2], image.shape[1]
    short = min(width, height)
    long = max(width, height)
    prompt = tidy_prompt(prompt)
    if short==long:
        return prompt, image
    if FORCE_PORTRAIT in prompt and width==long:
        return prompt.replace(FORCE_PORTRAIT, ""), image.permute(0, 2, 1, 3).contiguous()
    elif FORCE_LANDSCAPE in prompt and width==short:
        return prompt.replace(FORCE_LANDSCAPE, ""), image.permute(0, 2, 1, 3).contiguous()
    elif FORCE_SQUARE in prompt:
        average = int(long+short/2)
        return prompt.replace(FORCE_SQUARE, ""), EmptyImage().generate(average, average)[0]
    return prompt, image

def apply_checkpoint_prompt(prompt_pos: str, prompt_neg: str, checkpoint: CheckpointConfig) -> tuple[str, str]:
    if FORCE_NO_CP_PROMPT not in prompt_pos:
        # Put checkpoint's quality tags at START of +prompt. (maybe not important)
        prompt_pos = ",\n".join([checkpoint.model_pos_prompt, prompt_pos])
    else:
        prompt_pos = prompt_pos.replace(FORCE_NO_CP_PROMPT, "")

    if FORCE_NO_CP_PROMPT not in prompt_neg:
        # Put checkpoint's negative tags at END of -prompt. (maybe not important)
        prompt_neg += checkpoint.model_neg_prompt
    else:
        prompt_neg = prompt_neg.replace(FORCE_NO_CP_PROMPT, "")

    return prompt_pos, prompt_neg

def extract_lora_tags(prompt: str) -> tuple[str, list[str]]:
    return RE_LORA.sub("", prompt), RE_LORA.findall(prompt)

def reorder_prompt(prompt: str) -> str:
    prompt_pos, tag_content = extract_tag_from_text(prompt, TOP_TAG, remove_content=True)
    return ", ".join([tag_content, prompt_pos])

def remove_excluded_tags(prompt: str) -> str:
    # Find all <!tag> instructions.
    tags_marked_for_exclude = set(RE_EXCLUDE.findall(prompt))

    # Remove the exclusion instruction tags themselves, now that we have them stored.
    prompt = RE_EXCLUDE.sub("", prompt)

    for exclude_tag in tags_marked_for_exclude:
        # Extract tag content and remove the tags themselves, no matter what.
        prompt, _discard = extract_tag_from_text(prompt, exclude_tag, remove_content=True)

    return prompt

def remove_all_tag_syntax(prompt: str) -> str:
    """Remove all <tag></tag> syntax strings. Useful at the end of prompt processing to remove leftover tags if they weren't used."""
    # Detect all tag names present
    all_tag_names = set(RE_TAG_NAMES.findall(prompt))

    for tag in all_tag_names:
        # Remove the tags themselves, no matter what.
        prompt, content = extract_tag_from_text(prompt, tag, remove_content=False)

    return tidy_prompt(prompt)

def extract_face_prompts(prompt: str) -> tuple[str, str]:
    # Extract contents of <face> tags to send on to the FaceDetailer.
    cleaned_prompt, face_prompt = extract_tag_from_text(prompt, FACE_TAG)
    return cleaned_prompt, face_prompt

def render(checkpoint_config: CheckpointConfig, prompt_pos, prompt_neg, start_image=None, noise_seed=0, noise_strength=1.0, pass_index=1, lora_data: dict[str, float]={}):
    steps = checkpoint_config.steps
    cfg = checkpoint_config.cfg
    sampler_name = checkpoint_config.sampler
    scheduler = checkpoint_config.scheduler

    global MODEL_CACHE
    start = time.time()
    if checkpoint_config.path in MODEL_CACHE and False:
        # Optimization: If the checkpoint and LoRAs are the same as in 
        # previous prompt, don't load the models again.
        # NOTE: Not sure if this can cause the model to be stuck in memory for ever!
        model, clip, vae = MODEL_CACHE.get(checkpoint_config.path)
    else:
        model, clip, vae = CheckpointLoaderSimple().load_checkpoint(checkpoint_config.path)
        MODEL_CACHE[checkpoint_config.path] = model, clip, vae
        if len(MODEL_CACHE) > 5:
            MODEL_CACHE.popitem(last=False)
        print("Model load: ", time.time()-start)
        start = time.time()
    model, clip = apply_loras(model, clip, lora_data)
    print("LoRA load: ", time.time()-start)

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

def hash_from_dict(d: dict) -> str:
    # Convert dict to a canonical JSON string (sorted keys ensure determinism)
    s = json.dumps(d, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(s.encode()).hexdigest()

def apply_loras(model, clip, lora_data: dict[str, int]):
    if not lora_data:
        return model, clip
    model_lora, clip_lora = None, None
    for lora_name, weight in lora_data.items():
        print(f"Applying LoRA: {(lora_name, weight)}")
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model_lora or model, clip_lora or clip, lora, weight, weight)

    return model_lora, clip_lora

def ensure_required_loras(prompt: str, lora_configs: dict[str, LoRAConfig], api_token: str):
    # Extract lora tags.
    prompt_clean, lora_tags = extract_lora_tags(prompt)

    founds = re.findall(RE_LORA_TAGS, prompt)
    lora_files = folder_paths.get_filename_list("loras")

    lora_weights = {}
    any_downloaded = False
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
            any_downloaded = True
    if any_downloaded:
        # Run this function again, now that everything is downloaded.
        return ensure_required_loras(prompt, lora_configs, api_token)

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