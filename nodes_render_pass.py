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
    CheckpointLoaderSimple, KSampler, 
    CLIPTextEncode, CLIPSetLastLayer, 
    VAEEncode, VAEDecode, 
    EmptyImage, EmptyLatentImage, 
    ImageScaleBy, NODE_CLASS_MAPPINGS,
    PreviewImage, 
)

from torch import Tensor

NEG_TAG = "neg"
FACE_TAG = "face"
EYEDIR_TAG = "eyedir"
TOP_TAG = "!"
FORCE_NO_CP_PROMPT = "<!modelprompt>"
FORCE_PORTRAIT = "<ratio:portrait>"
FORCE_LANDSCAPE = "<ratio:landscape>"
FORCE_SQUARE = "<ratio:square>"

# NOTE: BE CAREFUL WITH REGEX!! Complex Regular Expressions on complex prompts can turn 
# into what's known as a Runaway Regex, and require near-infinite calculation!
# KEEP THESE SIMPLE, and then do simple string operations.
RE_TAG_NAMES = re.compile(r"<\/((?:\w|\s|!)+)>")
RE_EXCLUDE = re.compile(r"<!((?:\w|\s)+)>")
RE_LORA = re.compile(r"<lora.*?>")

MODEL_CACHE = OrderedDict() # filepath : loaded model, max 5 to avoid unnecessary re-loading but not overwhelm memory.

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
    CATEGORY = "Met's Nodes/Render Pass"
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

        ckpt_config: CheckpointConfig|None = next((cp for path, cp in checkpoint_datas.items() if cp.path==checkpoint_name), None)
        if not ckpt_config:
             raise Exception(f"Checkpoint config not found for: {checkpoint_name}\nYou need to provide it by plugging a Prepare Checkpoint node into a Prepare Render Pass Node, and then plugging that into this node.")

        prompt_pos = unroll_tag_stack(tidy_prompt(prompt_pos), tag_stack)
        prompt_neg = unroll_tag_stack(tidy_prompt(prompt_neg), tag_stack)

        # Randomize using {blue|red|green} syntax.
        # NOTE: Currently not done for the negative prompt, I don't think it would be useful.
        prompt_pos = randomize_prompt(prompt_pos, prompt_seed)

        if image==None:
            image = data.get('last_image', None)
        if image==None:
            image = EmptyImage().generate(1024, 1024)[0]

        # Apply aspect ratio override. (<ratio:portrait/landscape/square>)
        # This feature may discard or rotate the image, and is meant to be used with a blank input image.
        prompt_pos, image = override_width_height(prompt_pos, image)

        if image == None:
            image = EmptyImage().generate(1024, 1024)[0]

        # Extract contents of <face> tags.
        # NOTE: This should be done before handling <neg> tags, so that it's 
        # possible to send face negative prompts through by stacking exactly like so:
        # <face><neg>glowing eyes</neg></face>
        prompt_pos, face_pos = extract_face_prompts(prompt_pos)
        prompt_neg, face_neg = extract_face_prompts(prompt_neg)

        # Remove contents of tags which are marked for removal using exclamation mark syntax: <!tag>
        # Useful when a prompt wants to signify that it's not compatible with something.
        # Eg., to easily prompt a blink, mark descriptions of <eye>eyes</eye>, then use "blink <!eye>" in prompt.
        # NOTE: This must come AFTER apply_checkpoint_prompt(), otherwise it will remove the <!modelprompt> 
        # keyword before it has a chance to trigger.
        prompt_pos = remove_excluded_tags(prompt_pos)

        if is_prompt_additive and pass_index != 1:
            prev_pos, prev_neg = data.get("prompt_pos", ""), data.get("prompt_neg", "")
            if prev_pos:
                prompt_pos += ",\n"+prev_pos
                prev_face_pos = data.get("prompt_face_pos", "")
                face_pos += ",\n"+prev_face_pos
            if prev_neg:
                prompt_neg += ",\n"+prev_neg
                prev_face_neg = data.get("prompt_face_neg", "")
                face_neg += ",\n"+prev_face_neg

        # Store the prompt in its current state of processing to be sent on to subsequent render passes.
        # NOTE: This should happen BEFORE applying checkpoint prompt, since we don't want to send the checkpoint's prompt to the next render pass.
        data["prompt_pos"] = prompt_pos
        data["prompt_neg"] = prompt_neg
        data["prompt_face_pos"] = face_pos
        data["prompt_face_neg"] = face_neg

        # Apply the checkpoint's associated +/- prompts.
        # NOTE: This should happen AFTER storing the prompt, since we don't want to send the checkpoint's prompt to the next render pass.
        prompt_pos, prompt_neg = apply_checkpoint_prompt(prompt_pos, prompt_neg, ckpt_config)

        # Move contents of <neg> tags to negative prompt, 
        # and remove exact matches of negative prompt words from the positive prompt.
        prompt_pos, prompt_neg = move_neg_tags(prompt_pos, prompt_neg)

        # Move contents of <!> tags to beginning of prompt.
        prompt_pos = reorder_prompt(prompt_pos)

        prompt_pos = remove_all_tag_syntax(prompt_pos)
        prompt_neg = remove_all_tag_syntax(prompt_neg)

        # Scale image, if requested.
        if scale != 1.0:
            upscale_method = 'area' if scale < 1.0 else 'lanczos'
            image = ImageScaleBy().upscale(image, upscale_method, scale)[0]

        # This is probably the best moment to save the prompt that will appear on CivitAI:
        # Lora tags are still in there, but all our custom syntax has been handled.
        data["prompt_pos_final"] = tidy_prompt(prompt_pos)
        data["prompt_neg_final"] = tidy_prompt(prompt_neg)

        # Load and remove LoRA tags.
        api_token = data.get("civitai_api_key", "")
        lora_data = data.get("lora_datas", {})
        prompt_pos, lora_data = ensure_required_loras(prompt_pos, lora_data, api_token)

        # Render the image.
        model, vae, clip, final_image, last_latent, pos_encoded, neg_encoded = render(ckpt_config, prompt_pos, prompt_neg, image, noise_seed, noise, pass_index=pass_index, lora_data=lora_data)
        # Store a bunch of data that can be accessed by Split Data node.
        data['last_image'] = final_image
        data['last_model'] = model
        data['last_vae'] = vae
        data['last_clip'] = clip
        data['last_checkpoint_config'] = ckpt_config
        data['pos_encoded'] = pos_encoded
        data['neg_encoded'] = neg_encoded
        data['last_latent'] = last_latent
        if 'modelnames' not in data:
            data['modelnames'] = set()
        data['modelnames'].add(checkpoint_name)

        # Get image preview data (for this, since this is also a sampler node, ComfyUIManager's preview has to be disabled, since it overrides this.)
        res = PreviewImage().save_images(final_image, filename_prefix="RenderPass-")
        ui_image = res['ui']['images']

        # Return preview data + node outputs
        return {
            "ui": {"images": ui_image},
            "result": (data, final_image, data["prompt_pos_final"], data["prompt_neg_final"]),
        }

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
    CATEGORY = "Met's Nodes/Render Pass"
    DESCRIPTION="""Simplified wrapper for the Impact Pack's FaceDetailer node."""
    OUTPUT_NODE = True

    def face_pass_execute(self, data, image, noise, iterations, prompt_pos, prompt_neg):
        try:
            from impact.impact_pack import FaceDetailer
        except ModuleNotFoundError:
            raise Exception("FaceDetailer node must be installed (from Impact Pack).")
        data = data.copy()
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

        # Move contents of <neg> tags to negative prompt, 
        # and remove exact matches of negative prompt words from the positive prompt.
        prompt_pos, prompt_neg = move_neg_tags(prompt_pos, prompt_neg)

        prompt_pos = remove_excluded_tags(remove_all_tag_syntax(tidy_prompt(prompt_pos)))
        prompt_neg = remove_excluded_tags(remove_all_tag_syntax(tidy_prompt(prompt_neg)))

        # Save final prompt for potential Split Data node.
        data["prompt_pos_final"] = prompt_pos
        data["prompt_neg_final"] = prompt_neg

        clip_encoder = CLIPTextEncode()
        positive = clip_encoder.encode(clip, prompt_pos)[0]
        negative = clip_encoder.encode(clip, prompt_neg)[0]

        results = FaceDetailer().doit(
            image, model, clip, vae, guide_size=512, guide_size_for=True, max_size=1024, seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
             positive=positive, negative=negative, denoise=noise, feather=10, noise_mask=True, force_inpaint=True,
             bbox_threshold=0.5, bbox_dilation=40, bbox_crop_factor=3.0,
             sam_detection_hint='center-1', sam_dilation=0, sam_threshold=0.93, sam_bbox_expansion=0, sam_mask_hint_threshold=0.7,
             sam_mask_hint_use_negative='False', drop_size=10, bbox_detector=bbox_detector, wildcard="", cycle=iterations,
        )

        image = results[0]

        # Get image preview data (for this, since this is also a sampler node, ComfyUIManager's preview has to be disabled, since it overrides this.)
        res = PreviewImage().save_images(image, filename_prefix="FaceRenderPass-")
        ui_image = res['ui']['images']

        # Return preview data + node outputs
        return {
            "ui": {"images": ui_image},
            "result": (data, image, tidy_prompt(prompt_pos), tidy_prompt(prompt_neg)),
        }

### String functions ###
def unroll_tag_stack(prompt: str, tag_stack: dict[str, str]) -> str:
    tag_names = list(tag_stack.keys())
    def present_tags(prompt):
        return {tag for tag in tag_names if f'<{tag}>' in prompt}
    def unroll_tag(prompt, tag):
        return tidy_prompt(prompt.replace(f'<{tag}>', tag_stack[tag]))

    tags_to_unroll = present_tags(prompt)
    while tags_to_unroll:
        for tag in tags_to_unroll:
            prompt = unroll_tag(prompt, tag)
        tags_to_unroll = present_tags(prompt)

    return tidy_prompt(prompt)

def move_neg_tags(positive: str, negative: str) -> tuple[str, str]:
    """We support <neg>Moving this from positive to negative prompt</neg> and also excluding negative keywords from the positive prompt."""
    # Extract negative tags.
    _, positive, neg_tag_contents = extract_tag_from_text(positive, NEG_TAG)
    negative += ", " + neg_tag_contents

    lines = positive.split("\n")
    new_lines = []
    for line in lines:
        words = [word.strip() for word in line.split(",")]
        for neg_word in negative.split(","):
            neg_word = neg_word.strip()
            if not neg_word:
                continue
            if neg_word in words:
                words.remove(neg_word)
        if words:
            new_lines.append(", ".join(words))
        else:
            new_lines.append("")

    positive = "\n".join(new_lines)

    return positive, negative

def override_width_height(prompt, image: Tensor) -> tuple[str, Tensor]:
    width, height = image.shape[2], image.shape[1]
    short = min(width, height)
    long = max(width, height)

    if FORCE_PORTRAIT in prompt:
        prompt = prompt.replace(FORCE_PORTRAIT, "")
        if width==long:
            image = image.permute(0, 2, 1, 3).contiguous()
    
    if FORCE_LANDSCAPE in prompt:
        prompt = prompt.replace(FORCE_LANDSCAPE, "")
        if width==short:
            image = image.permute(0, 2, 1, 3).contiguous()

    if FORCE_SQUARE in prompt:
        prompt = prompt.replace(FORCE_SQUARE, "")
        if short != long:
            average = int((long+short)/2)
            image = EmptyImage().generate(average, average)[0]

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
    _, prompt_pos, tag_content = extract_tag_from_text(prompt, TOP_TAG)
    return ", ".join([tag_content, prompt_pos])

def remove_excluded_tags(prompt: str) -> str:
    # Find all <!tag> instructions.
    tags_marked_for_exclude = set(RE_EXCLUDE.findall(prompt))

    # Remove the exclusion instruction tags themselves, now that we have them stored.
    prompt = RE_EXCLUDE.sub("", prompt)

    for exclude_tag in tags_marked_for_exclude:
        # Extract tag content and remove the tags themselves, no matter what.
        _, prompt, _ = extract_tag_from_text(prompt, exclude_tag)

    return prompt

def remove_all_tag_syntax(prompt: str) -> str:
    """Remove all <tag></tag> syntax strings. Useful at the end of prompt processing to remove leftover tags if they weren't used."""
    # Detect all tag names present
    all_tag_names = set(RE_TAG_NAMES.findall(prompt))

    for tag in all_tag_names:
        # Remove the tags themselves, no matter what.
        prompt = extract_tag_from_text(prompt, tag)[0]

    return prompt

def extract_face_prompts(prompt: str) -> tuple[str, str]:
    # Extract contents of <face> tags to send on to the FaceDetailer.
    cleaned_prompt, _, eyedir_prompt = extract_tag_from_text(prompt, EYEDIR_TAG)
    cleaned_prompt, _, face_prompt = extract_tag_from_text(cleaned_prompt, FACE_TAG)
    return cleaned_prompt, ",".join([face_prompt, eyedir_prompt])

### Render functions ###
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
        # NOTE: This results in memory leak console warnings, which seems fair... Sad, though.
        model, clip, vae = MODEL_CACHE.get(checkpoint_config.path)
    else:
        # NOTE: We cannot download a missing model because the frontend will raise an error during input validation,
        # which is very fucking frustrating.
        model, clip, vae = CheckpointLoaderSimple().load_checkpoint(checkpoint_config.path)
        MODEL_CACHE[checkpoint_config.path] = model, clip, vae
        if len(MODEL_CACHE) > 5:
            MODEL_CACHE.popitem(last=False)
        print("Model load: ", time.time()-start)
        start = time.time()
    model, clip = apply_loras(model, clip, lora_data)
    print("LoRA load: ", time.time()-start)

    clip_encoder = CLIPTextEncode()
    clip_skip = CLIPSetLastLayer()
    clip = clip_skip.set_last_layer(clip, checkpoint_config.clip_skip)[0]
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

    return model, vae, clip, out_image, out_latent, pos_encoded, neg_encoded

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

    tags = re.findall(RE_LORA_TAGS, prompt)
    lora_files = folder_paths.get_filename_list("loras")

    lora_weights = {}
    any_downloaded = False
    for full_tag in tags:
        tag = full_tag[1:-1]
        pak = tag.split(":")
        type = pak[0].strip()
        if type != 'lora':
            continue
        name = None
        if len(pak) > 1 and len(pak[1]) > 0:
            name = pak[1].strip()
            name = Path(name).stem
        else:
            continue
        if name == None:
            continue
        weight = _clip_weight = 0
        if len(pak) > 2 and len(pak[2]) > 0:
            try:
                weight = float(pak[2].strip())
                _clip_weight = weight
            except ValueError:
                continue
        if weight == 0:
            continue
        for lora_file in lora_files:
            if Path(lora_file).stem.lower() == name.lower() or lora_file.startswith(name):
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
                "noise_seed": ("INT", {"control_after_generate": True, "tooltip": "Image noise seed", "min": 0, "max": 2**64}),
                "prompt_seed": ("INT", {"control_after_generate": True, "tooltip": "Seed to use for prompt randomization when the prompt uses {red|green|blue} syntax", "min": 0, "max": 2**64}),
            },
        }

    RETURN_NAMES = ("Data", )
    RETURN_TYPES = ("RENDER_PASS_DATA",)
    FUNCTION = "render_pass_prepare"
    CATEGORY = "Met's Nodes/Render Pass"
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

class SplitData:
    NAME = "Split Data"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "data": ("RENDER_PASS_DATA", {"tooltip": "Various data collected, to be used by a render pass."}),
            },
        }

    RETURN_NAMES = ("Prompt_Pos", "Prompt_Neg", "Prompt_Face_Pos", "Prompt_Face_Neg", "Image", "Latent", "Model_Names", "Model",      "Vae", "Clip", "Positive",     "Negative",      "Tag_Stack", "Checkpoint_Datas", "LoRA_Datas", "Prompt_Seed", "Noise_Seed", "Steps", "CFG", "Sampler",                        "Scheduler")
    RETURN_TYPES = ("STRING",     "STRING",     "STRING",          "STRING",          "IMAGE", "LATENT", "STRING",      "MODEL",      "VAE", "CLIP", "CONDITIONING", "CONDITIONING", "TAG_STACK", "CHECKPOINT_DATAS", "LORA_DATA",  "INT",         "INT",        "INT",   "FLOAT", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS)
    FUNCTION = "split_data"
    CATEGORY = "Met's Nodes/Render Pass"
    DESCRIPTION="""Split the data socket of a Render Pass node, for custom processing."""

    def split_data(self, data):
        prompt_pos = data.get("prompt_pos_final", "")
        prompt_neg = data.get("prompt_neg_final", "")
        prompt_face_pos = data.get("prompt_face_pos", "")
        prompt_face_neg = data.get("prompt_face_neg", "")
        image = data.get("last_image", None)
        model = data.get("last_model", None)
        vae = data.get("last_vae", None)
        clip = data.get("last_clip", None)
        prompt_seed = data.get("prompt_seed", 1)
        noise_seed = data.get("noise_seed", 1)
        tag_stack = data.get("tag_stack", {})
        checkpoint_datas = data.get("checkpoint_datas", {})
        lora_data = data.get("lora_datas", {})

        pos_encoded = data.get('pos_encoded', None)
        neg_encoded = data.get('neg_encoded', None)
        latent = data.get('last_latent', None)

        checkpoint_cfg = data.get("last_checkpoint_config")
        steps = checkpoint_cfg.steps
        cfg = checkpoint_cfg.cfg
        sampler_name = checkpoint_cfg.sampler
        scheduler = checkpoint_cfg.scheduler

        model_names = ",".join(list(data.get('modelnames')))

        return (prompt_pos, prompt_neg, prompt_face_pos, prompt_face_neg, image, latent, model_names, model, vae, clip, pos_encoded, neg_encoded, tag_stack, checkpoint_datas, lora_data, prompt_seed, noise_seed, steps, cfg, sampler_name, scheduler)
