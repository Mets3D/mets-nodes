import re, os, random

from .met_context import MetCheckpointPreset, MetContext, MetFaceContext
import folder_paths
import comfy.samplers
# NOTE: Requires Impact Pack, sadly.
import impact.core as core
from .regex_nodes import randomize_prompt, extract_tag_from_text, tidy_prompt

NEG_TAG = "neg"
FACE_TAG = "face"
TOP_TAG = "!"
FORCE_NO_CP_PROMPT = "<!modelprompt>"
FORCE_PORTRAIT = "<ratio:portrait>"
FORCE_LANDSCAPE = "<ratio:landscape>"
FORCE_SQUARE = "<ratio:square>"
CONTEXT_PREFIX = "context_"

RE_CHECKPOINT = re.compile(r"<checkpoint:([^>]+)>")
RE_LORA = re.compile(r"<lora.*?>")
RE_TAG_NAMES = re.compile(r"<\/((?:\w|\s)+)>")
RE_CONTEXT_TAGS = re.compile(rf"<{CONTEXT_PREFIX}.*?>")
RE_EXCLUDE = re.compile(r"<!((?:\w|\s)+)>")

class MegaPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_pos": ("STRING", {"multiline": True, "tooltip": "The base positive prompt."}),
                "prompt_neg": ("STRING", {"multiline": True, "tooltip": "The base negative prompt."}),
                "prompt_seed": ("INT", {"control_after_generate": True, "tooltip": "Prompt randomization seed."}),
                "noise_seed": ("INT", {"control_after_generate": True, "tooltip": "Noise seed."}),
                "width": ("INT", {"tooltip": "Image width.", "default": 1024, "min": 0, "max": 2**16}),
                "height": ("INT", {"tooltip": "Image height.", "default": 1024, "min": 0, "max": 2**16}),
            },
            "optional": {
                "checkpoint_datas": ("CHECKPOINT_DATAS",),
                "tag_stack": ("TAG_STACK",),
            },
        }

    RETURN_NAMES = ("Context 1", "Context 2", "Context 3", "Context 4", "FaceDetailer Context")
    RETURN_TYPES = ("METCONTEXT","METCONTEXT","METCONTEXT","METCONTEXT","METFACECONTEXT")
    FUNCTION = "mega_prompt"
    CATEGORY = "MetsNodes"
    DESCRIPTION="""Process the mega prompt into 4 consecutive rendering contexts."""

    def mega_prompt(self, prompt_pos, prompt_neg, prompt_seed, noise_seed, width, height, checkpoint_datas, tag_stack) -> tuple[MetContext, MetContext, MetContext, MetContext, MetFaceContext]:
        # Unroll <tags> if present in the tag_stack.
        prompt_pos = unroll_tag_stack(prompt_pos, tag_stack)
        prompt_neg = unroll_tag_stack(prompt_neg, tag_stack)

        # Randomize using {blue|red|green} syntax.
        # NOTE: Currently not done for the negative prompt, I don't think it would be useful.
        prompt_pos = randomize_prompt(prompt_pos, prompt_seed)

        # Extract contents of <face> tags to send on to the FaceDetailer.
        _, prompt_face_pos = extract_tag_from_text(prompt_pos, FACE_TAG, remove_content=False)
        _, prompt_face_neg = extract_tag_from_text(prompt_neg, FACE_TAG, remove_content=False)

        # Apply aspect ratio override.
        width, height, prompt_pos = override_width_height(prompt_pos, width, height)

        # Tidy prompts.
        prompt_face_pos = tidy_prompt(prompt_face_pos)
        prompt_face_neg = tidy_prompt(prompt_face_neg)

        contexts = []
        for i in range(1, 5):
            ctx_pos = extract_context_tags(prompt_pos, str(i))
            ctx_neg = extract_context_tags(prompt_neg, str(i))
            # Extract which checkpoint to use, specified by <checkpoint:identifier>.
            ctx_pos, ctx_neg, checkpoint = apply_checkpoint_data(ctx_pos, ctx_neg, checkpoint_datas)

            # Extract lora tags.
            lora_tags = extract_lora_tags(ctx_pos)

            # Move contents of <neg> tags to negative prompt, 
            # and remove exact matches of negative prompt words from the positive prompt.
            ctx_pos, ctx_neg = move_neg_tags(ctx_pos, ctx_neg)

            # Move contents of <!> tags to beginning of prompt.
            ctx_pos = reorder_prompt(ctx_pos)

            # Remove contents of tags which are marked for removal using exclamation mark syntax: <!tag>
            # Useful when a prompt wants to signify that it's not compatible with something.
            # Eg., to easily prompt a blink, mark descriptions of <eye>eyes</eye>, then use "blink <!eye>" in prompt.
            ctx_pos = remove_excluded_tags(ctx_pos)

            ctx_pos = remove_all_tag_syntax(tidy_prompt(ctx_pos))
            ctx_neg = remove_all_tag_syntax(tidy_prompt(ctx_neg))

            contexts.append(MetContext(
                checkpoint=checkpoint,
                # NOTE: It's important to offset the noise seed for subsequent samplers.
                # Stacking results of different checkpoints with the same noise pattern has poor results for some reason.
                noise_seed=noise_seed+i if noise_seed <= 0 else random.randint(1, 100000000),
                prompt_seed=prompt_seed,
                width=width,
                height=height,
                pos_prompt=ctx_pos,
                neg_prompt=ctx_neg,
                loras=lora_tags,
            ))

        face_context = MetFaceContext(
            checkpoint=contexts[0].checkpoint,
            face_iterations=1,
            face_noise_amount=0.32,
            pos_prompt=prompt_face_pos,
            neg_prompt=prompt_face_neg,
            noise_seed=contexts[0].noise_seed+4,
            loras=contexts[0].loras,
        )
        return (*contexts, face_context)

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
            positive = positive.replace(neg_word, "")
    return positive, negative

def override_width_height(prompt, width, height) -> tuple[int, int, str]:
    short = min(width, height)
    long = max(width, height)
    if FORCE_PORTRAIT in prompt:
        return short, long, prompt.replace(FORCE_PORTRAIT, "")
    elif FORCE_LANDSCAPE in prompt:
        return long, short, prompt.replace(FORCE_LANDSCAPE, "")
    elif FORCE_SQUARE in prompt:
        average = long+short/2
        return average, average, prompt.replace(FORCE_SQUARE, "")
    return width, height, prompt

def apply_checkpoint_data(prompt_pos: str, prompt_neg: str, checkpoint_datas: dict[str, MetCheckpointPreset]) -> tuple[str, str, MetCheckpointPreset|None]:
    match = RE_CHECKPOINT.search(prompt_pos)
    checkpoint_name = match.group(1) if match else ""
    checkpoint = checkpoint_datas.get(checkpoint_name)

    if checkpoint:
        if FORCE_NO_CP_PROMPT not in prompt_pos:
            # Put checkpoint's quality tags at START of +prompt. (maybe not important)
            prompt_pos = ",\n".join([checkpoint.model_pos_prompt, RE_CHECKPOINT.sub("", prompt_pos)])
        else:
            prompt_pos = prompt_pos.replace(FORCE_NO_CP_PROMPT, "")

        if FORCE_NO_CP_PROMPT not in prompt_neg:
            # Put checkpoint's negative tags at END of -prompt. (maybe not important)
            prompt_neg = ",\n".join([RE_CHECKPOINT.sub("", prompt_neg), checkpoint.model_neg_prompt])
        else:
            prompt_neg = prompt_neg.replace(FORCE_NO_CP_PROMPT, "")

    return prompt_pos, prompt_neg, checkpoint

def extract_lora_tags(prompt: str) -> list[str]:
    return RE_LORA.findall(prompt)

def reorder_prompt(prompt: str) -> str:
    prompt_pos, tag_content = extract_tag_from_text(prompt, TOP_TAG, remove_content=True)
    return ", ".join([tag_content, prompt_pos])

def extract_context_tags(prompt: str, context_id: str) -> str:
    for context_tag in RE_CONTEXT_TAGS.findall(prompt):
        if context_tag == f"<{CONTEXT_PREFIX + str(context_id)}>":
            prompt, _discard = extract_tag_from_text(prompt, context_tag[1:-1], remove_content=False)
        else:
            prompt, _discard = extract_tag_from_text(prompt, context_tag[1:-1], remove_content=True)

    return prompt

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

    return prompt

class MetContextBreak:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Context": ("METCONTEXT", {"tooltip": "Context to split up."}),
            }
        }

    RETURN_NAMES, RETURN_TYPES = map(list, zip(*{
        'Checkpoint Path': folder_paths.get_filename_list("checkpoints"),
        'Checkpoint Name': 'STRING',
        'Clip Skip': 'INT',

        '+Prompt': 'STRING',
        '-Prompt': 'STRING',
        'LoRA Tags': 'STRING',
        'Noise Seed': 'INT',
        'Steps': 'INT',
        'CFG': 'FLOAT',
        'Sampler Name': comfy.samplers.KSampler.SAMPLERS,
        'Scheduler Name': comfy.samplers.KSampler.SCHEDULERS,

        'Width': 'INT',
        'Height': 'INT',
        'Scale By': 'FLOAT',
        'Add Noise': 'FLOAT',
    }.items()))
    FUNCTION = "break_context"
    CATEGORY = "MetsNodes"
    DESCRIPTION="""Break context into primitive data sockets."""

    def break_context(self, Context):
        return (
            Context.checkpoint.path,
            Context.checkpoint.name,
            Context.checkpoint.clip_skip,

            Context.pos_prompt,
            Context.neg_prompt,
            "\n".join(Context.loras),
            Context.noise_seed,
            Context.checkpoint.steps,
            Context.checkpoint.cfg,
            Context.checkpoint.sampler,
            Context.checkpoint.scheduler,

            Context.width,
            Context.height,
            Context.scale,
            Context.add_noise,
        )

class MetFaceContextBreak:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "FaceContext": ("METFACECONTEXT", {"tooltip": "Context to split up."}),
            }
        }

    RETURN_NAMES, RETURN_TYPES = map(list, zip(*{
        'Checkpoint Path': folder_paths.get_filename_list("checkpoints"),
        'Checkpoint Name': 'STRING',
        'Clip Skip': 'INT',

        '+Prompt': 'STRING',
        '-Prompt': 'STRING',
        'LoRA Tags': 'STRING',
        'Noise Seed': 'INT',
        'Steps': 'INT',
        'CFG': 'FLOAT',
        'Sampler Name': comfy.samplers.KSampler.SAMPLERS,
        'Scheduler Name': core.SCHEDULERS,

        'Face Noise': 'FLOAT',
        'Face Iterations': 'INT',
    }.items()))
    FUNCTION = "break_context"
    CATEGORY = "MetsNodes"
    DESCRIPTION="""Break context into primitive data sockets."""

    def break_context(self, FaceContext):
        return (
            FaceContext.checkpoint.path,
            FaceContext.checkpoint.name,
            FaceContext.checkpoint.clip_skip,

            FaceContext.pos_prompt,
            FaceContext.neg_prompt,
            "\n".join(FaceContext.loras),
            FaceContext.noise_seed,
            FaceContext.checkpoint.steps,
            FaceContext.checkpoint.cfg,
            FaceContext.checkpoint.sampler,
            FaceContext.checkpoint.scheduler,

            FaceContext.face_noise_amount,
            FaceContext.face_iterations,
        )

class MetPrepareCheckpoint:
    DESCRIPTION = ("A simple string search and replace operation that is designed to nicely chain together. Can be used to build complex randomized prompts.")
    RETURN_NAMES, RETURN_TYPES = map(list, zip(*{"Checkpoint Datas": 'CHECKPOINT_DATAS'}.items()))
    FUNCTION = "prepare_checkpoint"
    CATEGORY = "MetsNodes"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "checkpoint_datas": ("CHECKPOINT_DATAS", {"tooltip": "Checkpoint datas"}),
                "identifier": ("STRING", {"tooltip": "Name used to identify this checkpoint preset in the prompt processor"}),
                "path": ("STRING", {"tooltip": "Filepath relative to checkpoints folder, excluding filename"}),
                "filename": ("STRING", {"tooltip": "Filename of checkpoint file (without extension)"}),
                "civitai_id": ("INT", {"tooltip": f"CivitAI model ID. Can be found in the model's page URL: civitai.com/models/<model id>", "min": 0, "max": 100000000}),
                "version_name": ("STRING", {"tooltip": f"CivitAI model version name. If not specified, we assume the left-most version shown in the horizontal list on the model's page", "default":""}),
                "clip_skip": ("INT", {"tooltip": "CLIP Skip value for this checkpoint", "default": -2, "min":-2, "max":0}),
                "steps": ("INT", {"tooltip": "Number of denoising steps to use with this checkpoint", "default": 25}),
                "cfg": ("FLOAT", {"tooltip": "CFG value to use with this checkpoint", "default": 4.5}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "Sampler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "Scheduler"}),
                "prompt_pos": ("STRING", {"multiline": True, "tooltip": "Base positive prompt associated with this checkpoint"}),
                "prompt_neg": ("STRING", {"multiline": True, "tooltip": "Base negative prompt associated with this checkpoint"}),
            },
        }

    def prepare_checkpoint(self, **kwargs):
        checkpoint_datas = kwargs.get('checkpoint_datas', {})
        checkpoint_datas.update({kwargs['identifier']: MetCheckpointPreset(
            civitai_model_id=kwargs['civitai_id'],
            clip_skip=kwargs['clip_skip'],
            path=os.sep.join([kwargs['path'], kwargs['filename']+".safetensors"])[1:],
            steps=kwargs['steps'],
            cfg=kwargs['cfg'],
            sampler=kwargs['sampler'],
            scheduler=kwargs['scheduler'],
            model_pos_prompt=kwargs['prompt_pos'],
            model_neg_prompt=kwargs['prompt_neg'],
        )})
        return (checkpoint_datas,)

class TagStacker:
    DESCRIPTION = ("Define a tag, which can be plugged into the MegaPrompt node, and un-furled using the <tag> syntax.")
    RETURN_NAMES, RETURN_TYPES = map(list, zip(*{"Tag Stack": 'TAG_STACK'}.items()))
    FUNCTION = "add_tag"
    CATEGORY = "MetsNodes/Tag Tools"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "tag_stack": ("TAG_STACK", {"tooltip": "Tag stack"}),
                "tag": ("STRING", {"multiline": False, "tooltip": "The name of this <tag>."}),
                "content": ("STRING", {"multiline": True, "tooltip": "The contents of the tag."}),
            }
        }

    def add_tag(self, tag_stack={}, tag="", content=""):
        tag_stack.update({tag: content})
        return (tag_stack,)
