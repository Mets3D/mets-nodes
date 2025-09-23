import re, os, random
from .met_context import MetCheckpointPreset, MetContext, MetFaceContext
import folder_paths
import comfy.samplers
# NOTE: Requires Impact Pack, sadly.
import impact.core as core
from .regex_nodes import randomize_prompt, extract_tag_from_text, remove_comment_lines, tidy_prompt

NEG_TAG = "neg"
FACE_TAG = "face"
TOP_TAG = "!"
FORCE_NO_CP_PROMPT = "<!modelprompt>"
FORCE_PORTRAIT = "<ratio:portrait>"
FORCE_LANDSCAPE = "<ratio:landscape>"
FORCE_SQUARE = "<ratio:square>"
CONTEXT_PREFIX = "context_"

RE_LORA = re.compile(r"<lora.*?>")
RE_TAG_NAMES = re.compile(r"<\/((?:\w|\s|!)+)>")
RE_CONTEXT_TAGS = re.compile(rf"(<({CONTEXT_PREFIX}[^:>]+)(?::(.*))?>)((?:\s|.)*?)(<\/\2>)")
RE_EXCLUDE = re.compile(r"<!((?:\w|\s)+)>")

class MegaPrompt:
    NAME = "Mega Prompt"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_contexts": ("INT", {"tooltip": "How many contexts should be used.", "min": 1, "max": 4, "default": 1}),
                "use_facedetailer": ("BOOLEAN", {"tooltip": "Whether facedetailer should be used or not."}),
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

    def mega_prompt(
            self, num_contexts, use_facedetailer, prompt_pos, prompt_neg, prompt_seed, noise_seed, width, height, checkpoint_datas, tag_stack
        ) -> tuple[MetContext, MetContext, MetContext, MetContext, MetFaceContext]:
        # Unroll <tags> if present in the tag_stack.
        prompt_pos = unroll_tag_stack(prompt_pos, tag_stack)
        prompt_neg = unroll_tag_stack(prompt_neg, tag_stack)

        # Randomize using {blue|red|green} syntax.
        # NOTE: Currently not done for the negative prompt, I don't think it would be useful.
        prompt_pos = randomize_prompt(prompt_pos, prompt_seed)

        # Apply aspect ratio override.
        width, height, prompt_pos = override_width_height(prompt_pos, width, height)

        contexts = []
        for i in range(1, 5):
            if i > num_contexts:
                contexts.append(None)
                continue

            # Extract which checkpoint to use for this context, and with what overridden parameters (if any).
            ctx_pos, ctx_params = extract_context_tags(prompt_pos, str(i))
            ctx_neg, _ = extract_context_tags(prompt_neg, str(i))

            checkpoint = checkpoint_datas.get(ctx_params.pop('checkpoint'), None)
            if not checkpoint:
                raise Exception(f"MegaPrompt error: A checkpoint is not specified for Context {i}.\nDo so by plugging in at least one Context Data, and then triggering it in the positive prompt using <checkpoint:name_of_checkpoint>.")

            # Apply the checkpoint's associated +/- prompts..
            ctx_pos, ctx_neg = apply_checkpoint_prompt(ctx_pos, ctx_neg, checkpoint)

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

            context = MetContext(
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
            )
            for obj in (context, checkpoint):
                for key, value in ctx_params.items():
                    if hasattr(obj, key):
                        val_type = type(getattr(obj, key))
                        value = val_type(value)
                        setattr(obj, key, value)

            # TODO: consider adding an equality check to previous context to avoid adding the same context multiple times, since that's kinda pointless.
            # Would also allow us to remove the num_context input, since it would become implicit. (maybe add a default checkpoint input instead so we aren't forced to use context def syntax)

            contexts.append(context)

        last_context = next(c for c in reversed(contexts) if c)
        face_pos, face_neg = extract_face_prompts(last_context)

        face_context = None
        if use_facedetailer:
            face_context = MetFaceContext(
                checkpoint=last_context.checkpoint,
                face_iterations=1,
                face_noise_amount=0.32,
                pos_prompt=face_pos,
                neg_prompt=face_neg,
                noise_seed=last_context.noise_seed+4,
                loras=last_context.loras,
            )
            for key, value_str in overrides.items():
                if hasattr(face_context, key):
                    val_type = type(getattr(face_context, key))
                    value = val_type(value_str)
                    setattr(face_context, key, value)

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

def apply_checkpoint_prompt(prompt_pos: str, prompt_neg: str, checkpoint:MetCheckpointPreset) -> tuple[str, str]:
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

def extract_lora_tags(prompt: str) -> list[str]:
    return RE_LORA.findall(prompt)

def reorder_prompt(prompt: str) -> str:
    prompt_pos, tag_content = extract_tag_from_text(prompt, TOP_TAG, remove_content=True)
    return ", ".join([tag_content, prompt_pos])

def extract_context_tags(prompt: str, context_id: str) -> tuple[str, dict[str, str]]:
    props = {}
    for tag_start, ctx_name, ctx_props, ctx_prompt, tag_end in RE_CONTEXT_TAGS.findall(prompt):
        whole_thing = tag_start+ctx_prompt+tag_end
        if ctx_name == CONTEXT_PREFIX + str(context_id):
            prompt = prompt.replace(tag_start, "").replace(tag_end, "")
            if ctx_props:
                props = {k.strip(): v.strip() for k, v in (pair.split("=") for pair in ctx_props.split(","))}
        else:
            prompt = prompt.replace(whole_thing, "")

    return prompt, props

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

def extract_face_prompts(context: MetContext) -> tuple[str, str]:
    # Extract contents of <face> tags to send on to the FaceDetailer.
    _, pos = extract_tag_from_text(context.pos_prompt, FACE_TAG)
    _, neg = extract_tag_from_text(context.neg_prompt, FACE_TAG)

    # Tidy face prompts.
    pos = remove_all_tag_syntax(tidy_prompt(pos))
    neg = remove_all_tag_syntax(tidy_prompt(neg))
    return pos, neg

class ContextBreak:
    NAME = "Context Break"
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
        if not Context:
            return (None,)
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

class FaceContextBreak:
    NAME = "Face Context Break"
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
        if not FaceContext:
            return (None,)
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

class PrepareCheckpoint:
    NAME = "Prepare Checkpoint"
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
    NAME = "Tag Stacker"
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
        tag_stack.update({tag: remove_comment_lines(content)})
        return (tag_stack,)
