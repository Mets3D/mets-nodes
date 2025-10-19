import os

from .dataclasses import CheckpointConfig, LoRAConfig
from .nodes_prompt_tags import remove_comment_lines

import comfy.samplers

class PrepareCheckpoint:
    NAME = "Prepare Checkpoint"
    DESCRIPTION = ("Stack information about checkpoints, to later easily switch between checkpoints in a Render Pass node.")
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
        checkpoint_datas.update({kwargs['identifier'].lower(): CheckpointConfig(
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

class PrepareLoRA:
    NAME = "Prepare LoRA"
    DESCRIPTION = ("Stack information about LoRAs, so they can later easily be downloaded and used with a Render Pass node.")
    RETURN_NAMES, RETURN_TYPES = map(list, zip(*{"LoRA Data": 'LORA_DATA'}.items()))
    FUNCTION = "prepare_lora"
    CATEGORY = "MetsNodes"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "lora_data": ("LORA_DATA", {"tooltip": "Accumulated LoRA data"}),
                "civitai_id": ("INT", {"tooltip": f"CivitAI model ID. Can be found in the model's page URL: civitai.com/models/<model id>", "min": 0, "max": 100000000}),
                "version_name": ("STRING", {"tooltip": f"CivitAI model version name. If not specified, we assume the left-most version shown in the horizontal list on the model's page", "default":""}),
                "path": ("STRING", {"tooltip": "Filepath relative to checkpoints folder, excluding filename"}),
                "filename": ("STRING", {"tooltip": "Filename of checkpoint file (without extension)"}),
                "triggers": ("STRING", {"multiline": True, "tooltip": "You can take note of the trigger words here. This is not actually used anywhere."}),
            },
        }

    def prepare_lora(self, **kwargs):
        lora_data = kwargs.get('lora_data', {})
        lora_data.update({kwargs['filename'].lower(): LoRAConfig(
            civitai_model_id=kwargs['civitai_id'],
            version=kwargs['version_name'],
            path=os.sep.join([kwargs['path'], kwargs['filename']+".safetensors"]),
        )})
        return (lora_data,)

class TagStacker:
    NAME = "Tag Stacker"
    DESCRIPTION = ('Define a tag, which can be plugged into the "Prepare Render Pass" node, and un-furled using the <tag> syntax.')
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
        content = remove_comment_lines(content)
        test_brackets(content, tag=tag)
        tag_stack.update({tag: content})
        return (tag_stack,)

def test_brackets(prompt: str, tag="") -> bool:
    """
    Verifies that all curly braces in the string are properly matched and ordered.
    Prints context with the problematic bracket in red, then raises Exception if an issue is found.
    """
    stack = []
    context_window = 20  # characters around the bracket to show

    RED = "\033[1m\033[31m"
    RESET = "\033[0m"

    for i, ch in enumerate(prompt):
        if ch == '{':
            stack.append(i)
        elif ch == '}':
            if not stack:
                # unmatched closing brace
                start = max(0, i - context_window)
                end = min(len(prompt), i + context_window + 1)
                context = (
                    prompt[start:i] + RED + prompt[i] + RESET + prompt[i + 1:end]
                )
                print(f"Tag Stacker ({tag}): Unmatched closing '}}':\n...{context}...")
                raise Exception(f'Unmatched closing "}}" in tag: "{tag}"')
            stack.pop()

    if stack:
        # unmatched opening brace(s) remaining
        first_unmatched_index = stack[0]
        start = max(0, first_unmatched_index - context_window)
        end = min(len(prompt), first_unmatched_index + context_window + 1)
        context = (
            prompt[start:first_unmatched_index]
            + RED
            + prompt[first_unmatched_index]
            + RESET
            + prompt[first_unmatched_index + 1:end]
        )
        print(f"Tag Stacker ({tag}): Unmatched opening '{{':\n...{context}...")
        raise Exception(f'Unmatched opening "{{" in tag "{tag}"')

    return True

class TagTweaker:
    NAME = "Tag Tweaker"
    DESCRIPTION = ("Search and replace among all value strings of the tag stack")
    RETURN_NAMES, RETURN_TYPES = map(list, zip(*{"Tag Stack": 'TAG_STACK'}.items()))
    FUNCTION = "tweak_tags"
    CATEGORY = "MetsNodes/Tag Tools"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "tag_stack": ("TAG_STACK", {"tooltip": "Tag stack"}),
                "find": ("STRING", {"multiline": True, "tooltip": "Text to replace in the tags."}),
                "replace": ("STRING", {"multiline": True, "tooltip": "Text to replace with."}),
            }
        }

    def tweak_tags(self, tag_stack={}, find="", replace=""):
        for key, value in tag_stack.items():
            if find in value:
                tag_stack[key] = value.replace(find, replace)

        return (tag_stack,)
