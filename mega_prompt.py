from .met_context import MetCheckpointPreset, MetContext, MetFaceContext, CHECKPOINTS
import folder_paths
import comfy.samplers
# NOTE: Requires Impact Pack, sadly.
import impact.core as core
import random
import re

class MegaPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_pos": ("STRING", {"multiline": True, "tooltip": "The base positive prompt.", "name": "+Prompt"}),
                "prompt_neg": ("STRING", {"multiline": True, "tooltip": "The base negative prompt.", "name": "-Prompt"}),
                "prompt_seed": ("INT", {"control_after_generate": True, "tooltip": "Prompt randomization seed.", "name": "Prompt Seed"}),
                "noise_seed": ("INT", {"control_after_generate": True, "tooltip": "Noise seed.", "name": "Noise Seed"}),
                "width": ("INT", {"tooltip": "Image width.", "name": "Width", "default": 1024, "min": 0, "max": 2**16}),
                "height": ("INT", {"tooltip": "Image height.", "name": "Height", "default": 1024, "min": 0, "max": 2**16}),
            },
            "optional": {
                "checkpoint_datas": ("CHECKPOINT_DATAS", {"name": "Checkpoint Datas"})
            }
        }

    RETURN_NAMES = ("Context 1", "Context 2", "Context 3", "Context 4", "FaceDetailer Context")
    RETURN_TYPES = ("METCONTEXT","METCONTEXT","METCONTEXT","METCONTEXT","METFACECONTEXT")
    FUNCTION = "mega_prompt"
    CATEGORY = "MetsNodes"
    DESCRIPTION="""Process the mega prompt into 4 consecutive rendering contexts."""

    def mega_prompt(self, prompt_pos, prompt_neg, prompt_seed, noise_seed, width, height, checkpoint_datas) -> tuple[MetContext, MetContext, MetContext, MetContext, MetFaceContext]:
        cp_pattern = re.compile(r"<checkpoint:([^>]+)>")

        match = cp_pattern.search(prompt_pos)
        checkpoint_name = match.group(1) if match else ""
        prompt_pos = cp_pattern.sub("", prompt_pos)
        checkpoint = checkpoint_datas.get(checkpoint_name)

        context_1 = MetContext(
            checkpoint=checkpoint,
            noise_seed=noise_seed if noise_seed <= 0 else random.randint(1, 100000000),
            prompt_seed=prompt_seed,
            width=width,
            height=height,
            pos_prompt=prompt_pos,
            neg_prompt=prompt_neg,
        )
        context_4 = context_3 = context_2 = context_1
        context_2.scale = 1.5
        face_context = MetFaceContext(
            checkpoint=context_1.checkpoint,
            face_iterations=1,
            face_noise_amount=0.32,
            pos_prompt="detailed face",
            neg_prompt="wrinkles, eye bags",
            noise_seed=context_1.noise_seed,
            prompt_seed=context_1.prompt_seed,
            loras=context_1.loras,
        )
        return (context_1, context_2, context_3, context_4, face_context)

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
            Context.checkpoint.path+".safetensors",
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
            FaceContext.checkpoint.path+".safetensors",
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
            "required": {
                "name": ("STRING", {"tooltip": "Filename of .safetensors file (without extension)", "name": "Name"}),
                "path": ("STRING", {"tooltip": "Filepath relative to checkpoints folder, excluding filename", "name": "Directory"}),
                "civitai_id": ("INT", {"tooltip": f"CivitAI model ID. Can be found in the model's page URL: civitai.com/models/<model id>", "min": 0, "max": 100000000, "name": "CivitAI ID"}),
                "version_name": ("STRING", {"tooltip": f"CivitAI model version name. If not specified, we assume the left-most version shown in the horizontal list on the model's page", "default":"", "name": "Version Name"}),
                "clip_skip": ("INT", {"tooltip": "CLIP Skip value for this checkpoint", "default": -2, "min":-2, "max":0, "name": "Clip Skip"}),
                "steps": ("INT", {"tooltip": "Number of denoising steps to use with this checkpoint", "default": 25, "name": "Steps"}),
                "cfg": ("FLOAT", {"tooltip": "CFG value to use with this checkpoint", "default": 4.5, "name": "CFG"}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "Sampler", "name": "Sampler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "Scheduler", "name": "Scheduler"}),
                "prompt_pos": ("STRING", {"multiline": True, "tooltip": "Base positive prompt associated with this checkpoint", "name": "Base Positive Prompt"}),
                "prompt_neg": ("STRING", {"multiline": True, "tooltip": "Base negative prompt associated with this checkpoint", "name": "Base Negative Prompt"}),
            },
            "optional": {
                "checkpoint_datas": ("CHECKPOINT_DATAS", {"name": "Checkpoint Datas"})
            }
        }

    def prepare_checkpoint(self, name, path, civitai_id, version_name, clip_skip, steps, cfg, sampler, scheduler, prompt_pos, prompt_neg, checkpoint_datas=None):
        if not checkpoint_datas:
            checkpoint_datas = {}
        checkpoint_datas.update({name: MetCheckpointPreset(
            civitai_model_id=civitai_id,
            path=path,
            steps=steps,
            cfg=cfg,
            sampler=sampler,
            scheduler=scheduler,
            model_pos_prompt=prompt_pos,
            model_neg_prompt=prompt_neg,
        )})
        return (checkpoint_datas,)