import os
from dataclasses import dataclass, field

COMMON_POSITIVE = "highly detailed, great lighting, atmospheric, illustration, 4k, masterpiece, best quality, detailed background, beautiful composition, artistic, HDR, high dynamic range, "
COMMON_NEGATIVE = "ugly, boring, empty, plain, colorless, gray, lowres, worst quality, low quality, bad quality, bad hands, sketch, jpeg artifacts, signature, watermark, text, old, oldest, censored, bad hands, patreon, flat eyes, blank stare, dead eyes, noisy eyes, complex eyes, overexposed sky, clipping, "

@dataclass
class MetCheckpointPreset:
    """Checkpoint data: The checkpoint's name, and the settings that give the best results with a given checkpoint."""
    civitai_model_id: int
    path: str

    steps: int
    cfg: float
    sampler: str
    scheduler: str = "normal"
    model_pos_prompt: str = COMMON_POSITIVE
    model_neg_prompt: str = COMMON_NEGATIVE
    clip_skip: int = -2

    @property
    def name(self) -> str:
        return self.path.split(os.sep)[-1]
    civitai_version_name: str = ""

@dataclass
class MetContext:
    """This class handles only primitive python data representing a complete rendering environment 
    for a single AI image rendering pass, to be used in ComfyUI."""
    checkpoint: MetCheckpointPreset

    ### Prompt data.
    pos_prompt: str="cute girl"
    neg_prompt: str=""
    noise_seed: int = -1
    prompt_seed: int = -1
    loras: list[str] = field(default_factory=list)
    use_model_pos_prompt: bool = True
    use_model_neg_prompt: bool = True

    # Render settings.
    width: int = 1024
    height: int = 1024
    scale: float = 1
    add_noise: float = 0.5

@dataclass
class MetFaceContext:
    """Minimal configuration for FaceDetailer."""
    checkpoint: MetCheckpointPreset

    face_iterations: int = 1
    face_noise_amount: float = 0.32

    pos_prompt: str = "detailed face"
    neg_prompt: str = ""
    noise_seed: int = -1
    loras: list[str] = field(default_factory=list)
