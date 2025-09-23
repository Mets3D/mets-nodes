import os
from dataclasses import dataclass, field


@dataclass
class MetCheckpointPreset:
    """Checkpoint data: The checkpoint's name, and the settings that give the best results with a given checkpoint."""
    civitai_model_id: int
    path: str

    steps: int
    cfg: float
    sampler: str
    scheduler: str = "normal"
    model_pos_prompt: str = ""
    model_neg_prompt: str = ""
    clip_skip: int = -2

    @property
    def name(self) -> str:
        return self.path.split(os.sep)[-1]
    civitai_version_name: str = ""

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

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

    # Render settings.
    width: int = 1024
    height: int = 1024
    noise: float = 0.5
    scale: float = 1.0

    brightness: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

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

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)