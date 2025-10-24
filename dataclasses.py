import os
from dataclasses import dataclass

@dataclass
class CivitAIModelConfigBase:
    civitai_model_id: int
    path: str   # Relative to whatever base folder this type of file should have. So this does not include "checkpoints/" or "loras/".

    version: str

    @property
    def name(self) -> str:
        return self.path.split(os.sep)[-1]

    @property
    def subdir(self) -> str:
        return self.path.rsplit(os.sep, 1)[0]

    @property
    def name(self) -> str:
        return self.path.split(os.sep)[-1]
    
    @property
    def name_noext(self) -> str:
        return self.name.split(".")[0]

    @property
    def civitai_url(self) -> str:
        return f"https://civitai.com/models/{self.civitai_model_id}"

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

@dataclass
class CheckpointConfig(CivitAIModelConfigBase):
    """Checkpoint data: The checkpoint's name, and the settings that give the best results with a given checkpoint."""
    steps: int
    cfg: float
    sampler: str
    scheduler: str = "normal"
    model_pos_prompt: str = ""
    model_neg_prompt: str = ""
    clip_skip: int = -2


@dataclass
class LoRAConfig(CivitAIModelConfigBase):
    pass
