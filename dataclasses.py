import os
from dataclasses import dataclass

@dataclass
class CheckpointConfig:
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
class LoRAConfig:
    civitai_model_id: int
    version: str
    path: str

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
