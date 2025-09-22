from .regex_nodes import (
    RegexNode, ExtractTagFromString, AutoExtractTags, ChainReplace,
    StableRandomChoiceNode, PromptTidy, 
)
from .model_downloader import DownloadCivitaiModel
from .mega_prompt import MegaPrompt, MetContextBreak, MetFaceContextBreak, MetPrepareCheckpoint, TagStacker

NODE_CLASS_MAPPINGS = {
    "RegexNode": RegexNode,
    "ExtractTagFromString": ExtractTagFromString,
    "AutoExtractTags": AutoExtractTags,
    "ChainReplace": ChainReplace,
    "StableRandomChoiceNode": StableRandomChoiceNode,
    "PromptTidy": PromptTidy,

    "DownloadCivitaiModel": DownloadCivitaiModel,

    "MegaPrompt": MegaPrompt,
    "ContextBreak": MetContextBreak,
    "FaceContextBreak": MetFaceContextBreak,
    "PrepareCheckpoint": MetPrepareCheckpoint,
    "TagStacker": TagStacker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RegexNode": "Regex Operations",
    "ExtractTagFromString": "Extract Tag From String",
    "AutoExtractTags": "Auto Extract Tags From String",
    "StableRandomChoiceNode": "Random Choice",
    "PromptTidy": "Tidy Prompt",

    "DownloadCivitaiModel": "Download CivitAI Model",

    "MegaPrompt": "Mega Prompt",
    "ContextBreak": "Context Break",
    "FaceContextBreak": "Face Context Break",
    "PrepareCheckpoint": "Prepare Checkpoint",
    "TagStacker": "Tag Stacker",
}
