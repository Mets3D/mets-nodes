from .regex_nodes import (
    RegexNode, ExtractTagFromString, AutoExtractTags, ChainReplace,
    StableRandomChoiceNode, PromptTidy, 
)
from .model_downloader import DownloadCivitaiModel

NODE_CLASS_MAPPINGS = {
    "RegexNode": RegexNode,
    "ExtractTagFromString": ExtractTagFromString,
    "AutoExtractTags": AutoExtractTags,
    "ChainReplace": ChainReplace,
    "StableRandomChoiceNode": StableRandomChoiceNode,
    "PromptTidy": PromptTidy,

    "DownloadCivitaiModel": DownloadCivitaiModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RegexNode": "Regex Operations",
    "ExtractTagFromString": "Extract Tag From String",
    "AutoExtractTags": "Auto Extract Tags From String",
    "StableRandomChoiceNode": "Random Choice",
    "PromptTidy": "Tidy Prompt",

    "DownloadCivitaiModel": "Download CivitAI Model",
}
