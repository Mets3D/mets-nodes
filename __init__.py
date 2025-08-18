from .regex_nodes import (
    RegexNode, ExtractTagFromString, AutoExtractTags, ChainReplace,
    StableRandomChoiceNode,
)
from .model_downloader import DownloadCivitaiModel

NODE_CLASS_MAPPINGS = {
    "RegexNode": RegexNode,
    "ExtractTagFromString": ExtractTagFromString,
    "AutoExtractTags": AutoExtractTags,
    "ChainReplace": ChainReplace,
    "StableRandomChoiceNode": StableRandomChoiceNode,

    "DownloadCivitaiModel": DownloadCivitaiModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RegexNode": "Regex Operations",
    "ExtractTagFromString": "Extract Tag From String",
    "AutoExtractTags": "Auto Extract Tags From String",
    "StableRandomChoiceNode": "Random Choice",

    "DownloadCivitaiModel": "Download CivitAI Model",
}
