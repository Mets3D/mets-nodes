from .regex_nodes import (
    RegexNode, ExtractTagFromString, AutoExtractTags, ChainReplace,
    StableRandomChoiceNode, PromptTidy, 
)
from .model_downloader import DownloadCivitaiModel
from .mega_prompt import MegaPrompt, ContextBreak, FaceContextBreak, PrepareCheckpoint, TagStacker, TagTweaker
from .image_adjust import AdjustImageNode

nodes = [
    RegexNode, ExtractTagFromString, AutoExtractTags, ChainReplace, StableRandomChoiceNode, PromptTidy, 
    DownloadCivitaiModel, MegaPrompt, ContextBreak, FaceContextBreak, PrepareCheckpoint, TagStacker, TagTweaker, AdjustImageNode
]

NODE_CLASS_MAPPINGS = {node.__name__: node for node in nodes}

NODE_DISPLAY_NAME_MAPPINGS = {node.__name__: node.NAME for node in nodes}
