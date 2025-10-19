from .regex_nodes import (
    RegexNode, ExtractTagFromString, AutoExtractTags, ChainReplace,
    StableRandomChoiceNode, PromptTidy, 
)
from .model_downloader import DownloadCivitaiModel
from .mega_prompt import ContextBreak, FaceContextBreak, PrepareCheckpoint, PrepareLoRA, TagStacker, TagTweaker
from .image_adjust import AdjustImageNode
from .render_pass_node import RenderPass, RenderPass_Prepare, RenderPass_Face

nodes = [
    RegexNode, ExtractTagFromString, AutoExtractTags, ChainReplace, StableRandomChoiceNode, PromptTidy, 
    DownloadCivitaiModel, ContextBreak, FaceContextBreak, PrepareCheckpoint, PrepareLoRA, TagStacker, TagTweaker, AdjustImageNode,
    RenderPass, RenderPass_Prepare, RenderPass_Face
]

NODE_CLASS_MAPPINGS = {node.__name__: node for node in nodes}

NODE_DISPLAY_NAME_MAPPINGS = {node.__name__: node.NAME for node in nodes}
