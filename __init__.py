from .nodes_prompt_tags import (
    RegexNode, ExtractTagFromString, AutoExtractTags,
    StableRandomChoiceNode, PromptTidy, 
)
from .nodes_downloader import DownloadCivitaiModel
from .nodes_databases import PrepareCheckpoint, PrepareLoRA, TagStacker, TagTweaker
from .nodes_image import AdjustImageNode
from .nodes_render_pass import RenderPass, RenderPass_Prepare, RenderPass_Face, SplitData

nodes = [
    RegexNode, ExtractTagFromString, AutoExtractTags, StableRandomChoiceNode, PromptTidy, 
    DownloadCivitaiModel, PrepareCheckpoint, PrepareLoRA, TagStacker, TagTweaker, AdjustImageNode,
    RenderPass, RenderPass_Prepare, RenderPass_Face, SplitData
]

NODE_CLASS_MAPPINGS = {node.__name__: node for node in nodes}

NODE_DISPLAY_NAME_MAPPINGS = {node.__name__: node.NAME for node in nodes}
