from .nodes_string import (
    RegexNode, ExtractTagFromString, AutoExtractTags,
    StableRandomChoiceNode, PromptTidy, StringSplit,
)
from .nodes_downloader import DownloadCivitaiModel
from .nodes_databases import PrepareCheckpoint, PrepareLoRA, TagStacker, TagTweaker, ExtraCheckpointData
from .nodes_image import AdjustImageNode, LoadImageFromDirectory, RescaleToPixelCount
from .nodes_render_pass import RenderPass, RenderPass_Prepare, RenderPass_Face, SplitData, MergeData
from .nodes_wan22 import Wan22DualLoRA, Wan22DualModel, Wan22LoRAStacker, Wan22Render

WEB_DIRECTORY = "./web"

nodes = [
    RegexNode, ExtractTagFromString, AutoExtractTags, StableRandomChoiceNode, PromptTidy,
    DownloadCivitaiModel, PrepareCheckpoint, PrepareLoRA, TagStacker, TagTweaker, AdjustImageNode,
    RenderPass, RenderPass_Prepare, RenderPass_Face, SplitData, ExtraCheckpointData, MergeData,
    Wan22DualLoRA, Wan22DualModel, Wan22LoRAStacker, Wan22Render,
    LoadImageFromDirectory, RescaleToPixelCount, StringSplit,
]

NODE_CLASS_MAPPINGS = {node.__name__: node for node in nodes}

NODE_DISPLAY_NAME_MAPPINGS = {node.__name__: node.NAME for node in nodes}
