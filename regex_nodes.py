import re
from typing import Tuple

class RegexNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_text": ("STRING", {"default": "", "multiline": True}),
                "pattern": ("STRING", {"default": ""}),
                "replace_with": ("STRING", {"default": ""}),
            },
            "optional": {
                "mode": (["search", "replace", "findall"], {"default": "search"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute_regex"
    CATEGORY = "text"

    def execute_regex(self, input_text, pattern, replace_with="", mode="search") -> Tuple[str]:
        try:
            if mode == "search":
                match = re.search(pattern, input_text)
                return (match.group(0) if match else "",)
            
            elif mode == "replace":
                result = re.sub(pattern, replace_with, input_text)
                return (result,)
            
            elif mode == "findall":
                matches = re.findall(pattern, input_text)
                return ("\n".join(matches) if matches else "",)
            
        except re.error as e:
            print(f"RegEx Error: {str(e)}")
            return ("",)
        except Exception as e:
            print(f"Non-Regex Error: {str(e)}")
            return ("",)

class ChainReplace:
    DESCRIPTION = ("A simple string search and replace operation that is designed to nicely chain together. Can be used to build complex randomized prompts.")
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string_replaced",)
    FUNCTION = "replace_string"
    CATEGORY = "Text/Tag Tools"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": False, "tooltip": "The string to search and replace within."}),
                "replace_with": ("STRING", {"multiline": True, "tooltip": "The string which will be substituted to the input."}),
                "replaced_by": ("STRING", {"multiline": False, "tooltip": "The specific tag to remove."}),
            }
        }

    def replace_string(self, text: str, replace_with: str, replaced_by: str) -> Tuple[str]:
        return (text.replace(replaced_by, replace_with),)

class ExtractTagFromString:
    DESCRIPTION = (
        "Extract a specified <tag> from a string. Examples:\n"
        "Put <neg>bad quality</neg> into your prompt, and then use this node to extract it and feed it to your negative prompt.\n"
        "Or use <face>blue eyes</face> to feed that into your face detailer. In this case, you would use preserve_tag_content."
    )
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("clean_text", "tag_content")
    FUNCTION = "extract"
    CATEGORY = "Text/Tag Tools"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "tooltip": "The string which will have tags removed from it."}),
                "tag": ("STRING", {"multiline": False, "tooltip": "The specific tag to remove."}),
                "preserve_tag_content": ("BOOLEAN", {"default": False, "tooltip": "Whether the contents of the tag should be included in the clean_text output."}),
            }
        }

    def extract(self, text: str, tag: str, preserve_tag_content: bool = False) -> Tuple[str, str]:
        clean_text, content = extract_tag_from_text(text, tag, remove_content=(not preserve_tag_content))
        return clean_text.strip(), content

class AutoExtractTags:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("clean_text", "tag_content")
    FUNCTION = "auto_extract"
    CATEGORY = "Text/Tag Tools"
    DESCRIPTION = ("Automatically remove all tags, as well as the contents of those tags which have a <!tag> present.\n"
    "For example, if you want to randomize facial expressions, you could use this node after a prompt processor, like this:\n"
    "<eye>blue eyes</eye>, {eyes open|winking, one eye open|eyes closed, blink, <!eye>}\n"
    "In this case, when the randomizer chooses blink, blue eyes will be removed from the prompt by this node."
    )

    def auto_extract(self, text: str) -> Tuple[str, str]:
        import re

        # Find all <!tag> directives
        marker_pattern = r"<!((?:\w|\s)+)>"
        tags_remove_content = set(re.findall(marker_pattern, text))
        text = re.sub(marker_pattern, "", text)

        # Detect all tag names present
        all_tag_names = set(re.findall(r"<\/((?:\w|\s)+)>", text))

        # Start processing
        clean_text = text
        all_contents = []

        for tag in all_tag_names:
            remove_content = tag in tags_remove_content
            # Remove content if it's explicitly marked, else preserve it
            clean_text, content = extract_tag_from_text(
                clean_text, tag, remove_content=remove_content
            )
            if remove_content and content:
                all_contents.append(content)

        return clean_text.strip(), ", ".join(all_contents)

def extract_tag_from_text(
    text: str,
    tag: str,
    remove_content: bool = False
) -> Tuple[str, str]:
    """
    Removes <tag>...</tag> blocks from `text`.
    
    - Always removes the <tag> and </tag> markers.
    - If remove_content=True, also removes the content inside the tags from clean_text.
    - Always returns the tag contents as a comma-joined string.
    """
    pattern = fr"<{tag}>(.*?)<\/{tag}>"
    matches = re.findall(pattern, text, re.DOTALL)
    tag_content = ", ".join(m.strip() for m in matches if m.strip())

    if remove_content:
        # Remove entire <tag>...</tag>
        clean_text = re.sub(pattern, "", text, flags=re.DOTALL)
    else:
        # Replace with inner content only (strip tags)
        clean_text = re.sub(pattern, lambda m: m.group(1), text, flags=re.DOTALL)

    return clean_text, tag_content