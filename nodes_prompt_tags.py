import re, random
from typing import Tuple

class RegexNode:
    NAME = "Regex Operations"

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
    CATEGORY = "Met's Nodes/Prompt Tags"

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

class ExtractTagFromString:
    NAME = "Extract Tag From String"
    DESCRIPTION = (
        "Extract a specified <tag> from a string. Examples:\n"
        "Put <neg>bad quality</neg> into your prompt, and then use this node to extract it and feed it to your negative prompt.\n"
        "Or use <face>blue eyes</face> to feed that into your face detailer. In this case, you would use preserve_tag_content."
    )
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("without_tags", "without_tag_contents", "tag_content")
    FUNCTION = "extract"
    CATEGORY = "Met's Nodes/Prompt Tags"

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
        without_tags, without_content, content = extract_tag_from_text(text, tag)
        text = without_tags if preserve_tag_content else without_content
        return text.strip(), content

def extract_tag_from_text(text: str, tag: str) -> tuple[str, str, str]:
    """
    Removes <tag>...</tag> blocks from `text`. Returns:
    - Input text with only the tags themselves removed, but not the content inside the tags.
    - Input text with the tags and their contents removed
    - The contents of the tags (comma-joined)
    """
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"

    contents = []
    without_contents = text

    # Extract tag contents.
    while close_tag in without_contents:
        before, after = without_contents.split(close_tag, 1)
        split = before.rsplit(open_tag, 1)
        if len(split) == 1:
            outside, inside = split[0], ""
            print(f"Missing opening {tag} tag:  {before}")
        else:
            outside, inside = split
        contents.append(inside)
        without_contents = outside+","+after

    pattern = fr"<[/|!]?{tag}>"
    without_tags = re.sub(pattern, ",", text, flags=re.DOTALL)
    without_contents = re.sub(pattern, ",", without_contents, flags=re.DOTALL)

    return tidy_prompt(without_tags), tidy_prompt(without_contents), tidy_prompt(",".join(contents))

class AutoExtractTags:
    NAME="Auto Extract Tags From String"
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
    CATEGORY = "Met's Nodes/Prompt Tags"
    DESCRIPTION = ("Automatically remove all tags, as well as the contents of those tags which have a <!tag> present.\n"
    "For example, if you want to randomize facial expressions, you could use this node after a prompt processor, like this:\n"
    "<eye>blue eyes</eye>, {eyes open|winking, one eye open|eyes closed, blink, <!eye>}\n"
    "In this case, when the randomizer chooses blink, blue eyes will be removed from the prompt by this node."
    )

    def auto_extract(self, text: str) -> Tuple[str, str]:
        return auto_extract_tags(text)

def auto_extract_tags(text: str) -> tuple[str, str]:
    # Find all <!tag> instructions
    marker_pattern = r"<!((?:\w|\s)+)>"
    tags_marked_for_removal = set(re.findall(marker_pattern, text))
    text = re.sub(marker_pattern, "", text)

    # Detect all tag names present
    all_tag_names = set(re.findall(r"<\/((?:\w|\s)+)>", text))

    clean_text = text
    tag_contents = []
    for tag in all_tag_names:
        do_remove_content = tag in tags_marked_for_removal
        # Extract tag content and remove the tags themselves, no matter what.
        clean_text, content = extract_tag_from_text(
            clean_text, tag, remove_content=do_remove_content
        )
        # Remove tag content, only if it's marked for exclusion.
        if do_remove_content and content:
            tag_contents.append(content)

    return clean_text.strip(), ", ".join(tag_contents)

class StableRandomChoiceNode:
    NAME="Random Choice"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"description": "Input string with {option1|option2|...} groups, supports nesting."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1, "description": "Base seed for stable randomness."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "randomize_prompt"
    CATEGORY = "Met's Nodes/Prompt Tags"
    DESCRIPTION="""Processes strings with nested {option1|option2|...} syntax.
    It selects one random option per group using a stable seed and a counter-based RNG,
    ensuring consistent output even if unrelated parts of the input change."""

    def randomize_prompt(self, prompt, seed=0):
        randomized = randomize_prompt(prompt, seed)
        return (randomized, )

def randomize_prompt(prompt, seed=0) -> str:
    """
    Process the input text, recursively replacing each {...|...} group
    with one option selected randomly and stably based on the seed.

    Args:
        text (str): Input string with nested option groups.
        seed (int): Base seed to control randomness and ensure reproducibility.

    Returns:
        tuple(str): Processed string with all groups resolved.
    """
    counter = 0  # Counts number of choices made so far, used in seed.

    pattern = re.compile(r'\{([^{}]*)\}')
    def helper(t):
        nonlocal counter
        match = pattern.search(t)
        if not match:
            return t

        options = match.group(1).split('|')
        choice_seed = seed + counter
        rand = random.Random(choice_seed)
        choice = rand.choice(options)

        counter += 1
        t = t[:match.start()] + choice + t[match.end():]
        return helper(t)

    prompt = helper(prompt)
    return tidy_prompt(prompt)

class PromptTidy:
    NAME="Tidy Prompt"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"description": "Input prompt."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "tidy_prompt"
    CATEGORY = "Met's Nodes/Prompt Tags"
    DESCRIPTION="""Remove excess commas, newlines, whitespaces from a prompt-style string."""

    def tidy_prompt(self, prompt):
        return (tidy_prompt(prompt), )

def tidy_prompt(prompt: str) -> str:
    prompt = remove_comment_lines(prompt)
    lines = [line.strip() for line in prompt.split("\n")]
    clean_lines = []

    for line in lines:
        # Clean up any unnecessary commas
        words = [word.strip() for word in line.split(",")]
        words = [word for word in words if word]
        if words:
            # Ensure commas at end of lines
            clean_lines.append(", ".join(words)+",")
        else:
            clean_lines.append("")
    prompt = "\n".join(clean_lines).strip()

    # Limit to 2 consecutive newlines
    while "\n\n\n" in prompt:
        prompt = prompt.replace("\n\n\n", "\n\n")

    return prompt

def remove_comment_lines(prompt: str) -> str:
    return re.sub(r"#.*", "", prompt)
