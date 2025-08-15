import re, random
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
    CATEGORY = "MetsNodes"

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
    CATEGORY = "MetsNodes/Tag Tools"

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
    CATEGORY = "MetsNodes/Tag Tools"

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
    CATEGORY = "MetsNodes/Tag Tools"
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

class StableRandomChoiceNode:
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
    CATEGORY = "MetsNodes"
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

    prompt = prettify_prompt(prompt)
    prompt = helper(prompt)
    return prompt

def pretty_format(prompt, indent_str="  "):
    result = []
    depth = 0
    i = 0
    length = len(prompt)

    while i < length:
        c = prompt[i]

        if c == '{':
            result.append('{\n')
            depth += 1
            result.append(indent_str * depth)
            i += 1
        elif c == '}':
            depth -= 1
            result.append('\n' + indent_str * depth + '}')
            i += 1
            # Peek next non-space char to decide whether to add newline+indent or not
            j = i
            while j < length and prompt[j] == ' ':
                j += 1
            if j < length and prompt[j] in ('{', '|'):
                # Add newline+indent before next block or option
                result.append('\n' + indent_str * depth)
            else:
                # Just continue without adding newline (so trailing spaces/text stay inline)
                pass
        elif c == '|':
            result.append('|\n' + indent_str * depth)
            i += 1
        else:
            result.append(c)
            i += 1

    return ''.join(result)

def sanitize_prompt(prompt: str) -> str:
    """
    Cleans and joins a multiline prompt into a well-formatted, comma-separated string.
    - Adds commas at the end of each line unless the line ends with { or (
    - Removes junk commas (like before |, }, or ))
    - Collapses excessive whitespace and empty lines
    """
    lines = prompt.splitlines()
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            continue

        # If it ends with { or (, leave it alone
        if re.search(r'[{(]$', stripped):
            cleaned_lines.append(stripped)
        else:
            # Remove trailing commas, then add one cleanly
            stripped = re.sub(r',+$', '', stripped)
            cleaned_lines.append(f"{stripped},")

    # Join with spaces or commas depending on your preference
    joined = ' '.join(cleaned_lines)

    # Remove commas before special closing symbols like }, ), |
    joined = re.sub(r',\s*([}\)|])', r'\1', joined)

    # Collapse multiple commas
    joined = re.sub(r',\s*,+', ',', joined)

    # Final trim
    return joined.strip(' ,')

def prettify_prompt(prompt: str) -> str:
    prompt = re.sub(r"#.*", "", prompt)
    prompt = sanitize_prompt(prompt)
    return pretty_format(prompt)

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