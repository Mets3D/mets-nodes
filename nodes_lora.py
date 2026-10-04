import difflib
import re

import folder_paths

# One LoRA per line: "name", "name: 0.8" or "name: 0.8, 0.5" (model strength, clip strength).
# The name may omit the directory and the extension. "#" starts a comment.
_LINE_RE = re.compile(
    r"^(?P<name>[^:]+?)\s*(?::\s*(?P<model>[-+]?\d*\.?\d+)\s*(?:,\s*(?P<clip>[-+]?\d*\.?\d+))?)?\s*$"
)
_EXTENSIONS = (".safetensors", ".ckpt", ".pt", ".pth", ".bin")


def _strip_ext(path: str) -> str:
    for ext in _EXTENSIONS:
        if path.endswith(ext):
            return path[: -len(ext)]
    return path


def _norm(path: str) -> str:
    return path.replace("\\", "/").strip().lower()


def _resolve_lora(name: str, available: list[str]) -> str:
    """Map a user-written name to exactly one entry of the LoRA file list, or raise."""
    wanted = _norm(name)
    wanted_stem = _strip_ext(wanted)
    has_dir = "/" in wanted

    def key(entry: str) -> str:
        n = _norm(entry)
        return n if has_dir else n.rsplit("/", 1)[-1]

    matches = [e for e in available if key(e) == wanted or _strip_ext(key(e)) == wanted_stem]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        options = "\n  ".join(sorted(matches))
        raise ValueError(f"LoRA '{name}' is ambiguous - add its directory. Matches:\n  {options}")
    close = difflib.get_close_matches(wanted_stem, [_strip_ext(key(e)) for e in available], n=5, cutoff=0.6)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    raise ValueError(f"LoRA '{name}' not found in the loras folder.{hint}")


def parse_lora_lines(text: str, available: list[str]) -> list[tuple[str, float, float]]:
    """Parse the node's text into (lora_file, model_strength, clip_strength), validating every line."""
    entries = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = _LINE_RE.match(line)
        if not match:
            raise ValueError(f"Line {number}: can't parse {raw.strip()!r} (expected 'name', 'name: 0.8' or 'name: 0.8, 0.5')")
        model = float(match["model"]) if match["model"] else 1.0
        clip = float(match["clip"]) if match["clip"] else model
        entries.append((_resolve_lora(match["name"], available), model, clip))
    return entries


class LoraTagStack:
    NAME = "LoRA Tag Stack"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loras": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "One LoRA per line: 'name', 'name: 0.8' or 'name: 0.8, 0.5' (model, clip strength). "
                               "Directory and extension are optional. '#' comments a line out. "
                               "Unknown or ambiguous names are an error.",
                }),
            },
            "optional": {
                "optional_lora_stack": ("LORA_STACK", {"tooltip": "Stack to append to, e.g. from another stacker."}),
            },
        }

    RETURN_TYPES = ("LORA_STACK", "STRING")
    RETURN_NAMES = ("lora_stack", "info")
    OUTPUT_TOOLTIPS = (
        "LoRA stack in the comfyui-easy-use format: a list of (lora_name, model_strength, clip_strength).",
        "The resolved stack, one 'file: model, clip' per line.",
    )
    FUNCTION = "stack"
    CATEGORY = "Met's Nodes/LoRA"
    DESCRIPTION = "Builds a LoRA stack from text, so the whole stack is a single editable value."

    def stack(self, loras: str, optional_lora_stack=None):
        stack = [entry for entry in (optional_lora_stack or []) if entry[0] != "None"]
        for file, model, clip in parse_lora_lines(loras, folder_paths.get_filename_list("loras")):
            if model == 0 and clip == 0:
                continue
            stack.append((file, model, clip))
        info = "\n".join(f"{file}: {model:g}, {clip:g}" for file, model, clip in stack)
        return (stack, info)
