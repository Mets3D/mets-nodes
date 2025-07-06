import re
from typing import List, Tuple

class RegexProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "pattern": ("STRING", {"multiline": False}),
                "operation": (
                    ["match", "findall", "substitute", "split"],
                ),
            },
            "optional": {
                "replacement": ("STRING", {"multiline": True}),
                "flags": (
                    ["IGNORECASE", "MULTILINE", "DOTALL", "VERBOSE"],
                    {"forceInput": True, "multiselect": True},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "LIST")
    RETURN_NAMES = ("output_text", "matches")
    FUNCTION = "run"
    CATEGORY = "Text/Regex"

    def _get_flags(self, flag_list: List[str]) -> int:
        flag_map = {
            "IGNORECASE": re.IGNORECASE,
            "MULTILINE": re.MULTILINE,
            "DOTALL": re.DOTALL,
            "VERBOSE": re.VERBOSE,
        }
        return sum(flag_map.get(f, 0) for f in (flag_list or []))

    def run(
        self,
        text: str,
        pattern: str,
        operation: str,
        replacement: str = "",
        flags: List[str] = None,
    ) -> Tuple[str, List[str]]:
        compiled = re.compile(pattern, self._get_flags(flags))

        matches = []
        output = ""

        try:
            if operation == "match":
                m = compiled.search(text)
                output = m.group(0) if m else ""
                matches = [m.group(0)] if m else []

            elif operation == "findall":
                matches = compiled.findall(text)
                matches = [str(m) for m in matches]
                output = ", ".join(matches)

            elif operation == "substitute":
                output = compiled.sub(replacement, text)
                matches = compiled.findall(text)
                matches = [str(m) for m in matches]

            elif operation == "split":
                split_parts = compiled.split(text)
                matches = split_parts
                output = ", ".join(split_parts)

        except re.error as e:
            output = f"Regex error: {str(e)}"
            matches = []

        return (output, matches)
