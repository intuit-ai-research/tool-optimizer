"""Shared console output helpers for inference (two-row layout, colored)."""

import shutil
import textwrap

_COLOR_BLUE = "\033[94m"
_COLOR_RESET = "\033[0m"

SPACE_BETWEEN_INFERENCES = 2  # blank lines between each inference


def print_inference_output(user_prompt: str, model_response: str) -> None:
    """Print user prompt and model response in two rows, both in blue, to console."""
    try:
        width = shutil.get_terminal_size().columns
        line_width = max(40, width - 2)
    except OSError:
        line_width = 78

    print(f"{_COLOR_BLUE}User prompt{_COLOR_RESET}")
    print("-" * min(40, line_width))
    for line in textwrap.wrap(user_prompt, width=line_width) or [""]:
        print(f"{_COLOR_BLUE}{line}{_COLOR_RESET}")
    print()
    print(f"{_COLOR_BLUE}Model response{_COLOR_RESET}")
    print("-" * min(40, line_width))
    for line in textwrap.wrap(model_response, width=line_width) or [""]:
        print(f"{_COLOR_BLUE}{line}{_COLOR_RESET}")
