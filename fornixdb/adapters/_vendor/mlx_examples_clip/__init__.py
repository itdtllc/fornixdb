"""Vendored CLIP image/text tower from Apple's mlx-examples (MIT).

Only the vision path is used by fornixdb's watch loop (mac_vision). Vendored
rather than pip-depended so the `[mac]` extra stays a small, permissively
licensed set (mlx + numpy + pillow) with no PyPI package that drags in a heavy
tree. Upstream + LICENSE are documented in the files here and in LICENSE.
"""
from .image_processor import CLIPImageProcessor
from .model import CLIPModel

__all__ = ["CLIPModel", "CLIPImageProcessor"]
