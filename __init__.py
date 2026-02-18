"""
Nano Banana Unified — ComfyUI custom nodes for Google AI image generation.
"""

from .nodes.gemini import NanoBananaGemini
from .nodes.imagen import NanoBananaImagen

NODE_CLASS_MAPPINGS = {
    "NanoBananaGemini": NanoBananaGemini,
    "NanoBananaImagen": NanoBananaImagen,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaGemini": "Nano Banana Gemini",
    "NanoBananaImagen": "Nano Banana Imagen",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
