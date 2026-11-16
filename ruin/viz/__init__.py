"""RUIN visualization layer (v0.2+).

Currently provides:
- Text frame export (legacy)
- Matplotlib GIF animation of the Probability Square + Disruption Field
"""

from .square import animate_probability_square, save_field_gif, save_text_frames

__all__ = ["animate_probability_square", "save_field_gif", "save_text_frames"]
