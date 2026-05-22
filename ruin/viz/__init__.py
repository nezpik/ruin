"""RUIN visualization layer (v0.2+).

Currently provides:
- Text frame export (legacy)
- Matplotlib GIF animation of the Probability Square + Disruption Field
"""

from .square import save_text_frames, save_field_gif, animate_probability_square

__all__ = ["save_text_frames", "save_field_gif", "animate_probability_square"]