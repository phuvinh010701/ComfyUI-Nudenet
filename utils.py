from typing_extensions import TypedDict
from onnxruntime import InferenceSession
import numpy as np
import cv2


class ModelLoader(TypedDict):
    session: InferenceSession
    input_width: int
    input_height: int
    input_name: str


def pixelate(image: np.ndarray, blocks: int = 3) -> np.ndarray:
    (h, w) = image.shape[:2]
    x_steps = np.linspace(0, w, blocks + 1, dtype="int")
    y_steps = np.linspace(0, h, blocks + 1, dtype="int")

    for i in range(1, len(y_steps)):
        for j in range(1, len(x_steps)):
            start_x = x_steps[j - 1]
            start_y = y_steps[i - 1]
            end_x = x_steps[j]
            end_y = y_steps[i]

            roi = image[start_y:end_y, start_x:end_x]
            mean_color = np.mean(roi, axis=(0, 1))
            image[start_y:end_y, start_x:end_x] = mean_color

    return image


def overlay(background, foreground, alpha_mask, x_offset=None, y_offset=None, overlay_strength=3.0):
    if background.dtype != np.float32:
        background = background.astype(np.float32)
    if foreground.dtype != np.float32:
        foreground = foreground.astype(np.float32)

    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    if bg_channels != 3 and bg_channels != 4:
        return background

    if fg_channels < 4:
        foreground = cv2.cvtColor(foreground, cv2.COLOR_RGB2RGBA)

    fg_h, fg_w, fg_channels = foreground.shape

    if x_offset is None:
        x_offset = (bg_w - fg_w) // 2
    if y_offset is None:
        y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1:
        return background

    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, -x_offset)
    fg_y = max(0, -y_offset)

    foreground = foreground[fg_y : fg_y + h, fg_x : fg_x + w]
    background_subsection = background[bg_y : bg_y + h, bg_x : bg_x + w]

    foreground_colors = foreground[:, :, :3]
    alpha_mask = alpha_mask[:, :, np.newaxis]

    composite = background_subsection * (
        alpha_mask * overlay_strength
    ) + foreground_colors * ((1 - alpha_mask) * overlay_strength)

    background[bg_y : bg_y + h, bg_x : bg_x + w] = composite

    return background
