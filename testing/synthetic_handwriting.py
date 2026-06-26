#!/usr/bin/env python3
"""Synthetic handwriting fixtures for math whiteboard tests.

This module is intentionally standalone: it renders LaTeX or plain math text
into rough black ink contours and places one or more rendered lines on a white
board image. It does not import app, OCR, browser, or model code.
"""

from __future__ import annotations

import base64
import os
import re
import tempfile
import warnings
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "whiteboard-mpl"))
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"matplotlib\..*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"pyparsing\..*")
warnings.filterwarnings("ignore", message=r".*deprecated.*", module=r"matplotlib\..*")
warnings.filterwarnings("ignore", message=r".*deprecated.*", module=r"pyparsing\..*")
warnings.filterwarnings("ignore", message=r".*deprecated.*")

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image, ImageDraw, ImageFont


Point = Tuple[float, float]
Placement = Dict[str, Any]


@dataclass(frozen=True)
class SyntheticHandwriting:
    width: int
    height: int
    contours: List[List[Point]]
    data_url: str

    def to_json(self) -> Dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "contours": self.contours,
            "dataUrl": self.data_url,
        }


@dataclass(frozen=True)
class PlacedHandwriting:
    latex: str
    x: float
    y: float
    width: int
    height: int
    contours: List[List[Point]]
    data_url: str

    @property
    def bbox(self) -> Dict[str, float]:
        return {
            "xMin": self.x,
            "yMin": self.y,
            "xMax": self.x + self.width,
            "yMax": self.y + self.height,
        }

    def to_json(self) -> Dict[str, object]:
        return {
            "latex": self.latex,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "bbox": self.bbox,
            "contours": translate_contours(self.contours, self.x, self.y),
            "dataUrl": self.data_url,
        }


@dataclass(frozen=True)
class BoardFixture:
    width: int
    height: int
    lines: List[PlacedHandwriting]
    data_url: str

    @property
    def contours(self) -> List[List[Dict[str, float]]]:
        out: List[List[Dict[str, float]]] = []
        for line in self.lines:
            out.extend(translate_contours(line.contours, line.x, line.y))
        return out

    def to_json(self) -> Dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "lines": [line.to_json() for line in self.lines],
            "contours": self.contours,
            "dataUrl": self.data_url,
        }


def normalize_latex(latex: str) -> str:
    """Make token-spaced test LaTeX friendlier to matplotlib mathtext."""
    replacements = {
        r"\left": "",
        r"\right": "",
        r"\operatorname": r"\mathrm",
    }
    out = latex.strip()
    for old, new in replacements.items():
        out = out.replace(old, new)
    out = " ".join(out.split())
    out = out.replace("\u2032", "'")
    out = re.sub(r"([A-Za-z0-9\)])\s*'\s*", r"\1^{\\prime}", out)
    # These fixtures often use token-spaced LaTeX because OCR labels tend to be
    # token-spaced. Mathtext needs structural tokens compacted before rendering.
    out = re.sub(r"\s*([_^{}])\s*", r"\1", out)
    out = re.sub(r"(\\[A-Za-z]+)\s+(?=[_^{}])", r"\1", out)
    return out


def is_plain_math_text(latex: str) -> bool:
    return not any(token in latex for token in ("\\", "{", "}", "^", "_", "'", "\u2032"))


def handwriting_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
        "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def render_plain_text_mask(text: str, font_size: int = 82) -> np.ndarray:
    font = handwriting_font(font_size)
    normalized = " ".join(text.strip().split())
    scratch = Image.new("L", (10, 10), 255)
    draw = ImageDraw.Draw(scratch)
    bbox = draw.textbbox((0, 0), normalized, font=font)
    width = max(1, bbox[2] - bbox[0] + 34)
    height = max(1, bbox[3] - bbox[1] + 34)
    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    draw.text((17 - bbox[0], 17 - bbox[1]), normalized, fill=0, font=font)
    gray = np.asarray(image)
    return (255 - gray).astype(np.uint8)


def render_latex_mask(latex: str, font_size: int = 46, dpi: int = 180) -> np.ndarray:
    if is_plain_math_text(latex):
        return render_plain_text_mask(latex)

    expression = f"${normalize_latex(latex)}$"
    fig = plt.Figure(figsize=(1, 1), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    text = fig.text(
        0,
        0,
        expression,
        fontsize=font_size,
        color="black",
        math_fontfamily="cm",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        canvas.draw()
    bbox = text.get_window_extent(renderer=canvas.get_renderer()).expanded(1.08, 1.22)
    width = max(1, int(np.ceil(bbox.width)))
    height = max(1, int(np.ceil(bbox.height)))
    plt.close(fig)

    fig = plt.Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    fig.patch.set_facecolor("white")
    fig.text(
        0.03,
        0.5,
        expression,
        fontsize=font_size,
        color="black",
        math_fontfamily="cm",
        va="center",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    plt.close(fig)
    gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
    return (255 - gray).astype(np.uint8)


def fit_mask(mask: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    ys, xs = np.where(mask > 8)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    cropped = mask[
        max(0, ys.min() - 8) : min(mask.shape[0], ys.max() + 9),
        max(0, xs.min() - 8) : min(mask.shape[1], xs.max() + 9),
    ]
    scale = min(max_width / cropped.shape[1], max_height / cropped.shape[0], 1.0)
    if scale < 0.999:
        cropped = cv2.resize(
            cropped,
            (max(1, int(cropped.shape[1] * scale)), max(1, int(cropped.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return cropped


def elastic_ink(mask: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    noise_scale = max(7, min(w, h) // 6)
    dx = cv2.GaussianBlur(rng.normal(0, 1.0, (h, w)).astype(np.float32), (0, 0), noise_scale) * 9.0
    dy = cv2.GaussianBlur(rng.normal(0, 1.0, (h, w)).astype(np.float32), (0, 0), noise_scale) * 5.0
    warped = cv2.remap(
        mask,
        xx + dx,
        yy + dy,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    angle = float(rng.normal(-1.5, 1.8))
    shear = float(rng.normal(-0.035, 0.025))
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    matrix[0, 1] += shear
    warped = cv2.warpAffine(
        warped,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    if seed % 2:
        warped = cv2.dilate(warped, kernel, iterations=1)
    else:
        warped = cv2.morphologyEx(warped, cv2.MORPH_CLOSE, kernel, iterations=1)
    paper = rng.normal(0, 5, (h, w)).astype(np.int16)
    ink = np.clip(warped.astype(np.int16) + paper, 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(ink, 34, 255, cv2.THRESH_BINARY)
    return binary


def contours_from_mask(mask: np.ndarray) -> List[List[Point]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    usable: List[List[Point]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 2.0:
            continue
        epsilon = max(0.6, cv2.arcLength(contour, True) * 0.0035)
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        usable.append([(float(x), float(y)) for x, y in approx])
    usable.sort(key=lambda pts: (min(p[1] for p in pts), min(p[0] for p in pts)))
    return usable


def png_data_url(mask: np.ndarray) -> str:
    image = Image.fromarray(255 - mask).convert("L")
    rgb = Image.merge("RGB", (image, image, image))
    buf = BytesIO()
    rgb.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_handwriting(
    latex: str,
    *,
    max_width: int = 1080,
    max_height: int = 190,
    seed: int = 0,
) -> SyntheticHandwriting:
    mask = render_latex_mask(latex)
    mask = fit_mask(mask, max_width=max_width, max_height=max_height)
    mask = elastic_ink(mask, seed=seed)
    contours = contours_from_mask(mask)
    return SyntheticHandwriting(
        width=int(mask.shape[1]),
        height=int(mask.shape[0]),
        contours=contours,
        data_url=png_data_url(mask),
    )


def validate_line_gaps(line_count: int, line_gaps: Optional[Sequence[float]]) -> Optional[List[float]]:
    if line_gaps is None:
        return None
    gaps = [float(gap) for gap in line_gaps]
    expected = max(0, line_count - 1)
    if len(gaps) != expected:
        raise ValueError(f"line_gaps must contain {expected} values for {line_count} lines")
    if any(gap < 0 for gap in gaps):
        raise ValueError("line_gaps values must be non-negative")
    return gaps


def default_line_placements(
    line_count: int,
    *,
    board_width: int,
    board_height: int,
    margin_x: int = 92,
    margin_y: int = 74,
    line_gap: int = 34,
    line_height: int = 132,
    line_gaps: Optional[Sequence[float]] = None,
) -> List[Placement]:
    """Return vertically stacked placements with small x offsets."""
    custom_gaps = validate_line_gaps(line_count, line_gaps)
    placements: List[Placement] = []
    y = float(margin_y)
    x = float(margin_x)
    offsets = [0, 24, -12, 38, 8, -24]
    for index in range(line_count):
        if y + line_height > board_height - margin_y:
            y = float(margin_y)
            x += int(board_width * 0.46)
        placements.append({"x": x + offsets[index % len(offsets)], "y": y})
        if index < line_count - 1:
            gap = custom_gaps[index] if custom_gaps is not None else float(line_gap)
            y += line_height + gap
    return placements


def anchor_to_xy(
    anchor: str,
    width: int,
    height: int,
    board_width: int,
    board_height: int,
    margin: int,
) -> Tuple[float, float]:
    anchors = {
        "top-left": (margin, margin),
        "top-center": ((board_width - width) / 2, margin),
        "top-right": (board_width - width - margin, margin),
        "center-left": (margin, (board_height - height) / 2),
        "center": ((board_width - width) / 2, (board_height - height) / 2),
        "center-right": (board_width - width - margin, (board_height - height) / 2),
        "bottom-left": (margin, board_height - height - margin),
        "bottom-center": ((board_width - width) / 2, board_height - height - margin),
        "bottom-right": (board_width - width - margin, board_height - height - margin),
    }
    return anchors.get(anchor, anchors["center"])


def place_handwriting_lines(
    latex_lines: Sequence[str],
    *,
    board_width: int = 1400,
    board_height: int = 800,
    placements: Optional[Sequence[Placement]] = None,
    seed: int = 0,
    max_line_width: Optional[int] = None,
    max_line_height: int = 132,
    line_gaps: Optional[Sequence[float]] = None,
) -> BoardFixture:
    """Render and position multiple handwritten math lines on one board."""
    validate_line_gaps(len(latex_lines), line_gaps)
    if placements is None:
        placements = default_line_placements(
            len(latex_lines),
            board_width=board_width,
            board_height=board_height,
            line_height=max_line_height,
            line_gaps=line_gaps,
        )

    rendered_lines: List[PlacedHandwriting] = []
    max_width = max_line_width or max(260, board_width - 220)

    for index, latex in enumerate(latex_lines):
        rendered = render_handwriting(
            latex,
            max_width=max_width,
            max_height=max_line_height,
            seed=seed + index + 1,
        )
        placement = placements[index] if index < len(placements) else {}
        if "anchor" in placement:
            x, y = anchor_to_xy(
                str(placement["anchor"]),
                rendered.width,
                rendered.height,
                board_width,
                board_height,
                int(placement.get("margin", 80)),
            )
            x += float(placement.get("dx", 0))
            y += float(placement.get("dy", 0))
        else:
            x = float(placement.get("x", 90))
            y = float(placement.get("y", 80 + index * (max_line_height + 34)))
        rendered_lines.append(
            PlacedHandwriting(
                latex=latex,
                x=max(0.0, min(float(board_width - rendered.width), x)),
                y=max(0.0, min(float(board_height - rendered.height), y)),
                width=rendered.width,
                height=rendered.height,
                contours=rendered.contours,
                data_url=rendered.data_url,
            )
        )

    return BoardFixture(
        width=board_width,
        height=board_height,
        lines=rendered_lines,
        data_url=board_png_data_url(board_width, board_height, rendered_lines),
    )


def board_png_data_url(width: int, height: int, lines: Sequence[PlacedHandwriting]) -> str:
    board = Image.new("RGB", (width, height), "white")
    for line in lines:
        raw = base64.b64decode(line.data_url.split(",", 1)[1])
        line_png = Image.open(BytesIO(raw)).convert("RGB")
        board.paste(line_png, (int(round(line.x)), int(round(line.y))))
    buf = BytesIO()
    board.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def save_board_png(board: BoardFixture, path: str) -> None:
    raw = base64.b64decode(board.data_url.split(",", 1)[1])
    with open(path, "wb") as f:
        f.write(raw)


def translate_contours(
    contours: Sequence[Sequence[Point]],
    x: float,
    y: float,
) -> List[List[Dict[str, float]]]:
    return [
        [{"x": round(px + x, 3), "y": round(py + y, 3)} for px, py in contour]
        for contour in contours
    ]
