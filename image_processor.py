"""OCR utilities for image uploads."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

from PIL import Image
import pytesseract


@dataclass
class OCRResult:
    text: str
    success: bool
    error_message: Optional[str] = None


def extract_text_from_image_bytes(data: bytes, filename: str = "image.png") -> OCRResult:
    try:
        image = Image.open(io.BytesIO(data))
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        text = pytesseract.image_to_string(image)
        return OCRResult(text=text.strip(), success=True)
    except Exception as exc:
        return OCRResult(text="", success=False, error_message=f"{filename}: {exc}")



