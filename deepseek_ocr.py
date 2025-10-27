from __future__ import annotations

import io
import os
import tempfile
from typing import Dict, List, Any

from PIL import Image


class DeepSeekOCRClient:
    """Wrapper around deepseek-ai/DeepSeek-OCR Hugging Face model.

    Provides simple methods for OCR from image bytes and full PDF bytes.
    """

    def __init__(self) -> None:
        from transformers import AutoModel, AutoTokenizer
        import torch

        self.model_name = "deepseek-ai/DeepSeek-OCR"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        # Avoid flash-attn requirement; use default attention. Cast to bf16 if supported
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_safetensors=True,
        )
        self.model.eval()

        if torch.cuda.is_available():
            try:
                self.model = self.model.to(torch.bfloat16).cuda()
            except Exception:
                self.model = self.model.cuda()

    def _infer_image_path(self, image_path: str, prompt: str = "<image>\nFree OCR.") -> str:
        """Run inference on an image path and return text output."""
        # Model implements a custom .infer when trust_remote_code=True
        # Fallback to string cast if object is complex
        try:
            out = self.model.infer(
                self.tokenizer,
                prompt=prompt,
                image_file=image_path,
                output_path=tempfile.gettempdir(),
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=False,
                test_compress=True,
            )
        except AttributeError:
            # If custom .infer isn't exposed, raise a helpful error
            raise RuntimeError("DeepSeek-OCR model does not expose .infer; ensure trust_remote_code=True")

        if isinstance(out, str):
            return out
        # Some versions may return complex objects; best-effort stringify
        return str(out)

    def extract_text_from_image_bytes(self, image_bytes: bytes, prompt: str = "<image>\nFree OCR.") -> Dict[str, Any]:
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
                tmp_img.write(image_bytes)
                tmp_img.flush()
                path = tmp_img.name

            text = self._infer_image_path(path, prompt=prompt)
            return {"success": True, "pages": [text]}
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            try:
                if 'path' in locals() and os.path.exists(path):
                    os.unlink(path)
            except Exception:
                pass

    def extract_text_from_pdf(self, file_bytes: bytes, language: str = "auto") -> Dict[str, Any]:
        """Extract text from a PDF by rasterizing pages and running the model per page."""
        import pdfplumber

        pages_text: List[str] = []
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    try:
                        pil_img = page.to_image().original.convert("RGB")
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
                            pil_img.save(tmp_img, format="PNG")
                            img_path = tmp_img.name
                        text = self._infer_image_path(img_path, prompt="<image>\n<|grounding|>Convert the document to markdown. ")
                        pages_text.append(text)
                    finally:
                        try:
                            if 'img_path' in locals() and os.path.exists(img_path):
                                os.unlink(img_path)
                        except Exception:
                            pass
        except Exception as e:
            return {"success": False, "error": str(e)}

        return {"success": True, "pages": pages_text}


