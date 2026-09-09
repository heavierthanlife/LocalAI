"""OCR manager using EasyOCR (Python 3.12 compatible)."""
import os
import threading
import logging

logger = logging.getLogger(__name__)


def _resolve_gpu_flag():
    """OCR_GPU env: 'true' / 'false' / 'auto' (default).

    auto probes torch.cuda.is_available() — works on CPU-only installs too.
    """
    setting = os.getenv('OCR_GPU', 'auto').strip().lower()
    if setting in ('1', 'true', 'yes'):
        return True
    if setting in ('0', 'false', 'no'):
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


class OCRManager:
    """Thread-safe singleton OCR manager. Uses EasyOCR for Chinese + English."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.reader = None
        self._init_ocr()

    def _init_ocr(self):
        try:
            import easyocr
            use_gpu = _resolve_gpu_flag()
            mode = 'GPU' if use_gpu else 'CPU'
            try:
                self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=use_gpu)
            except Exception as gpu_err:
                if use_gpu:
                    logger.warning(f"EasyOCR GPU init failed ({gpu_err}), falling back to CPU")
                    self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
                    mode = 'CPU'
                else:
                    raise
            # NOTE: do not touch reader.gpu — not exposed in all easyocr versions
            logger.info(f"EasyOCR initialized (Chinese + English, {mode})")
        except ImportError:
            logger.warning("EasyOCR not installed. Image text extraction will be skipped.")
            self.reader = None
        except Exception as e:
            logger.warning(f"EasyOCR init failed: {e}")
            self.reader = None

    def is_available(self):
        return self.reader is not None

    def run_ocr(self, image_np):
        if self.reader is None:
            return ""
        try:
            result = self.reader.readtext(image_np, detail=0, paragraph=True)
            if result:
                return "\n".join(result)
            return ""
        except Exception as e:
            logger.error(f"OCR run error: {e}")
            return ""

    def ocr_text_from_bytes(self, image_bytes):
        """EasyOCR on raw image bytes → joined text lines (deterministic).

        FIX-2026-09-09-022: VL 抽检的 OCR ground-truth 层——文本/数字密集的标书图
        用 OCR 读出精确文字，mimo 等弱 VL 不再单独承担"读数"职责。
        """
        if self.reader is None:
            return ""
        try:
            import numpy as np
            from PIL import Image as _PIL
            import io
            img = _PIL.open(io.BytesIO(image_bytes))
            if img.mode in ('RGBA', 'P', 'CMYK'):
                img = img.convert('RGB')
            max_dim = 2000
            if max(img.size) > max_dim:
                ratio = max_dim / max(img.size)
                img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)),
                                 _PIL.Resampling.LANCZOS)
            arr = np.array(img)
            return self.run_ocr(arr)
        except Exception as e:
            logger.error(f"OCR bytes error: {e}")
            return ""


ocr_manager = OCRManager()
run_ocr = ocr_manager.run_ocr
ocr_text_from_bytes = ocr_manager.ocr_text_from_bytes
