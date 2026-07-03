"""OCR manager using EasyOCR (Python 3.12 compatible)."""
import threading
import logging

logger = logging.getLogger(__name__)


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
            self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
            logger.info("EasyOCR initialized (Chinese + English, CPU)")
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


ocr_manager = OCRManager()
run_ocr = ocr_manager.run_ocr
