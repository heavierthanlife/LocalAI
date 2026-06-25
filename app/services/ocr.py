"""OCR manager (PaddleOCR with EasyOCR fallback)."""
import threading
import logging

logger = logging.getLogger(__name__)


class OCRManager:
    """Thread-safe singleton OCR manager."""

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
        self.engine_name = None
        self._init_ocr()

    def _init_ocr(self):
        try:
            from paddleocr import PaddleOCR
            try:
                self.reader = PaddleOCR(use_textline_orientation=True, lang='ch')
            except TypeError:
                self.reader = PaddleOCR(use_angle_cls=True, lang='ch')
            self.engine_name = "PaddleOCR"
            logger.info("PaddleOCR initialized successfully.")
        except ImportError:
            logger.warning("PaddleOCR not installed. Will try EasyOCR.")
        except Exception as e:
            logger.warning(f"PaddleOCR init failed: {e}. Will try EasyOCR.")
        if self.reader is None:
            try:
                import easyocr
                self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
                self.engine_name = "EasyOCR"
                logger.info("EasyOCR initialized as fallback.")
            except ImportError:
                logger.error("No OCR engine available. Install 'paddleocr' or 'easyocr'.")
                self.reader = None
            except Exception as e:
                logger.error(f"EasyOCR init failed: {e}")
                self.reader = None

    def is_available(self):
        return self.reader is not None

    def run_ocr(self, image_np):
        if self.reader is None:
            return ""
        try:
            if self.engine_name == "PaddleOCR":
                result = self.reader.ocr(image_np, cls=True)
                if result and result[0]:
                    return "\n".join([line[1][0] for line in result[0]])
            elif self.engine_name == "EasyOCR":
                result = self.reader.readtext(image_np, detail=0, paragraph=True)
                if result:
                    return "\n".join(result)
            return ""
        except Exception as e:
            logger.error(f"OCR run error: {e}")
            return ""


# Global singletons
ocr_manager = OCRManager()
run_ocr = ocr_manager.run_ocr
