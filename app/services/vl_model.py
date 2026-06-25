"""Vision-Language model for image description (Qwen VL / DashScope)."""
import os
import io
import base64
import logging
from PIL import Image

logger = logging.getLogger(__name__)


class VLModel:
    """Singleton vision-language model client for image/page description."""

    def __init__(self):
        self.api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        if not self.api_key:
            self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.model_name = "qwen3-vl-plus-2025-12-19"
        self.client = None
        self.max_image_size = 1024
        self._init_client()

    def _init_client(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            logger.info(f"VL client initialized with model {self.model_name}")
        except ImportError:
            logger.error("OpenAI package not installed. VL model disabled.")
            self.client = None
        except Exception as e:
            logger.error(f"VL client init failed: {e}")
            self.client = None

    def is_available(self):
        return self.client is not None and self.api_key is not None

    def _preprocess_image(self, image_bytes):
        try:
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            w, h = img.size
            if max(w, h) > self.max_image_size:
                scale = self.max_image_size / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            return buffer.getvalue()
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}, using original")
            return image_bytes

    def encode_image_to_base64(self, image_bytes):
        processed = self._preprocess_image(image_bytes)
        return base64.b64encode(processed).decode('utf-8')

    def describe_image(self, image_bytes, prompt="请描述这张图片的内容"):
        if not self.is_available():
            return "⚠️ VL模型不可用，请检查API密钥。"
        try:
            base64_image = self.encode_image_to_base64(image_bytes)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        {"type": "text", "text": prompt}
                    ]
                }],
                max_tokens=800,
                temperature=0.7
            )
            description = response.choices[0].message.content
            if not description:
                return "⚠️ 未获得图片描述，请稍后重试。"
            return description
        except Exception as e:
            error_msg = str(e)
            logger.error(f"VL image description failed: {error_msg}")
            if "InvalidModel" in error_msg or "model not found" in error_msg:
                return f"⚠️ 模型 {self.model_name} 不可用，请检查模型名称或API密钥。"
            elif "rate limit" in error_msg.lower():
                return "⚠️ 请求过于频繁，请稍后再试。"
            elif "content policy" in error_msg.lower():
                return "⚠️ 图片内容不符合安全规范，无法描述。"
            else:
                return f"⚠️ 图片描述失败: {error_msg[:100]}"

    def describe_pdf_page(self, image_bytes, page_num):
        return self.describe_image(
            image_bytes,
            f"请详细描述这个PDF页面(第{page_num}页)的内容，包括标题、段落、表格、图表等关键信息。"
        )


# Global singleton
vl_model = VLModel()
