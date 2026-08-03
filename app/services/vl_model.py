"""Vision-Language model for image description — multi-provider (NVIDIA, Mimo, DashScope)."""
import os
import io
import base64
import logging
from PIL import Image

logger = logging.getLogger(__name__)

VL_PROVIDER_CONFIG = {
    'nvidia': {
        'name': 'NVIDIA',
        'env_key': 'NVIDIA_API_KEY',
        'base_url': 'https://integrate.api.nvidia.com/v1',
        'default_model': 'nvidia/nvlm-d-72b',
        'models': ['nvidia/nvlm-d-72b', 'nvidia/llama-3.2-nv-vision-34b'],
    },
    'mimo': {
        'name': 'Mimo',
        'env_key': 'MIMO_API_KEY',
        'base_url': 'https://token-plan-cn.xiaomimimo.com/v1',
        'default_model': 'mimo-v2.5-pro',
        'models': ['mimo-v2.5-pro', 'mimo-v2.5'],
    },
    'dashscope': {
        'name': '阿里云DashScope',
        'env_key': 'DASHSCOPE_API_KEY',
        'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'default_model': 'qwen3-vl-plus-2025-12-19',
        'models': ['qwen3-vl-plus-2025-12-19', 'qwen-vl-max', 'qwen-vl-plus'],
    },
}


def _get_active_vl_config():
    cfg = {'provider_id': 'nvidia', 'model': 'nvidia/nvlm-d-72b',
           'api_key': '', 'base_url': '', 'api_key_valid': False}
    try:
        from app.services.runtime_config import get as rc_get
        provider_id = rc_get('active_vl_provider', '') or 'auto'
        model_id = rc_get('active_vl_model', '') or 'auto'
    except Exception:
        provider_id = 'auto'
        model_id = 'auto'

    if provider_id == 'auto':
        for pid in ['nvidia', 'dashscope', 'mimo']:
            pcfg = VL_PROVIDER_CONFIG.get(pid, {})
            key = os.getenv(pcfg.get('env_key', ''), '').strip()
            if key:
                provider_id = pid
                break
        if provider_id == 'auto':
            provider_id = 'nvidia'

    if provider_id not in VL_PROVIDER_CONFIG:
        provider_id = 'nvidia'

    pcfg = VL_PROVIDER_CONFIG[provider_id]
    api_key = os.getenv(pcfg['env_key'], '').strip()
    if not api_key:
        api_key = os.getenv('DASHSCOPE_API_KEY', '').strip() or os.getenv('QWEN_API_KEY', '').strip() or os.getenv('DEEPSEEK_API_KEY', '').strip()

    if model_id == 'auto' or model_id not in pcfg['models']:
        model_id = pcfg['default_model']

    return {
        'provider_id': provider_id,
        'model': model_id,
        'api_key': api_key,
        'base_url': pcfg['base_url'],
        'api_key_valid': bool(api_key),
    }


class VLModel:
    """Singleton vision-language model client for image/page description."""

    def __init__(self):
        self._client = None
        self._model_name = ''
        self._provider_id = ''
        self.max_image_size = 1024
        self._init_client()

    @property
    def api_key(self):
        return _get_active_vl_config().get('api_key', '')

    @property
    def model_name(self):
        return self._model_name

    @property
    def provider_id(self):
        return self._provider_id

    def _init_client(self):
        cfg = _get_active_vl_config()
        self._provider_id = cfg['provider_id']
        self._model_name = cfg['model']
        api_key = cfg['api_key']
        base_url = cfg['base_url']

        if not api_key:
            self._client = None
            logger.warning("VL model: no API key configured")
            return

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"VL client initialized: provider={cfg['provider_id']}, model={self._model_name}")
        except ImportError:
            logger.error("OpenAI package not installed. VL model disabled.")
            self._client = None
        except Exception as e:
            logger.error(f"VL client init failed: {e}")
            self._client = None

    def is_available(self):
        if self._client is None:
            return False
        cfg = _get_active_vl_config()
        return cfg['api_key_valid']

    def reload(self):
        """Re-initialize client (e.g. after runtime config change)."""
        logger.info("VL model reload triggered")
        self._init_client()

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
            response = self._client.chat.completions.create(
                model=self._model_name,
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

    def describe_with_crosscheck(self, image_bytes, base_prompt: str = "") -> str:
        """Describe image with cross-validation: two prompts, compare for consistency.

        If the two descriptions have significant numerical differences, the result
        is prefixed with a [⚠️数据可能不一致] warning. This prevents downstream
        LLMs from blindly trusting a single VL hallucination.

        Costs 2x API calls — use only for critical content (tables, invoices, etc.)
        """
        if not self.is_available():
            return "⚠️ VL模型不可用，请检查API密钥。"

        prompt_a = base_prompt or "请详细描述这张图片的内容，特别注意其中的文字、数字和表格"
        prompt_b = "请从不同角度再次描述这张图片，重点关注数字、金额、日期等关键数据"

        desc1 = self.describe_image(image_bytes, prompt_a)
        desc2 = self.describe_image(image_bytes, prompt_b)

        # Cross-check
        try:
            from app.services.prompt_safety import vl_cross_check
            check = vl_cross_check(desc1, desc2)
        except Exception:
            check = {'consistent': True, 'note': 'check skipped'}

        if check.get('consistent'):
            # Use the more detailed description (usually the first)
            return desc1
        else:
            note = check.get('note', 'description mismatch')
            logger.warning(f"VL cross-check inconsistency: {note}")
            return f"[⚠️ 图片描述可能不一致: {note}]\n\n{desc1}\n\n---\n二次描述(供参考):\n{desc2}"


# Global singleton
vl_model = VLModel()
