"""Headroom — context compression for LLM token savings.

Install: pip install headroom-ai
Usage: Automatically compresses file content, Bocha search results,
       and conversation history before sending to the LLM.
       CCR (Cache Context Retrieval) ensures zero information loss.
"""
import logging

logger = logging.getLogger(__name__)
_available = False


def init():
    """Check if headroom is available."""
    global _available
    try:
        from headroom import compress  # noqa: F401
        _available = True
        logger.info("Headroom compression loaded")
    except ImportError:
        logger.info("Headroom not installed — token compression disabled")


def compress_messages(messages: list) -> list:
    """Compress LLM message list. Returns original if headroom unavailable.
    
    Headroom expects: [{\"role\": \"user\", \"content\": \"...\"}, ...]
    Returns: [{\"role\": \"user\", \"content\": \"...\"}, ...] (compressed)
    """
    if not _available or not messages or len(messages) < 2:
        return messages
    try:
        from headroom import compress
        result = compress(messages, content_type='messages')
        if hasattr(result, 'messages') and result.messages:
            saved = sum(len(m.get('content','')) for m in messages) - sum(len(m.get('content','')) for m in result.messages)
            if saved > 0:
                logger.info(f"Headroom: saved ~{saved} chars from messages")
            return result.messages
    except Exception as e:
        logger.warning(f"Headroom failed: {e}")
    return messages


def compress_file_content(text: str) -> str:
    """Compress file content string via headroom message format."""
    if not _available or not text or len(text) < 500:
        return text
    try:
        from headroom import compress
        result = compress([{'role': 'system', 'content': text}], content_type='document')
        if hasattr(result, 'messages') and result.messages:
            compressed = result.messages[0].get('content', text)
            if compressed and len(compressed) < len(text):
                logger.info(f"Headroom: file {len(text)} -> {len(compressed)} chars")
            return compressed or text
    except Exception as e:
        logger.warning(f"Headroom file compression failed: {e}")
    return text


def compress_search_results(text: str) -> str:
    """Compress Bocha search results."""
    return compress_file_content(text)  # same pattern


# Initialize on import
init()
