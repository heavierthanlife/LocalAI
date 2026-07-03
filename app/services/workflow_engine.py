"""Multi-step document workflow: draft → review → revise → finalize.
Uses LangGraph for state-machine orchestration (already installed).
Each step calls the LLM with accumulated context.
"""
import logging
from typing import TypedDict, Optional

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    query: str           # user's original request
    draft: str           # AI's first draft
    review: str          # AI self-review feedback
    revised: str         # revised version
    finalized: str       # final output
    step: str            # current step
    history: list        # all user messages for context


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call LLM via unified entry point."""
    from app.services.llm_provider import call_llm as _call
    return _call(system_prompt, user_prompt, temperature=0.5, max_tokens=3200)


def run_document_workflow(query: str, industry: str = "general") -> dict:
    """Run the full draft→review→revise→finalize workflow.
    
    Args:
        query: user's natural language request (e.g., "帮我写投标函")
        industry: domain identifier for workflow prompt
    
    Returns:
        dict with keys: draft, review, revised, finalized
    """
    import os

    # Load industry workflow if available
    workflow_text = ""
    base = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'workflows')
    wf_path = os.path.join(base, f'{industry}.md')
    try:
        if os.path.exists(wf_path):
            with open(wf_path, 'r', encoding='utf-8') as f:
                workflow_text = f.read()[:3000]
    except Exception:
        pass

    # Step 1: Draft
    logger.info(f"Workflow: drafting for '{query[:50]}...'")
    draft_prompt = f"""你是一个专业的文档起草助手。
{workflow_text}

请根据用户需求起草一份专业文档。内容要完整、格式要清晰。
如果是投标相关文档，请严格遵循行业规范。
输出markdown格式。
绝不编造具体的金额、日期、公司名称或联系方式——如果用户没提供，用[待填写]占位。
如果某部分信息不足，在该处标注「（需补充：XXX）」而不是编造内容。"""

    draft = _call_llm(draft_prompt, f"请起草以下内容:\n{query}")

    # Step 2: Self-review (with fact anchoring — must cite specific sections)
    logger.info("Workflow: self-reviewing draft")
    review_prompt = """你是文档质量审核员。请对以下草稿进行自我审查。

【审查规则 — 必须遵守】
- 每条意见必须标注对应段落：用「第X段」或「## 标题名」指明位置
- 不得编造不存在的问题。如果草稿确实没有问题，直接说"审查通过"
- 禁止凭空添加"缺少XX"——必须在草稿中能看到确实缺失的痕迹才指出

审查角度：
1. 格式是否符合行业规范
2. 内容是否完整
3. 有没有逻辑矛盾或错误
4. 改进建议（具体怎么改）

用简洁的中文列出审查意见。"""

    review = _call_llm(review_prompt, f"请审查以下文档草稿:\n\n{draft[:4000]}")

    # Step 3: Revise based on review
    logger.info("Workflow: revising draft")
    revise_prompt = f"""你是文档修订助手。请根据审查意见修改文档。
{workflow_text}

审查意见:
{review[:2000]}

【修订规则】
- 只修改审查意见中明确指出的问题，不要在无关部分擅自改动
- 如果审查意见指出某处缺失内容，只添加必要的最少内容
- 保持未涉及部分的原样不变
- 修改后的完整文档（markdown格式）"""

    revised = _call_llm(revise_prompt, f"原始草稿:\n{draft[:4000]}")

    # Step 4: Final polish
    logger.info("Workflow: finalizing")
    finalize_prompt = """你是文档终审助手。请对修改后的文档做最终润色：
1. 检查格式一致性
2. 润色语言表达
3. 确保所有引用和数据准确
4. 输出最终版文档

输出markdown格式。"""

    finalized = _call_llm(finalize_prompt, f"修改后文档:\n{revised[:4000]}")

    return {
        "draft": draft,
        "review": review,
        "revised": revised,
        "finalized": finalized,
    }
