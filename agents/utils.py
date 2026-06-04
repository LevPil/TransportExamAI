import re
from typing import Any, List

def join_docs(docs: List[Any], max_chars: int = 3500) -> str:
    parts = []
    total = 0
    for doc in docs:
        text = getattr(doc, "page_content", str(doc)).strip()
        if not text:
            continue
        available = max_chars - total
        if available <= 0:
            break
        parts.append(text[:available])
        total += len(parts[-1])
    return "\n\n".join(parts).strip()

def invoke_llm_text(llm, prompt: str) -> str:
    result = llm.invoke(prompt)
    return getattr(result, "content", str(result)).strip()

def estimate_test_max_tokens(prompt_text: str, default: int = 1600) -> int:
    match = re.search(r"(?is)КОЛИЧЕСТВО\s+ВОПРОСОВ:\s*(\d+)", prompt_text)
    if not match:
        return default
    question_count = max(1, int(match.group(1)))
    return min(4096, max(default, 260 * question_count + 500))
