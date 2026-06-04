import re
from typing import List, Tuple

from langchain_core.tools import Tool

from agents.langchain_agent_factory import run_langchain_react_agent
from agents.utils import join_docs
from domain_config import get_domain_instructions
from llm_provider import get_llm


TECHNICAL_PREFIXES = (
    "thought:",
    "action:",
    "action input:",
    "observation:",
    "question:",
    "final answer:",
)


def _parse_topics(raw_text: str) -> List[str]:
    topics: List[str] = []

    if not raw_text:
        return topics

    banned_fragments = (
        "остальные темы",
        "не были найдены",
        "в результате поиска",
        "поэтому предлагаются",
        "только три основные темы",
        "agent stopped due to",
        "не удалось",
        "релевантные фрагменты",
    )

    for line in raw_text.splitlines():
        line = line.strip()

        if not line:
            continue

        line = re.sub(
            r"^Final Answer:\s*",
            "",
            line,
            flags=re.IGNORECASE,
        ).strip()

        match = re.match(r"^\d+\s*[\.)]\s*(.+)$", line)

        if not match:
            continue

        topic = match.group(1).strip()
        topic_lower = topic.lower()

        if any(fragment in topic_lower for fragment in banned_fragments):
            continue

        if topic_lower.startswith(
            (
                "thought:",
                "action:",
                "action input:",
                "observation:",
                "question:",
                "final answer:",
            )
        ):
            continue

        topics.append(topic)

    return topics[:5]


def analyze_document_topics_agent(
    retriever,
    is_api_mode: bool,
    domain: str,
) -> Tuple[List[str], str]:
    domain_instructions = get_domain_instructions(domain)
    context_cache = {"context": ""}

    def search_document_fragments(query: str) -> str:
        try:
            docs = retriever.invoke(query)
            rag_context = join_docs(docs, max_chars=3500)
        except Exception as exc:
            rag_context = f"Ошибка поиска по документу: {exc}"

        context_cache["context"] = rag_context

        return rag_context or "Релевантные фрагменты документа не найдены."

    tools = [
        Tool.from_function(
            name="search_document_fragments",
            func=search_document_fragments,
            description=(
                "Ищет релевантные фрагменты учебного документа через RAG. "
                "На вход принимает поисковый запрос по теме документа. "
                "Инструмент нужно вызвать один раз перед составлением тем."
            ),
        )
    ]

    llm = get_llm(
        is_api_mode=is_api_mode,
        temperature=0.0,
        max_tokens=700,
    )

    task = f"""
Проанализируй учебный документ и предложи преподавателю 3-5 тем для тестирования.

ТЫ РАБОТАЕШЬ КАК LANGCHAIN REACT AGENT.

Для корректной работы ты ОБЯЗАН соблюдать ReAct-формат.

Сначала обязательно вызови инструмент:

Thought: нужно найти основные фрагменты учебного документа.
Action: search_document_fragments
Action Input: основные темы правила обязанности инструкции безопасность порядок действий

После получения Observation запрещено снова писать Action или Action Input.

После Observation следующий ответ обязан быть только финальным:

Thought: теперь я знаю итоговый ответ
Final Answer:
1. Название темы — кратко что проверяется
2. Название темы — кратко что проверяется
3. Название темы — кратко что проверяется

ОТРАСЛЕВАЯ СПЕЦИФИКА:
{domain_instructions}

Правила финального ответа:
- верни только 3-5 тем;
- не генерируй тест;
- не добавляй длинных пояснений;
- не используй слова Thought, Action, Action Input, Observation в Final Answer;
- каждая тема должна быть пригодна для последующей генерации теста.
- не добавляй итоговые комментарии после списка;
- не пиши фразы вида "остальные темы не найдены";
- если нашёл только 3 темы, просто верни 3 темы без объяснения причины;
- каждая строка финального ответа должна начинаться с номера: 1., 2., 3., 4. или 5.;
"""

    try:
        raw_topics = run_langchain_react_agent(
            llm=llm,
            tools=tools,
            user_task=task,
            max_iterations=3,
        )
    except Exception:
        raw_topics = ""

    topics = _parse_topics(raw_topics)

    if not topics:
        fallback_query = (
            "основные темы правила обязанности инструкции безопасность порядок действий"
        )
        fallback_context = search_document_fragments(fallback_query)

        fallback_prompt = f"""
Ты — агент анализа учебного документа TransportExam AI.

ФРАГМЕНТЫ ДОКУМЕНТА:
{fallback_context}

ОТРАСЛЕВАЯ СПЕЦИФИКА:
{domain_instructions}

Верни 3-5 тем для тестирования.

Формат:
1. Название темы — кратко что проверяется
2. Название темы — кратко что проверяется
3. Название темы — кратко что проверяется

Не добавляй пояснений.
"""

        try:
            result = llm.invoke(fallback_prompt)
            raw_topics = getattr(result, "content", str(result)).strip()
            topics = _parse_topics(raw_topics)
        except Exception:
            topics = []

    return topics[:5], context_cache["context"]