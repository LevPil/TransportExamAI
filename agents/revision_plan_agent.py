import re
from typing import Dict, List

from langchain_core.tools import Tool

from agents.langchain_agent_factory import run_langchain_react_agent
from agents.utils import invoke_llm_text
from llm_provider import get_llm
from test_format import normalize_answer_letter


def _extract_wrong_questions_from_report(
    answers_report: str,
    limit: int = 3,
) -> List[Dict[str, str]]:
    if not answers_report:
        return []

    wrong_items: List[Dict[str, str]] = []

    pattern = re.compile(
        r"Вопрос\s+(\d+)\s*:\s*(.*?)\n"
        r"Студент\s+выбрал\s*:\s*([АБВГA-D])\s*,\s*Правильно\s*:\s*([АБВГA-D])\.\s*"
        r"Результат\s*:\s*ОШИБКА",
        re.IGNORECASE | re.DOTALL,
    )

    for match in pattern.finditer(answers_report):
        question_text = re.sub(r"\s+", " ", match.group(2)).strip()

        wrong_items.append(
            {
                "id": match.group(1),
                "question": question_text,
                "student_answer": normalize_answer_letter(match.group(3)),
                "correct_answer": normalize_answer_letter(match.group(4)),
            }
        )

        if len(wrong_items) >= limit:
            break

    return wrong_items


def _format_wrong_questions(wrong_items: List[Dict[str, str]]) -> str:
    if not wrong_items:
        return (
            "Явных ошибок в отчёте не найдено. "
            "Нужно закрепить материал и повторить ключевые правила."
        )

    return "\n".join(
        f"{item['id']}. {item['question']} "
        f"(студент выбрал {item['student_answer']}, правильно {item['correct_answer']})"
        for item in wrong_items
    )


def _build_revision_context(
    retriever,
    wrong_items: List[Dict[str, str]],
    user_input: str = "",
    max_chars: int = 2200,
) -> str:
    if not retriever:
        return ""

    queries: List[str] = []

    for item in wrong_items:
        question = item.get("question", "").strip()
        if question:
            queries.append(question)

    if user_input.strip():
        queries.append(user_input.strip())

    if not queries:
        queries = ["основные правила безопасность порядок действий обязанности инструкции"]

    seen_fragments = set()
    context_parts: List[str] = []
    total = 0

    for query in queries[:3]:
        try:
            docs = retriever.invoke(query)
        except Exception:
            docs = []

        for doc in docs[:1]:
            text = getattr(doc, "page_content", str(doc)).strip()
            text = re.sub(r"\s+", " ", text)

            if not text:
                continue

            fingerprint = text[:180]

            if fingerprint in seen_fragments:
                continue

            seen_fragments.add(fingerprint)

            fragment = f"Фрагмент по запросу '{query}':\n{text}"
            available = max_chars - total

            if available <= 0:
                break

            context_parts.append(fragment[:available])
            total += len(context_parts[-1])

        if total >= max_chars:
            break

    return "\n\n".join(context_parts).strip()


def run_revision_plan_agent(
    retriever,
    student_score: str,
    is_api_mode: bool,
    answers_report: str = "",
    user_input: str = "Составь персональный план повторения по моим ошибкам.",
) -> str:
    wrong_items = _extract_wrong_questions_from_report(answers_report)
    wrong_questions = _format_wrong_questions(wrong_items)
    tool_state = {"collect_revision_data_called": False}

    def collect_revision_data(_: str = "") -> str:
        if tool_state["collect_revision_data_called"]:
            return (
                "Данные уже были получены ранее. "
                "Повторно вызывать collect_revision_data запрещено. "
                "Следующий шаг агента должен быть только Final Answer с персональным планом."
            )

        tool_state["collect_revision_data_called"] = True

        rag_context = _build_revision_context(
            retriever=retriever,
            wrong_items=wrong_items,
            user_input=user_input,
            max_chars=2200,
        )

        if not rag_context:
            rag_context = (
                "RAG-контекст не найден. "
                "Используй только данные об ошибках студента."
            )

        return (
            "ДАННЫЕ ДЛЯ ПЕРСОНАЛЬНОГО ПЛАНА ПОВТОРЕНИЯ\n\n"
            f"РЕЗУЛЬТАТ СТУДЕНТА:\n"
            f"{student_score or 'результат не указан'}\n\n"
            f"ОШИБКИ СТУДЕНТА:\n"
            f"{wrong_questions}\n\n"
            f"ЗАПРОС СТУДЕНТА:\n"
            f"{user_input or 'Составь персональный план повторения по моим ошибкам.'}\n\n"
            f"ФРАГМЕНТЫ УЧЕБНОГО ДОКУМЕНТА:\n"
            f"{rag_context}\n\n"
            "СТАТУС ИНСТРУМЕНТА:\n"
            "Данные получены успешно. "
            "Инструмент collect_revision_data больше вызывать нельзя. "
            "Следующий шаг агента — Final Answer."
        )

    if is_api_mode:
        revision_data = collect_revision_data("")

        llm = get_llm(
            is_api_mode=True,
            temperature=0.0,
            max_tokens=700,
        )

        prompt = f"""
Ты — ИИ-методист TransportExam AI.

Составь персональный план повторения для студента строго на основе переданных данных.

ДАННЫЕ:
{revision_data}

Правила:
- не используй Thought, Action, Action Input, Observation, Final Answer;
- не упоминай инструменты;
- не выдумывай факты, которых нет в данных;
- не используй интернет;
- не копируй данные дословно;
- ответ должен быть коротким и полезным;
- если явных ошибок нет, так и напиши в разделе слабых мест.

Формат ответа:

### Персональный план повторения

1. Слабые места:
- 2-4 коротких пункта по ошибкам студента.

2. Что повторить:
- 3 конкретные темы, правила или действия.

3. Как тренироваться:
- 2 коротких практических действия на 10-15 минут.

4. Мини-подсказка:
- одно главное правило простыми словами.
"""

        try:
            response = invoke_llm_text(llm, prompt).strip()

            if not response:
                return (
                    "Не удалось сформировать персональный план через GigaChat. "
                    "Попробуйте повторить запрос."
                )

            return response

        except Exception as exc:
            return f"GigaChat не смог сформировать персональный план: {exc}"

    tools = [
        Tool.from_function(
            name="collect_revision_data",
            func=collect_revision_data,
            description=(
                "Собирает все данные для персонального плана повторения: "
                "результат студента, ошибки студента, запрос студента "
                "и RAG-фрагменты учебного документа. "
                "Инструмент нужно вызвать только один раз."
            ),
        )
    ]

    llm = get_llm(
        is_api_mode=False,
        temperature=0.0,
        max_tokens=700,
    )

    task = """
    Составь персональный план повторения для студента.

    Сначала обязательно вызови инструмент collect_revision_data.

    Первый шаг должен быть строго таким:

    Thought: нужно получить данные для персонального плана повторения
    Action: collect_revision_data
    Action Input:

    После получения Observation инструмент больше не вызывай.

    После Observation сформируй финальный ответ строго через служебный маркер Final Answer.

    Правильный формат второго шага:

    Thought: данные получены, теперь можно сформировать итоговый ответ
    Final Answer:
    ### Персональный план повторения

    1. Слабые места:
    - 2-4 коротких пункта по ошибкам студента.

    2. Что повторить:
    - 3 конкретные темы, правила или действия.

    3. Как тренироваться:
    - 2 коротких практических действия на 10-15 минут.

    4. Мини-подсказка:
    - одно главное правило простыми словами.

    Правила:
    - используй только данные из Observation;
    - не выдумывай факты, которых нет в данных;
    - не используй интернет;
    - не вызывай collect_revision_data повторно;
    - в пользовательской части ответа после Final Answer: не используй слова Thought, Action, Action Input, Observation.
    """

    try:
        response = run_langchain_react_agent(
            llm=llm,
            tools=tools,
            user_task=task,
            max_iterations=3,
        )

        response = response.strip()

        if not response:
            return (
                "Агент персонального плана повторения не вернул ответ. "
                "Попробуйте повторить запрос."
            )

        if "Agent stopped due to iteration limit or time limit" in response:
            return (
                "Не удалось сформировать персональный план через агента. "
                "Попробуйте повторить запрос или уменьшите объём загруженного документа."
            )

        return response

    except Exception as exc:
        return f"Агент персонального плана повторения не смог сформировать ответ: {exc}"