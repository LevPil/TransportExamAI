import re
from typing import Any, Dict, List, Optional
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from llm_provider import get_llm

chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Ты - ИИ-ассистент TransportExam AI.
Контекст: {context}

Текущий тест: {current_test}

Если это преподаватель: доработай тест по его просьбе и выведи новый полный вариант теста (с ключами).
Если это студент: его результат: {student_score}.
Его детальные ответы:
{answers_report}

Помоги студенту разобрать ошибки. Не давай прямых ответов сразу — объясняй правила и регламенты.
""",
        ),
        ("user", "{input}"),
    ]
)

def chat_with_assistant(
    context,
    current_test,
    student_score,
    user_input,
    is_api_mode,
    answers_report="",
):
    llm = get_llm(is_api_mode=is_api_mode)
    chain = chat_prompt | llm | StrOutputParser()

    return chain.invoke(
        {
            "context": context,
            "current_test": current_test,
            "student_score": student_score,
            "answers_report": answers_report,
            "input": user_input,
        }
    )

student_rag_chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Ты — ИИ-помощник студента TransportExam AI.

Твоя задача — отвечать на уточняющие вопросы студента после прохождения теста.
Отвечай как обычный учебный помощник, а НЕ как агент персонального плана повторения.

У тебя есть:
1. История диалога со студентом.
2. Фрагменты учебного документа из RAG.
3. Результат студента и отчёт об ошибках.

Как использовать историю диалога:
- Используй историю только для понимания, какую тему студент уточняет.
- Не начинай ответ словами "я уже объяснял", "как я говорил ранее", "мы уже обсуждали".
- Если студент пишет "расскажи подробнее", "почему так?", "приведи пример", "про второй пункт" — найди ближайшую тему в предыдущем ответе ассистента и продолжи её.
- Если по истории непонятно, какой пункт имеет в виду студент, уточни это.
- Не придумывай, что уже было объяснено, если этого нет в истории.

Как использовать RAG:
- Основные факты бери из фрагментов учебного документа.
- Не используй интернет.
- Не выдумывай факты, которых нет в документе, отчёте об ошибках или истории диалога.
- Если в RAG-контексте нет точного ответа, скажи, что точный фрагмент в загруженном документе не найден.

РЕЗУЛЬТАТ СТУДЕНТА:
{student_score}

ОТЧЁТ ОБ ОШИБКАХ:
{answers_report}

ФРАГМЕНТЫ УЧЕБНОГО ДОКУМЕНТА:
{rag_context}

Правила ответа:
- Отвечай на русском языке.
- Ответ должен быть коротким и полезным.
- Не составляй заново полный персональный план, если студент прямо этого не просит.
- Не говори "я уже объяснял".
- Если студент спрашивает про пункт из плана, объясни именно этот пункт.
""",
        ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("user", "{input}"),
    ]
)


def _join_rag_docs(docs: List[Any], max_chars: int = 3000) -> str:
    parts = []
    total = 0

    for doc in docs:
        text = getattr(doc, "page_content", str(doc)).strip()
        text = re.sub(r"\s+", " ", text)

        if not text:
            continue

        available = max_chars - total

        if available <= 0:
            break

        parts.append(text[:available])
        total += len(parts[-1])

    return "\n\n".join(parts).strip()


def build_langchain_chat_history(
    messages: Optional[List[Dict[str, Any]]],
    max_messages: int = 8,
) -> List[BaseMessage]:
    if not messages:
        return []
    result: List[BaseMessage] = []
    for msg in messages[-max_messages:]:
        role = msg.get("role")
        content = str(msg.get("content", "")).strip()
        message_type = msg.get("message_type")
        if not content:
            continue
        if role == "user" and message_type == "revision_plan_request":
            continue
        if role == "user":
            result.append(HumanMessage(content=content))
        elif role == "assistant":
            result.append(AIMessage(content=content))
    return result


def build_contextual_rag_query(
    user_input: str,
    chat_history: Optional[List[Dict[str, Any]]],
    answers_report: str,
) -> str:

    last_assistant_message = ""

    for msg in reversed(chat_history or []):
        if msg.get("role") == "assistant":
            last_assistant_message = str(msg.get("content", "")).strip()
            break

    return (
        f"Уточняющий вопрос студента:\n{user_input}\n\n"
        f"Последний ответ ассистента или план повторения:\n"
        f"{last_assistant_message[:1200]}\n\n"
        f"Отчёт об ошибках студента:\n"
        f"{answers_report[:1200]}"
    )


def chat_with_student_rag(
    retriever,
    current_test: str,
    student_score: str,
    user_input: str,
    is_api_mode: bool,
    answers_report: str = "",
    chat_history: Optional[List[Dict[str, Any]]] = None,
) -> str:

    try:
        docs = retriever.invoke(user_input)
        rag_context = _join_rag_docs(docs, max_chars=3000)
    except Exception:
        rag_context = ""

    langchain_history = build_langchain_chat_history(
        messages=chat_history,
        max_messages=8,
    )

    llm = get_llm(
        is_api_mode=is_api_mode,
        temperature=0.2,
        max_tokens=800,
    )

    chain = student_rag_chat_prompt | llm | StrOutputParser()

    try:
        return chain.invoke(
            {
                "chat_history": langchain_history,
                "rag_context": rag_context
                or "По запросу студента релевантные фрагменты в документе не найдены.",
                "current_test": current_test,
                "student_score": student_score or "результат не указан",
                "answers_report": answers_report or "отчёт об ответах отсутствует",
                "input": user_input,
            }
        ).strip()
    except Exception as e:
        return f"Не удалось сформировать ответ по учебному документу: {str(e)}"