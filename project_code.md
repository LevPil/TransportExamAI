This file is a merged representation of a subset of the codebase, containing files not matching ignore patterns, combined into a single document by Repomix.

# File Summary

## Purpose
This file contains a packed representation of a subset of the repository's contents that is considered the most important context.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching these patterns are excluded: data/**/*
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
agents/__init__.py
agents/chat_agent.py
agents/langchain_agent_factory.py
agents/revision_plan_agent.py
agents/test_generator_agent.py
agents/test_refiner_agent.py
agents/topic_agent.py
agents/utils.py
app.py
chroma_db/chroma.sqlite3
chroma_db/e2b77e31-6af3-4b7e-8b49-65d5608f5b61/data_level0.bin
chroma_db/e2b77e31-6af3-4b7e-8b49-65d5608f5b61/header.bin
chroma_db/e2b77e31-6af3-4b7e-8b49-65d5608f5b61/index_metadata.pickle
chroma_db/e2b77e31-6af3-4b7e-8b49-65d5608f5b61/length.bin
chroma_db/e2b77e31-6af3-4b7e-8b49-65d5608f5b61/link_lists.bin
config.py
database.py
docker-compose.yml
Dockerfile
domain_config.py
llm_provider.py
model.pt
models/preprocessor.joblib
models/student_performance_pipeline.joblib
models/student_performance_rf.joblib
rag_pipeline.py
requirements.txt
services/__init__.py
services/document_service.py
services/ml_service.py
services/result_service.py
services/test_service.py
speech_utils.py
state.py
test_format.py
train_ml_model.py
ui/__init__.py
ui/components.py
ui/sidebar.py
ui/student_page.py
ui/teacher_page.py
uploaded_docs/71342950.pdf
uploaded_docs/PDD.pdf
```

# Files

## File: agents/__init__.py
```python
from agents.topic_agent import analyze_document_topics_agent
from agents.test_generator_agent import build_prompt_for_selected_topic, generate_test_from_prompt
from agents.test_refiner_agent import ensure_test_is_parseable, refine_test_action, validate_test_with_llm
from agents.chat_agent import chat_with_assistant, chat_with_student_rag
from agents.revision_plan_agent import run_revision_plan_agent
```

## File: agents/chat_agent.py
```python
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
```

## File: agents/langchain_agent_factory.py
```python
from __future__ import annotations
from typing import Any, List
import traceback
try:
    from langchain.agents import AgentExecutor, create_react_agent
except ImportError:
    from langchain_classic.agents import AgentExecutor, create_react_agent

from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool


REACT_AGENT_PROMPT = PromptTemplate.from_template(
    """
Ты — специализированный LangChain ReAct Agent системы TransportExam AI.

Твоя задача — решить запрос пользователя с помощью доступных инструментов.

У тебя есть два разных слоя ответа:

1. СЛУЖЕБНЫЙ ФОРМАТ АГЕНТА.
Его читает LangChain-парсер. Его нужно соблюдать строго.

2. ПОЛЬЗОВАТЕЛЬСКИЙ ОТВЕТ.
Это текст после строки Final Answer:. Именно его увидит пользователь.

ВАЖНО:
- Строки Thought, Action, Action Input и Final Answer — это служебный формат агента.
- Они нужны только для LangChain-парсера.
- Пользовательский ответ начинается только после строки Final Answer:.
- Если ты уже получил Observation с нужными данными, НЕ вызывай инструмент повторно.
- После Observation можно сделать только Final Answer.
- Нельзя писать пользовательский Markdown-ответ сразу после Thought.
- Если финальный ответ готов, обязательно сначала напиши строку Final Answer:.

Доступные инструменты:
{tools}

Названия инструментов:
{tool_names}

СТРОГИЙ ФОРМАТ ДЛЯ ВЫЗОВА ИНСТРУМЕНТА:

Question: исходная задача пользователя
Thought: нужно получить данные через инструмент
Action: название инструмента из списка [{tool_names}]
Action Input: входные данные для инструмента

СТРОГИЙ ФОРМАТ ПОСЛЕ ПОЛУЧЕНИЯ Observation:

Thought: данные получены, теперь можно сформировать итоговый ответ
Final Answer:
текст ответа для пользователя

КРИТИЧЕСКОЕ ПРАВИЛО:
После Observation запрещено начинать ответ с заголовка, списка или обычного текста.
После Observation первая смысловая служебная строка должна быть:
Final Answer:

В пользовательском тексте после Final Answer: не используй слова Thought, Action, Action Input, Observation.

Question: {input}
Thought:{agent_scratchpad}
"""
)
def _extract_final_answer_from_error(error_text: str) -> str:
    marker = "Final Answer:"

    if marker not in error_text:
        return ""

    answer = error_text.split(marker, 1)[1].strip()

    stop_markers = [
        "For troubleshooting, visit:",
        "The above exception was the direct cause",
        "Traceback",
    ]

    for stop_marker in stop_markers:
        if stop_marker in answer:
            answer = answer.split(stop_marker, 1)[0].strip()

    return answer

def run_langchain_react_agent(
    llm: Any,
    tools: List[BaseTool],
    user_task: str,
    max_iterations: int = 3,
) -> str:

    print("\n" + "=" * 80)
    print("[AGENT DEBUG] START")
    print("=" * 80)
    print(f"[AGENT DEBUG] max_iterations = {max_iterations}")
    print(f"[AGENT DEBUG] tools = {[tool.name for tool in tools]}")
    print("[AGENT DEBUG] user_task:")
    print(user_task)
    print("=" * 80)

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=REACT_AGENT_PROMPT,
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=max_iterations,
        early_stopping_method="force",
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    try:
        result = executor.invoke({"input": user_task})

        print("\n" + "=" * 80)
        print("[AGENT DEBUG] RAW RESULT")
        print("=" * 80)
        print(result)

        if isinstance(result, dict):
            intermediate_steps = result.get("intermediate_steps", [])

            print("\n" + "=" * 80)
            print("[AGENT DEBUG] INTERMEDIATE STEPS")
            print("=" * 80)

            if not intermediate_steps:
                print("[AGENT DEBUG] intermediate_steps is empty")
            else:
                for idx, step in enumerate(intermediate_steps, start=1):
                    print(f"\n[AGENT DEBUG] STEP {idx}")
                    print("-" * 80)

                    try:
                        action, observation = step

                        print("[AGENT DEBUG] ACTION:")
                        print(action)

                        print("\n[AGENT DEBUG] TOOL:")
                        print(getattr(action, "tool", None))

                        print("\n[AGENT DEBUG] TOOL INPUT:")
                        print(getattr(action, "tool_input", None))

                        print("\n[AGENT DEBUG] OBSERVATION:")
                        print(observation)

                    except Exception as step_exc:
                        print("[AGENT DEBUG] Could not unpack intermediate step:")
                        print(step)
                        print(f"[AGENT DEBUG] Step error: {step_exc}")

            output = str(result.get("output", "")).strip()

            print("\n" + "=" * 80)
            print("[AGENT DEBUG] FINAL OUTPUT")
            print("=" * 80)
            print(output if output else "[EMPTY OUTPUT]")
            print("=" * 80)
            print("[AGENT DEBUG] END")
            print("=" * 80 + "\n")

            return output

        output = str(result).strip()

        print("\n" + "=" * 80)
        print("[AGENT DEBUG] NON-DICT RESULT")
        print("=" * 80)
        print(output)
        print("=" * 80)
        print("[AGENT DEBUG] END")
        print("=" * 80 + "\n")

        return output

    except Exception as exc:
        print("\n" + "=" * 80)
        print("[AGENT DEBUG] EXCEPTION")
        print("=" * 80)
        print(f"[AGENT DEBUG] Exception type: {type(exc).__name__}")
        print(f"[AGENT DEBUG] Exception message: {exc}")
        print("\n[AGENT DEBUG] TRACEBACK:")
        traceback.print_exc()
        print("=" * 80)
        print("[AGENT DEBUG] END WITH ERROR")
        print("=" * 80 + "\n")

        recovered_answer = _extract_final_answer_from_error(str(exc))

        if recovered_answer:
            print("\n" + "=" * 80)
            print("[AGENT DEBUG] RECOVERED FINAL ANSWER FROM PARSING ERROR")
            print("=" * 80)
            print(recovered_answer)
            print("=" * 80 + "\n")

            return recovered_answer

        raise
```

## File: agents/revision_plan_agent.py
```python
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
```

## File: agents/test_generator_agent.py
```python
from typing import Tuple
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from agents.test_refiner_agent import ensure_test_is_parseable
from agents.utils import estimate_test_max_tokens, invoke_llm_text, join_docs
from domain_config import get_domain_instructions
from llm_provider import get_llm
from test_format import TEST_FORMAT_RULES

def build_prompt_for_selected_topic(
    retriever,
    topic: str,
    domain: str,
    question_count: int = 3,
    difficulty: str = "Средний",
    extra_requirements: str = "",
) -> Tuple[str, str]:
    docs = retriever.invoke(topic)
    context = join_docs(docs, max_chars=4000)
    domain_instructions = get_domain_instructions(domain)
    extra_block = extra_requirements.strip() if extra_requirements.strip() else "Нет дополнительных требований."
    prompt = f"""Ты — строгий ИИ-экзаменатор и методист в сфере транспорта.

Создай тест по выбранной теме.

ТЕМА ТЕСТА:
{topic}

КОЛИЧЕСТВО ВОПРОСОВ:
{question_count}

УРОВЕНЬ СЛОЖНОСТИ:
{difficulty}

ОТРАСЛЕВАЯ СПЕЦИФИКА:
{domain_instructions}

ОСНОВНЫЕ ФРАГМЕНТЫ УЧЕБНОГО ДОКУМЕНТА:
{context}

ДОПОЛНИТЕЛЬНЫЕ ТРЕБОВАНИЯ ПРЕПОДАВАТЕЛЯ:
{extra_block}

{TEST_FORMAT_RULES}

Сгенерируй только сам тест. Не добавляй пояснений до или после теста.
"""
    return prompt.strip(), context

def generate_test_from_prompt(prompt_text: str, is_api_mode: bool) -> str:
    llm = get_llm(
        is_api_mode=is_api_mode,
        temperature=0.2,
        max_tokens=estimate_test_max_tokens(prompt_text, default=1600),
    )
    content = invoke_llm_text(llm, prompt_text)
    return ensure_test_is_parseable(
        content,
        is_api_mode=is_api_mode,
        repair_hint="Проверь результат генерации и обязательно восстанови финальный блок ключей ответов.",
    )
```

## File: agents/test_refiner_agent.py
```python
from typing import Any, Dict
from test_format import (
    TEST_FORMAT_RULES,
    extract_raw_question_blocks,
    normalize_test_format,
    validate_test_structure,
)
from llm_provider import get_llm
from agents.utils import invoke_llm_text

def ensure_test_is_parseable(raw_text: str, is_api_mode: bool, repair_hint: str = "") -> str:
    normalized = normalize_test_format(raw_text)
    validation = validate_test_structure(normalized)
    if validation.get("ok"):
        return validation.get("normalized_text", normalized)
    llm = get_llm(is_api_mode=is_api_mode, temperature=0.0, max_tokens=1400)
    errors = "\n".join(f"- {err}" for err in validation.get("errors", [])) or "- Неизвестная ошибка формата"
    hint = repair_hint.strip() if repair_hint.strip() else "Исправь только оформление теста."
    repair_prompt = f"""Ты — строгий валидатор и редактор тестов TransportExam AI.

ЗАДАЧА:
{hint}

Нужно вернуть ПОЛНЫЙ тест в едином формате. Не добавляй пояснений до или после теста.
Если блока "Ключи ответов:" нет, выбери правильный вариант по смыслу вопроса и вариантов, затем добавь ключи.
Если ключи уже есть, сохрани смысл вопросов и только исправь формат.

ОБНАРУЖЕННЫЕ ОШИБКИ:
{errors}

{TEST_FORMAT_RULES}

ТЕКУЩИЙ ТЕКСТ ТЕСТА:
{normalized}

Верни только исправленный тест. Последними строками обязательно должны быть:
Ключи ответов:
1-А
2-Б
... для всех вопросов.
"""
    repaired = normalize_test_format(invoke_llm_text(llm, repair_prompt))
    repaired_validation = validate_test_structure(repaired)
    if repaired_validation.get("ok"):
        return repaired_validation.get("normalized_text", repaired)
    return repaired or normalized

def validate_test_with_llm(text: str, is_api_mode: bool) -> Dict[str, Any]:
    structural = validate_test_structure(text)
    errors = "\n".join(f"- {err}" for err in structural.get("errors", [])) or "- Нет"
    warnings_text = "\n".join(f"- {warn}" for warn in structural.get("warnings", [])) or "- Нет"

    llm = get_llm(is_api_mode=is_api_mode, temperature=0.0, max_tokens=700)
    prompt = f"""Ты — LLM-валидатор оформления тестов для Streamlit-парсера.
Проверяй только формат, а не качество учебного содержания.

ПРАВИЛА ФОРМАТА:
{TEST_FORMAT_RULES}

РЕЗУЛЬТАТ БЫСТРОГО PYTHON-ПАРСЕРА:
Вопросов распознано парсером: {structural.get('questions_count')}
Сырых блоков вопросов найдено: {structural.get('raw_questions_count')}
Ошибки:
{errors}
Предупреждения:
{warnings_text}

ТЕКСТ ДЛЯ ПРОВЕРКИ:
{text}

Верни короткий отчёт строго по шаблону:
ИТОГ: КОРРЕКТНО или ЕСТЬ ОШИБКИ
ВОПРОСОВ НАЙДЕНО: число
ОШИБКИ:
- список ошибок или "нет"
РЕКОМЕНДАЦИЯ:
- что исправить в первую очередь
"""
    report = invoke_llm_text(llm, prompt)
    return {"report": report, "python_validation": structural}


def refine_test_action(current_test: str, instruction: str, is_api_mode: bool) -> str:
    current_count = max(1, len(extract_raw_question_blocks(current_test)))
    llm = get_llm(is_api_mode=is_api_mode, temperature=0.2, max_tokens=min(4096, max(1800, 260 * current_count + 800)))
    prompt = f"""Ты — эксперт-методист в сфере транспорта.
Измени тест по инструкции преподавателя.

ВАЖНО:
- Если преподаватель просит добавить вопросы, можно увеличить количество вопросов.
- После изменений перенумеруй все вопросы по порядку: 1., 2., 3. и так далее.
- У каждого вопроса должно быть ровно 4 варианта: А), Б), В), Г).
- В конце обязательно должен быть блок "Ключи ответов:" с ключом для каждого вопроса.
- Не добавляй пояснений вне теста.

ТЕКУЩИЙ ТЕСТ:
{current_test}

ИНСТРУКЦИЯ ПРЕПОДАВАТЕЛЯ:
{instruction}

{TEST_FORMAT_RULES}

Верни только обновлённый тест.
"""
    content = invoke_llm_text(llm, prompt)
    return ensure_test_is_parseable(
        content,
        is_api_mode=is_api_mode,
        repair_hint="Проверь результат доработки и обязательно восстанови финальный блок ключей ответов.",
    )
```

## File: agents/topic_agent.py
```python
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
```

## File: agents/utils.py
```python
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
```

## File: app.py
```python
import streamlit as st
from config import APP_ICON, APP_LAYOUT, APP_TITLE
from database import SessionLocal
from state import init_session_state
from ui.sidebar import render_sidebar
from ui.student_page import render_student_page
from ui.teacher_page import render_teacher_page

st.set_page_config(page_title=APP_TITLE, layout=APP_LAYOUT, page_icon=APP_ICON)

@st.cache_resource
def get_db_session():
    return SessionLocal()

def main():
    init_session_state()
    db = get_db_session()
    role, is_api_mode = render_sidebar()

    if role == "Преподаватель":
        render_teacher_page(db=db, is_api_mode=is_api_mode)
    else:
        render_student_page(db=db, is_api_mode=is_api_mode)

if __name__ == "__main__":
    main()
```

## File: config.py
```python
import os

APP_TITLE = "TransportExam AI"
APP_ICON = "🚆"
APP_LAYOUT = "wide"

MODEL_PATH = os.path.join("models", "student_performance_pipeline.joblib")
```

## File: docker-compose.yml
```yaml
version: '3.8'

services:
  db:
    image: postgres:15
    container_name: transport_db
    restart: always
    environment:
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: adminpassword
      POSTGRES_DB: transport_exam
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

## File: Dockerfile
```dockerfile
FROM ubuntu:latest
LABEL authors="delar"

ENTRYPOINT ["top", "-b"]
```

## File: domain_config.py
```python
DOMAIN_OPTIONS = [
    "Общая транспортная безопасность (Базовый)",
    "Автомобильный транспорт (ПДД, ДОПОГ, логистика дальнобойщиков)",
    "Железнодорожный транспорт (ПТЭ, инструкции по движению поездов)",
    "Гражданская авиация (ФАП, навигация, безопасность полетов)",
    "Морской и речной транспорт (МППСС, устав службы)",
]

DOMAIN_SPECIFICS = {
    "Общая транспортная безопасность (Базовый)": """
ФОКУС: безопасность пассажиров и грузов, антитеррористическая защищенность,
эвакуация, первая помощь, пожарная безопасность на объектах транспортной инфраструктуры.
СТИЛЬ ВОПРОСОВ: практические нештатные ситуации общего характера: бесхозный предмет,
досмотр, эвакуация, действия персонала при угрозе.
""",
    "Автомобильный транспорт (ПДД, ДОПОГ, логистика дальнобойщиков)": """
ФОКУС: ПДД РФ, режим труда и отдыха водителей, тахографы, ДОПОГ/ADR,
крепление грузов, габариты, весовые ограничения, логистические решения.
СТИЛЬ ВОПРОСОВ: практические задачи водителя и логиста: маркировка опасного груза,
поломка ТС, выбор безопасного маршрута, действия при ДТП.
""",
    "Железнодорожный транспорт (ПТЭ, инструкции по движению поездов)": """
ФОКУС: ПТЭ, ИСИ, ИДП, регламент переговоров, сигналы, маневровая работа,
действия машиниста и диспетчера при отказах, автоблокировка, экстренное торможение.
СТИЛЬ ВОПРОСОВ: строго регламентированные сценарии: сигнал светофора, приказ ДНЦ,
скорость маневров, порядок действий при неисправности.
""",
    "Гражданская авиация (ФАП, навигация, безопасность полетов)": """
ФОКУС: ФАП, Воздушный кодекс, безопасность полетов, предполетная подготовка,
аварийно-спасательные процедуры, CRM, метеоусловия, взаимодействие с УВД.
СТИЛЬ ВОПРОСОВ: алгоритмы действий экипажа, GO/NO-GO, дефицит времени,
разгерметизация, эвакуация, взаимодействие пилотов и диспетчеров.
""",
    "Морской и речной транспорт (МППСС, устав службы)": """
ФОКУС: МППСС-72, SOLAS, лоция, навигация, борьба за живучесть судна,
спасательные средства, швартовка, портовые правила.
СТИЛЬ ВОПРОСОВ: расхождение судов, навигационные огни и знаки, тревога
«человек за бортом», действия экипажа при аварийной ситуации.
""",
}


def get_domain_instructions(domain: str) -> str:
    return DOMAIN_SPECIFICS.get(domain, "Сформулируй профессиональные вопросы на основе текста.")
```

## File: llm_provider.py
```python
import os
from langchain_openai import ChatOpenAI
from langchain_gigachat import GigaChat

def read_api_key_from_file(*filenames: str) -> str:
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    for filename in filenames:
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                key = f.read().strip()
                if key:
                    return key
    return ""

GIGACHAT_AUTH_KEY = (
    os.getenv("GIGACHAT_AUTH_KEY")
    or read_api_key_from_file("gigachat_key.txt")
)

def check_gigachat_connection():
    if not GIGACHAT_AUTH_KEY:
        return False, "GigaChat: ключ не найден. Создайте файл gigachat_key.txt в корне проекта."
    try:
        llm = GigaChat(credentials=GIGACHAT_AUTH_KEY, verify_ssl_certs=False, scope="GIGACHAT_API_PERS")
        llm.invoke("Привет")
        return True, "GigaChat: Успешное подключение"
    except Exception as e:
        return False, f"GigaChat: Ошибка ({str(e)})"

def get_llm(is_api_mode: bool = False, temperature: float = 0.3, max_tokens: int = 2048):
    if is_api_mode:
        if not GIGACHAT_AUTH_KEY:
            raise ValueError("GigaChat API-ключ не найден. Создайте файл gigachat_key.txt в корне проекта.")

        return GigaChat(
            credentials=GIGACHAT_AUTH_KEY,
            verify_ssl_certs=False,
            scope="GIGACHAT_API_PERS",
            temperature=temperature,
            max_tokens=max_tokens,
        )

    return ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        model="local-model",
        temperature=temperature,
        max_tokens=max_tokens,
    )
```

## File: models/preprocessor.joblib
```

```

## File: models/student_performance_rf.joblib
```

```

## File: rag_pipeline.py
```python
import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

UPLOAD_DIR = "uploaded_docs"
CHROMA_DB_DIR = "chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small", model_kwargs={'device': 'cpu'})
vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)

def process_and_add_document(file_path, document_id):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        return False

    documents = loader.load()

    for doc in documents:
        doc.metadata["document_id"] = document_id

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(documents)

    vectorstore.add_documents(documents=splits)
    return True

def get_retriever_by_doc_id(document_id, k=3):
    return vectorstore.as_retriever(
        search_kwargs={'filter': {'document_id': document_id}, 'k': k}
    )
```

## File: requirements.txt
```
streamlit>=1.31.0
langchain
langchain-classic
langchain-community
langchain-core
langchain-openai
langchain-gigachat
chromadb
sentence-transformers
scikit-learn
pandas
numpy
joblib
python-dotenv
gigachat
pydub
SpeechRecognition
torch
transformers
pypdf
docx2txt
psycopg2
torchvision
```

## File: services/__init__.py
```python

```

## File: services/document_service.py
```python
import os
from database import Document
from rag_pipeline import UPLOAD_DIR, process_and_add_document

def get_all_documents(db):
    return db.query(Document).all()

def save_uploaded_document(db, uploaded_file):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    new_doc = Document(filename=uploaded_file.name, filepath=file_path)
    db.add(new_doc)
    db.commit()
    success = process_and_add_document(file_path, new_doc.id)
    return new_doc, success
```

## File: services/ml_service.py
```python
import os
import re
import joblib
import pandas as pd
from config import MODEL_PATH
from database import TestResult

def ml_model_exists() -> bool:
    return os.path.exists(MODEL_PATH)

def calculate_average_score(student_results):
    percentages = []
    for result in student_results:
        try:
            match = re.match(r"(\d+)\s*/\s*(\d+)", result.score)
            if match:
                correct = int(match.group(1))
                total = int(match.group(2))
                if total > 0:
                    percentages.append(correct / total)
        except Exception:
            pass
    if not percentages:
        return "Данные отсутствуют (студент еще не сдавал тесты в системе)"
    avg_percentage = (sum(percentages) / len(percentages)) * 100
    return f"{avg_percentage:.1f}% правильных ответов"

def build_performance_input(gender, topic, absence, raised_hands, visited_res, announcements, discussion):
    clean_absence = absence.split()[0]
    return pd.DataFrame([{
        "gender": gender,
        "NationalITy": "KW",
        "PlaceofBirth": "KuwaIT",
        "StageID": "MiddleSchool",
        "GradeID": "G-04",
        "SectionID": "A",
        "Topic": topic,
        "Semester": "F",
        "Relation": "Father",
        "raisedhands": raised_hands,
        "VisITedResources": visited_res,
        "AnnouncementsView": announcements,
        "Discussion": discussion,
        "ParentAnsweringSurvey": "Yes",
        "ParentschoolSatisfaction": "Good",
        "StudentAbsenceDays": clean_absence,
    }])

def predict_student_performance(input_df):
    pipeline = joblib.load(MODEL_PATH)
    return pipeline.predict(input_df)[0]


def get_student_results(db, student_name):
    return db.query(TestResult).filter(TestResult.student_name == student_name).all()
```

## File: services/result_service.py
```python
from database import Test, TestResult

def get_all_results(db):
    return db.query(TestResult).all()

def get_existing_student_names(db):
    return list(set([r.student_name for r in db.query(TestResult).all()]))

def evaluate_student_answers(parsed_questions, user_answers):
    correct_count = 0
    total_count = len(parsed_questions)
    report_lines = []
    detailed_list = []
    for q in parsed_questions:
        u_ans = user_answers[q["id"]]
        c_ans = q["correct"]
        is_ok = u_ans == c_ans
        if is_ok:
            correct_count += 1
        u_text = f"{u_ans}) {q['options'][u_ans]}"
        c_text = f"{c_ans}) {q['options'][c_ans]}"
        detailed_list.append({
            "id": q["id"],
            "question": q["question"],
            "user_ans": u_text,
            "correct_ans": c_text,
            "is_correct": is_ok,
        })
        report_lines.append(
            f"Вопрос {q['id']}: {q['question']}\n"
            f"Студент выбрал: {u_ans}, Правильно: {c_ans}. "
            f"Результат: {'ВЕРНО' if is_ok else 'ОШИБКА'}"
        )
    score_str = f"{correct_count}/{total_count}"
    return score_str, "\n".join(report_lines), detailed_list

def save_test_result(db, student_name: str, test_id: int, score: str):
    result = TestResult(student_name=student_name, test_id=test_id, score=score)
    db.add(result)
    db.commit()
    return result

def build_results_dataframe_rows(db):
    rows = []
    for result in get_all_results(db):
        test_obj = db.query(Test).filter(Test.id == result.test_id).first()
        rows.append({
            "ФИО Студента": result.student_name,
            "Название теста": test_obj.title if test_obj else "Удаленный тест",
            "Набранный балл": result.score,
            "Дата сдачи": result.completed_at.strftime("%d.%m.%Y %H:%M"),
        })
    return rows
```

## File: services/test_service.py
```python
from database import Test
from test_format import normalize_test_format, parse_test_questions, validate_test_structure

def parse_test_content(text):
    questions = parse_test_questions(text)
    return questions if questions else None

def get_all_tests(db):
    return db.query(Test).all()

def save_published_test(db, title: str, content: str, document_id: int):
    final_validation = validate_test_structure(content)
    if not final_validation.get("ok"):
        return None, final_validation
    normalized_content = final_validation.get("normalized_text", normalize_test_format(content))
    parsed = parse_test_content(normalized_content)
    if parsed is None:
        final_validation.setdefault("errors", []).append(
            "Ошибка парсера: тест формально прошёл проверку, но не отобразился как интерактивный тест."
        )
        final_validation["ok"] = False
        final_validation["is_valid"] = False
        return None, final_validation
    keys_part = normalized_content.split("Ключи ответов:", 1)[-1].strip()
    new_test = Test(
        title=title,
        content=normalized_content,
        correct_answers=keys_part,
        document_id=document_id,
    )
    db.add(new_test)
    db.commit()
    return new_test, final_validation
```

## File: speech_utils.py
```python
import io
import os
import tempfile
import uuid
import warnings
import wave
import requests
import torch
import urllib3
import streamlit as st
from transformers import pipeline

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="transformers")
urllib3.disable_warnings()


def read_api_key_from_file(*filenames: str) -> str:
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    for filename in filenames:
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                key = f.read().strip()
                if key:
                    return key
    return ""


SALUTESPEECH_AUTH_KEY = (
        os.getenv("SALUTESPEECH_AUTH_KEY")
        or read_api_key_from_file("salutespeech_key.txt", "salutespeech.key", "api_audio_key.txt")
)


@st.cache_resource(show_spinner=False)
def load_speech_models():
    print("Инициализация локальных аудиомоделей (Whisper + Silero)...")

    whisper_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device="cpu")
    device = torch.device("cpu")
    local_file = "model.pt"

    if not os.path.isfile(local_file):
        torch.hub.download_url_to_file("https://models.silero.ai/models/tts/ru/v4_ru.pt", local_file)

    silero_model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
    silero_model.to(device)

    return whisper_pipe, silero_model


def get_salute_token():
    if not SALUTESPEECH_AUTH_KEY:
        return None

    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    headers = {
        "Authorization": f"Basic {SALUTESPEECH_AUTH_KEY}",
        "RqUID": str(uuid.uuid4()),
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {"scope": "SALUTE_SPEECH_PERS"}
    try:
        response = requests.post(url, headers=headers, data=data, verify=False)
        response.raise_for_status()
        return response.json().get("access_token")
    except Exception:
        return None


def check_salutespeech_connection():
    if not SALUTESPEECH_AUTH_KEY:
        return False, "SaluteSpeech: ключ не найден. Создайте файл salutespeech_key.txt в корне проекта."

    token = get_salute_token()
    if token:
        return True, "SaluteSpeech: Токен получен"

    return False, "SaluteSpeech: Ошибка авторизации или неверный ключ"


def process_audio_stt(audio_bytes, is_api_mode=False):
    if not audio_bytes:
        return ""

    if is_api_mode:
        token = get_salute_token()
        if token:
            try:
                with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
                    sample_rate = wav_file.getframerate()
                    sampwidth = wav_file.getsampwidth()
                    raw_frames = wav_file.readframes(wav_file.getnframes())

                content_type = f"audio/x-pcm;bit={sampwidth * 8};rate={sample_rate}"
                url = "https://smartspeech.sber.ru/rest/v1/speech:recognize"
                headers = {"Authorization": f"Bearer {token}", "Content-Type": content_type}

                response = requests.post(url, headers=headers, data=raw_frames, verify=False)
                response.raise_for_status()
                return response.json().get("result", [""])[0]
            except Exception as e:
                print(f"Ошибка SaluteSpeech STT: {e}")
                return f"Ошибка Сбера: {str(e)}"
        return "Ошибка Сбера: API-ключ SaluteSpeech не найден или токен не получен"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_filename = temp_audio.name

    try:
        whisper_pipe, _ = load_speech_models()
        result = whisper_pipe(temp_filename, generate_kwargs={"language": "russian"})
        return result["text"].strip()
    finally:
        os.remove(temp_filename)


def generate_audio_tts(text, speaker="xenia", is_api_mode=False):
    clean_text = text.replace("**", "").replace("*", "").replace("#", "")
    if len(clean_text) > 800:
        clean_text = clean_text[:800] + "... Текст длинный, читайте на экране."

    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_out.close()

    if is_api_mode:
        token = get_salute_token()
        if token:
            url = "https://smartspeech.sber.ru/rest/v1/text:synthesize?format=wav16&voice=Nec_24000"
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/text"}
            try:
                response = requests.post(url, headers=headers, data=clean_text.encode("utf-8"), verify=False)
                response.raise_for_status()
                with open(temp_out.name, "wb") as f:
                    f.write(response.content)
                return temp_out.name
            except Exception as e:
                print(f"Ошибка SaluteSpeech TTS: {e}")

    try:
        _, silero_model = load_speech_models()
        silero_model.save_wav(text=clean_text, speaker=speaker, sample_rate=48000, audio_path=temp_out.name)
        return temp_out.name
    except Exception as e:
        print(f"Ошибка Silero: {e}")
        return None
```

## File: state.py
```python
import streamlit as st

def init_session_state():
    defaults = {
        "messages": [],
        "test_completed": False,
        "current_test_id": None,
        "auto_trigger": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_teacher_draft_state():
    keys_to_delete = [
        "teacher_topics",
        "teacher_topics_raw_context",
        "selected_teacher_topic",
        "generation_prompt",
        "generation_context",
        "draft_test_title",
        "draft_test_content",
        "draft_doc_id",
        "draft_validation",
        "proposed_test_content",
        "llm_test_validation_report",
        "llm_test_validation_structural",
        "draft_test_editor",
        "draft_test_editor_version",
        "proposed_test_editor_version",
        "generation_prompt_editor",
    ]
    for key in keys_to_delete:
        st.session_state.pop(key, None)
    for key in list(st.session_state.keys()):
        if key.startswith("draft_test_editor_") or key.startswith("proposed_test_editor_"):
            del st.session_state[key]

def reset_student_test_state(active_test_id):
    st.session_state.current_test_id = active_test_id
    st.session_state.test_completed = False
    st.session_state.messages = []
    st.session_state.student_score = ""
    st.session_state.answers_report = ""
    st.session_state.detailed_results = []
    for key in list(st.session_state.keys()):
        if key.startswith("q_"):
            del st.session_state[key]
```

## File: test_format.py
```python
import re
from typing import Any, Dict, List

TEST_FORMAT_RULES = """
ЕДИНЫЙ ФОРМАТ ТЕСТА:
1. Текст вопроса
А) Вариант ответа
Б) Вариант ответа
В) Вариант ответа
Г) Вариант ответа

2. Текст вопроса
А) Вариант ответа
Б) Вариант ответа
В) Вариант ответа
Г) Вариант ответа

Ключи ответов:
1-А
2-В

КРИТИЧЕСКИЕ ПРАВИЛА:
- Вопросы нумеруются строго как "1.", "2.", "3.".
- У каждого вопроса ровно 4 варианта: А), Б), В), Г).
- Правильный ответ нельзя раскрывать в тексте вопроса или рядом с вариантом.
- Все ответы указываются только в финальном блоке "Ключи ответов:".
- Блок "Ключи ответов:" обязателен и должен идти последним, без текста после него.
- Для каждого вопроса должен быть ровно один ключ в виде "номер-буква", например "1-А".
- Буквы вариантов и ключей должны быть кириллические: А, Б, В, Г.
"""

_LATIN_TO_CYR = str.maketrans({"A": "А", "B": "Б", "C": "В", "D": "Г"})


def normalize_answer_letter(letter: str) -> str:
    return letter.upper().translate(_LATIN_TO_CYR)


def normalize_test_format(text: str) -> str:
    if not text:
        return ""

    clean = text.replace("\r\n", "\n").replace("\r", "\n")
    clean = clean.replace("**", "").replace("*", "").replace("`", "")

    clean = re.sub(
        r"(?im)^\s*(?:Ключи\s+ответов|Правильные\s+ответы|Ответы|Ключи)\s*:\s*",
        "Ключи ответов:\n",
        clean,
    )

    clean = re.sub(r"(?im)^\s*Вопрос\s*№?\s*(\d+)\s*[:\.-]\s*", r"\1. ", clean)

    def repl_option(match):
        return f"{normalize_answer_letter(match.group(1))}) {match.group(2).strip()}"

    clean = re.sub(r"(?im)^\s*([ABCDАБВГ])\s*[\)\.\:\-]\s*(.+)$", repl_option, clean)

    def repl_key(match):
        return f"{match.group(1)}-{normalize_answer_letter(match.group(2))}"

    clean = re.sub(r"(?im)^\s*(\d+)\s*[\-\.\:\)]\s*([ABCDАБВГ])\s*$", repl_key, clean)

    if not re.search(r"(?is)Ключи ответов:\s*", clean):
        inline_keys: Dict[int, str] = {}
        question_pattern = re.compile(r"(?ms)^\s*(\d+)\s*[\.)]\s*(.*?)(?=^\s*\d+\s*[\.)]\s*|\Z)")
        inline_answer_pattern = re.compile(
            r"(?im)^\s*(?:Правильный\s+ответ|Верный\s+ответ|Ответ)\s*[:\-]\s*([ABCDАБВГ])\s*\.?\s*$"
        )
        for q_match in question_pattern.finditer(clean):
            q_num = int(q_match.group(1))
            inline_match = inline_answer_pattern.search(q_match.group(2))
            if inline_match:
                inline_keys[q_num] = normalize_answer_letter(inline_match.group(1))

        if inline_keys:
            clean = inline_answer_pattern.sub("", clean)
            keys_block = "\n".join(f"{qid}-{letter}" for qid, letter in sorted(inline_keys.items()))
            clean = f"{clean.strip()}\n\nКлючи ответов:\n{keys_block}"

    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    return clean

def extract_answer_keys(text: str) -> Dict[int, str]:
    clean = normalize_test_format(text)
    key_match = re.search(r"(?is)Ключи ответов:\s*(.+)$", clean)
    if not key_match:
        return {}

    keys: Dict[int, str] = {}
    for match in re.finditer(r"(?im)^\s*(\d+)\s*[\-\.\:\)]\s*([АБВГ])\s*$", key_match.group(1)):
        keys[int(match.group(1))] = match.group(2).upper()
    return keys


def extract_raw_question_blocks(text: str) -> List[Dict[str, Any]]:
    clean = normalize_test_format(text)
    key_header = re.search(r"(?is)\n?\s*Ключи ответов:\s*", clean)
    main_part = clean[:key_header.start()].strip() if key_header else clean.strip()

    expected = ["А", "Б", "В", "Г"]
    raw_questions: List[Dict[str, Any]] = []
    question_pattern = re.compile(r"(?ms)^\s*(\d+)\s*[\.)]\s*(.*?)(?=^\s*\d+\s*[\.)]\s*|\Z)")

    for q_match in question_pattern.finditer(main_part):
        q_num = int(q_match.group(1))
        block = q_match.group(2).strip()
        option_matches = list(re.finditer(r"(?im)^\s*([АБВГ])\)\s*(.+)$", block))
        letters = [m.group(1).upper() for m in option_matches]
        options = {m.group(1).upper(): m.group(2).strip() for m in option_matches}
        first_option_start = option_matches[0].start() if option_matches else len(block)
        q_text = re.sub(r"\s+", " ", block[:first_option_start].strip())

        raw_questions.append({
            "id": q_num,
            "question": q_text,
            "options": options,
            "letters": letters,
            "missing_options": [letter for letter in expected if letter not in letters],
            "duplicate_options": sorted({letter for letter in letters if letters.count(letter) > 1}),
        })

    return raw_questions

def parse_test_questions(text: str) -> List[Dict[str, Any]]:
    clean = normalize_test_format(text)
    key_header = re.search(r"(?is)\n?\s*Ключи ответов:\s*", clean)
    if not key_header:
        return []

    keys = extract_answer_keys(clean)
    questions: List[Dict[str, Any]] = []

    for raw in extract_raw_question_blocks(clean):
        q_num = raw["id"]
        options = raw["options"]
        if raw["question"] and set(options.keys()) == {"А", "Б", "В", "Г"} and q_num in keys:
            questions.append({
                "id": q_num,
                "question": raw["question"],
                "options": options,
                "correct": keys[q_num],
            })

    questions.sort(key=lambda x: x["id"])
    return questions

def validate_test_structure(text: str, min_questions: int = 1, max_questions: int = 50) -> Dict[str, Any]:
    clean = normalize_test_format(text)
    raw_questions = extract_raw_question_blocks(clean)
    questions = parse_test_questions(clean)
    keys = extract_answer_keys(clean)
    errors: List[str] = []
    warnings: List[str] = []

    if "Ключи ответов:" not in clean:
        errors.append("Не найден блок 'Ключи ответов:'.")

    if len(raw_questions) < min_questions:
        errors.append(f"Найдено слишком мало вопросов: {len(raw_questions)}. Минимум: {min_questions}.")

    if len(raw_questions) > max_questions:
        warnings.append(f"Найдено много вопросов: {len(raw_questions)}. Проверьте, удобно ли проходить такой тест студенту.")

    question_ids = [q["id"] for q in raw_questions]
    expected_ids = list(range(1, len(raw_questions) + 1))
    if question_ids and question_ids != expected_ids:
        warnings.append("Нумерация вопросов не последовательная. Лучше использовать 1., 2., 3. без пропусков.")

    if len(question_ids) != len(set(question_ids)):
        errors.append("Есть повторяющиеся номера вопросов.")

    for q in raw_questions:
        if not q["question"]:
            errors.append(f"Вопрос {q['id']}: не найден текст вопроса перед вариантами ответа.")
        if q["missing_options"]:
            errors.append(f"Вопрос {q['id']}: отсутствуют варианты {', '.join(q['missing_options'])}.")
        if q["duplicate_options"]:
            errors.append(f"Вопрос {q['id']}: повторяются варианты {', '.join(q['duplicate_options'])}.")
        if set(q["options"].keys()) != {"А", "Б", "В", "Г"}:
            errors.append(f"Вопрос {q['id']}: должны быть ровно варианты А), Б), В), Г).")

    missing_keys = [qid for qid in question_ids if qid not in keys]
    if missing_keys:
        errors.append("Нет ключей ответов для вопросов: " + ", ".join(map(str, missing_keys)))

    extra_keys = [qid for qid in keys if qid not in question_ids]
    if extra_keys:
        warnings.append("Есть лишние ключи ответов без вопросов: " + ", ".join(map(str, extra_keys)))

    for qid, letter in keys.items():
        if letter not in {"А", "Б", "В", "Г"}:
            errors.append(f"Вопрос {qid}: некорректный ключ ответа '{letter}'.")

    return {
        "ok": len(errors) == 0,
        "is_valid": len(errors) == 0,
        "questions_count": len(questions),
        "raw_questions_count": len(raw_questions),
        "errors": errors,
        "warnings": warnings,
        "normalized_text": clean,
    }
```

## File: train_ml_model.py
```python
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

DATA_PATH = "data/xAPI-Edu-Data.csv"
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "student_performance_pipeline.joblib")

os.makedirs(MODELS_DIR, exist_ok=True)

if not os.path.exists(DATA_PATH):
    print(f"Критическая ошибка: Файл датасета не найден!")
    print(f"Пожалуйста, поместите скачанный файл датасета по пути: '{DATA_PATH}'")
    exit(1)

print(f"[*] Загрузка оригинального датасета из {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

print("[*] Датасет успешно загружен. Форма данных:", df.shape)

def main():

    df = pd.DataFrame(pd.read_csv(DATA_PATH))

    print("\nРаспределение классов (L=Fail, M=Pass, H=High Distinction):")
    print(df['Class'].value_counts())

    X = df.drop('Class', axis=1)
    y = df['Class']

    categorical_features = [
        'gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID',
        'SectionID', 'Topic', 'Semester', 'Relation', 'ParentAnsweringSurvey',
        'ParentschoolSatisfaction', 'StudentAbsenceDays'
    ]
    numeric_features = ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\n[*] Начинаем обучение RandomForestClassifier...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print("\n" + "=" * 40)
    print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ МОДЕЛИ:")
    print("=" * 40)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nМодель успешно обучена и сохранена по пути: {MODEL_PATH}")


if __name__ == "__main__":
    main()
```

## File: ui/__init__.py
```python

```

## File: ui/components.py
```python
import os
import streamlit as st
from speech_utils import generate_audio_tts

def render_tts_button(text: str, key: str, is_api_mode: bool, speaker: str = "xenia"):
    if st.button("🔊", key=key, help="Прослушать вопрос"):
        with st.spinner("..."):
            audio_path = generate_audio_tts(text, speaker=speaker, is_api_mode=is_api_mode)
            if audio_path:
                with open(audio_path, "rb") as f:
                    st.audio(f.read(), format="audio/wav")
                os.remove(audio_path)

def render_validation_messages(validation):
    if validation.get("ok"):
        st.success(f"Формат теста корректен. Найдено вопросов: {validation.get('questions_count')}.")
    else:
        st.warning("Python-валидатор нашёл проблемы в формате теста:")
        for err in validation.get("errors", []):
            st.write(f"- {err}")
    for warn in validation.get("warnings", []):
        st.info(f"Предупреждение: {warn}")
```

## File: ui/sidebar.py
```python
import streamlit as st
from llm_provider import check_gigachat_connection
from speech_utils import check_salutespeech_connection

def render_sidebar():
    st.sidebar.title("Настройки Системы")
    role = st.sidebar.radio("Выберите роль пользователя:", ["Студент", "Преподаватель"])
    st.sidebar.divider()
    st.sidebar.subheader("Режим работы ИИ")
    engine_mode = st.sidebar.radio(
        "Архитектура нейросетей:",
        ["Локальный (LM Studio + Whisper)", "Облачный API (GigaChat + Sber)"],
    )
    is_api_mode = engine_mode == "Облачный API (GigaChat + Sber)"
    if is_api_mode:
        if st.sidebar.button("Проверить подключение к API"):
            with st.sidebar.status("Проверка токенов доступа...", expanded=True) as status:
                giga_ok, giga_msg = check_gigachat_connection()
                st.write(giga_msg)
                sber_ok, sber_msg = check_salutespeech_connection()
                st.write(sber_msg)
                if giga_ok and sber_ok:
                    status.update(label="API системы работают стабильно!", state="complete")
                else:
                    status.update(label="Обнаружены проблемы с API!", state="error")
    st.sidebar.divider()
    return role, is_api_mode
```

## File: ui/student_page.py
```python
import os
import streamlit as st
from agents import chat_with_student_rag, run_revision_plan_agent
from rag_pipeline import get_retriever_by_doc_id
from services.result_service import evaluate_student_answers, save_test_result
from services.test_service import get_all_tests, parse_test_content
from speech_utils import generate_audio_tts, process_audio_stt
from state import reset_student_test_state

def render_student_page(db, is_api_mode: bool):
    st.title("Обучающий интерфейс Студента")
    student_name = st.text_input("Введите ваше ФИО для фиксации результатов:", placeholder="Например: Иванов И.И.")
    all_tests = get_all_tests(db)
    if not all_tests:
        st.info("В системе пока нет созданных тестов. Попросите преподавателя сгенерировать тест.")
        return
    test_options = {t.title: t for t in all_tests}
    selected_test_title = st.selectbox("Выберите доступный квалификационный тест:", list(test_options.keys()))
    active_test = test_options[selected_test_title]
    if st.session_state.get("current_test_id") != active_test.id:
        reset_student_test_state(active_test.id)
    if not st.session_state.test_completed:
        render_test_passing_screen(db, active_test, student_name, is_api_mode)
    else:
        render_student_result_screen(active_test, is_api_mode)

def render_test_passing_screen(db, active_test, student_name: str, is_api_mode: bool):
    st.subheader(f"Прохождение теста: {active_test.title}")

    parsed_questions = parse_test_content(active_test.content)
    if not parsed_questions:
        st.error("Ошибка структуры теста. Формат файла поврежден, невозможно отобразить интерактивные вопросы.")
        st.text_area("Сырой текст теста:", value=active_test.content, height=300, disabled=True)
        return
    user_answers = {}
    for q in parsed_questions:
        render_question(active_test.id, q, user_answers, is_api_mode)
    st.write("---")
    if st.button("Завершить тест и проверить", type="primary"):
        if not student_name.strip():
            st.error("Пожалуйста, введите ваше ФИО перед отправкой теста!")
        elif None in user_answers.values():
            st.warning("Вы ответили не на все вопросы. Пожалуйста, завершите квиз полностью!")
        else:
            score_str, answers_report, detailed_list = evaluate_student_answers(parsed_questions, user_answers)
            save_test_result(db, student_name, active_test.id, score_str)
            st.session_state.student_score = score_str
            st.session_state.answers_report = answers_report
            st.session_state.detailed_results = detailed_list
            st.session_state.test_completed = True
            st.rerun()

def render_question(test_id: int, q, user_answers, is_api_mode: bool):
    col_q, col_audio = st.columns([0.9, 0.1])
    with col_q:
        st.write(f"**Вопрос {q['id']}:** {q['question']}")
    with col_audio:
        if st.button("🔊", key=f"play_q_{test_id}_{q['id']}", help="Прослушать вопрос"):
            with st.spinner("..."):
                audio_path = generate_audio_tts(q["question"], speaker="xenia", is_api_mode=is_api_mode)
                if audio_path:
                    with open(audio_path, "rb") as f:
                        st.audio(f.read(), format="audio/wav")
                    os.remove(audio_path)

    opt_labels = [f"{letter}) {text}" for letter, text in q["options"].items()]
    selected_opt_full = st.radio(
        f"Выберите ответ на вопрос {q['id']}:",
        opt_labels,
        index=None,
        key=f"q_{test_id}_{q['id']}",
    )
    user_answers[q["id"]] = selected_opt_full[0].upper() if selected_opt_full else None

def render_student_result_screen(active_test, is_api_mode: bool):
    st.success(f"Тест завершен! Ваш результат: **{st.session_state.student_score}**")
    st.subheader("Детализация ваших ответов")
    for res in st.session_state.detailed_results:
        if res["is_correct"]:
            st.success(f"**Вопрос {res['id']}:** {res['question']}\n\n✅ **Ваш ответ:** {res['user_ans']}")
        else:
            st.error(
                f"**Вопрос {res['id']}:** {res['question']}\n\n"
                f"❌ **Ваш ответ:** {res['user_ans']}\n\n"
                f"✔️ **Правильный ответ:** {res['correct_ans']}"
            )
    st.divider()
    render_revision_agent_chat(active_test, is_api_mode)

def render_revision_agent_chat(active_test, is_api_mode: bool):
    st.subheader("Агент персонального плана повторения")
    if len(st.session_state.messages) == 0:
        if st.button("Составить персональный план повторения", type="primary"):
            auto_query = "Составь персональный план повторения по моим ошибкам."
            st.session_state.messages.append({
                "role": "user",
                "content": auto_query,
                "message_type": "revision_plan_request",
            })
            st.rerun()
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant":
                if st.button("🔊 Озвучить вывод", key=f"speak_assistant_{idx}"):
                    with st.spinner("Генерирую голосовой ответ..."):
                        audio_file_path = generate_audio_tts(
                            msg["content"],
                            speaker="xenia",
                            is_api_mode=is_api_mode,
                        )
                        if audio_file_path:
                            with open(audio_file_path, "rb") as f:
                                st.session_state.messages[idx]["audio"] = f.read()
                            os.remove(audio_file_path)
                if msg.get("audio"):
                    st.audio(msg["audio"], format="audio/wav")
    if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
        user_text = st.session_state.messages[-1]["content"]
        user_message_type = st.session_state.messages[-1].get("message_type", "student_followup")
        with st.chat_message("assistant"):
            retriever = get_retriever_by_doc_id(active_test.document_id)
            if user_message_type == "revision_plan_request":
                with st.spinner("Агент анализирует ошибки и ищет правила в учебном документе..."):
                    response = run_revision_plan_agent(
                        retriever=retriever,
                        student_score=st.session_state.student_score,
                        is_api_mode=is_api_mode,
                        answers_report=st.session_state.answers_report,
                        user_input=user_text,
                    )
            else:
                with st.spinner("Ищу ответ в учебном документе с учётом истории диалога..."):
                    response = chat_with_student_rag(
                        retriever=retriever,
                        current_test="",
                        student_score=st.session_state.student_score,
                        user_input=user_text,
                        is_api_mode=is_api_mode,
                        answers_report=st.session_state.answers_report,
                        chat_history=st.session_state.messages[:-1],
                    )
            st.markdown(response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "audio": None,
            })
            st.rerun()
    if len(st.session_state.messages) > 0:
        student_input = st.chat_input("Уточните, какую тему или ошибку разобрать подробнее...")
        student_audio = st.audio_input("Или задайте вопрос голосом:")
        user_text = None
        if student_input:
            user_text = student_input
        elif student_audio:
            audio_bytes = student_audio.getvalue()
            if st.session_state.get("last_student_audio") != audio_bytes:
                st.session_state["last_student_audio"] = audio_bytes
                user_text = process_audio_stt(audio_bytes, is_api_mode=is_api_mode)
        if user_text:
            st.session_state.messages.append({
                "role": "user",
                "content": user_text,
                "message_type": "student_followup",
            })
            st.rerun()
```

## File: ui/teacher_page.py
```python
import pandas as pd
import streamlit as st
from langchain_core.messages import HumanMessage
from agents import (
    analyze_document_topics_agent,
    build_prompt_for_selected_topic,
    generate_test_from_prompt,
    refine_test_action,
    validate_test_with_llm,
)
from config import MODEL_PATH
from domain_config import DOMAIN_OPTIONS
from llm_provider import get_llm
from rag_pipeline import get_retriever_by_doc_id
from services.document_service import get_all_documents, save_uploaded_document
from services.ml_service import (
    build_performance_input,
    calculate_average_score,
    get_student_results,
    ml_model_exists,
    predict_student_performance,
)
from services.result_service import build_results_dataframe_rows, get_all_results, get_existing_student_names
from services.test_service import parse_test_content, save_published_test
from state import reset_teacher_draft_state
from test_format import normalize_test_format, validate_test_structure
from ui.components import render_validation_messages
from speech_utils import process_audio_stt

def render_teacher_page(db, is_api_mode: bool):
    st.title("Панель Управления Преподавателя")
    tab1, tab2, tab3 = st.tabs([
        "Управление материалами и тестами",
        "Результаты тестирования студентов",
        "Радар успеваемости (ML Аналитика)",
    ])
    with tab1:
        render_materials_and_tests_tab(db, is_api_mode)
    with tab2:
        render_results_tab(db)
    with tab3:
        render_ml_radar_tab(db, is_api_mode)

def render_materials_and_tests_tab(db, is_api_mode: bool):
    st.subheader("1. Загрузка учебных регламентов и кодексов")
    uploaded_file = st.file_uploader("Загрузите файл (PDF, DOCX, TXT):", type=["pdf", "docx", "txt"])
    if uploaded_file is not None:
        if st.button("Обработать и добавить в учебную базу", type="primary"):
            with st.spinner("Индексирование документа и генерация эмбеддингов в ChromaDB..."):
                new_doc, success = save_uploaded_document(db, uploaded_file)
            if success:
                st.success(f"Документ '{new_doc.filename}' успешно загружен и векторизован!")
                reset_teacher_draft_state()
            else:
                st.error("Ошибка при обработке структуры документа.")

    st.divider()
    st.subheader("2. Агент подготовки теста: темы → промпт → тест")

    documents = get_all_documents(db)
    if not documents:
        st.info("Сначала загрузите хотя бы один учебный документ.")
        return

    doc_options = {doc.filename: doc for doc in documents}
    selected_doc_name = st.selectbox("Выберите документ-источник для теста:", list(doc_options.keys()))
    selected_doc = doc_options[selected_doc_name]

    domain = st.selectbox("Отраслевая специфика теста:", DOMAIN_OPTIONS)

    col_count, col_diff = st.columns([1, 2])
    with col_count:
        question_count = st.number_input("Количество вопросов:", min_value=1, max_value=20, value=3, step=1)
    with col_diff:
        difficulty = st.selectbox("Уровень сложности:", ["Базовый", "Средний", "Сложный"])

    test_title = st.text_input("Название нового теста:", placeholder="Например: Итоговый квиз по ПТЭ ЖД")
    extra_requirements = st.text_area(
        "Дополнительные требования к вопросам (необязательно):",
        placeholder="Например: сделай упор на правила тушения пожаров в локомотивах",
    )

    render_topic_step(selected_doc, domain, question_count, difficulty, test_title, extra_requirements, is_api_mode)
    render_prompt_step(selected_doc, domain, question_count, difficulty, extra_requirements, is_api_mode)
    render_draft_step(db, selected_doc, test_title, is_api_mode)

def render_topic_step(selected_doc, domain, question_count, difficulty, test_title, extra_requirements, is_api_mode):
    st.markdown("### Шаг 1. Агент анализирует документ и предлагает темы")
    if st.button("Проанализировать документ и предложить темы", type="primary"):
        if not test_title.strip():
            st.error("Пожалуйста, введите название теста перед анализом.")
        else:
            with st.spinner("Агент анализирует документ через RAG и выделяет темы..."):
                retriever = get_retriever_by_doc_id(selected_doc.id)
                topics, raw_context = analyze_document_topics_agent(
                    retriever=retriever,
                    is_api_mode=is_api_mode,
                    domain=domain,
                )

            st.session_state.teacher_topics = topics
            st.session_state.teacher_topics_raw_context = raw_context
            st.session_state.draft_test_title = test_title
            st.session_state.draft_doc_id = selected_doc.id
            st.session_state.draft_domain = domain
            st.session_state.draft_question_count = int(question_count)
            st.session_state.draft_difficulty = difficulty
            st.session_state.draft_extra_requirements = extra_requirements

            for key in [
                "selected_teacher_topic",
                "generation_prompt",
                "generation_context",
                "draft_test_content",
                "draft_validation",
                "proposed_test_content",
                "llm_test_validation_report",
                "llm_test_validation_structural",
                "draft_test_editor",
                "draft_test_editor_version",
                "proposed_test_editor_version",
                "generation_prompt_editor",
            ]:
                st.session_state.pop(key, None)

            st.rerun()


def render_prompt_step(selected_doc, domain, question_count, difficulty, extra_requirements, is_api_mode):
    if "teacher_topics" not in st.session_state:
        return

    st.success("Агент предложил темы для теста.")
    topics = st.session_state.teacher_topics

    if not topics:
        st.warning("Темы не удалось выделить автоматически. Можно вручную ввести тему ниже.")
        selected_topic = st.text_input("Введите тему теста вручную:")
    else:
        selected_topic = st.radio(
            "Выберите тему, по которой будет создан тест:",
            topics,
            key="selected_teacher_topic_radio",
        )

    custom_topic = st.text_input(
        "Или уточните/замените тему вручную:",
        value=selected_topic if selected_topic else "",
    )
    final_topic = custom_topic.strip() or selected_topic

    st.markdown("### Шаг 2. Сформировать промпт по выбранной теме")
    if st.button("Сформировать промпт для генерации теста"):
        if not final_topic:
            st.error("Сначала выберите или введите тему.")
        else:
            with st.spinner("Система ищет основные положения по теме и собирает промпт..."):
                retriever = get_retriever_by_doc_id(selected_doc.id)
                prompt_text, topic_context = build_prompt_for_selected_topic(
                    retriever=retriever,
                    topic=final_topic,
                    domain=domain,
                    question_count=int(question_count),
                    difficulty=difficulty,
                    extra_requirements=extra_requirements,
                )

            st.session_state.selected_teacher_topic = final_topic
            st.session_state.generation_prompt = prompt_text
            st.session_state.generation_context = topic_context
            st.session_state.pop("generation_prompt_editor", None)
            st.rerun()

    if "generation_prompt" in st.session_state:
        st.markdown("### Шаг 3. Проверить и при необходимости отредактировать промпт")
        prompt_for_generation = st.text_area(
            "Промпт, который будет отправлен в LLM:",
            value=st.session_state.generation_prompt,
            height=320,
            key="generation_prompt_editor",
        )

        if st.button("⚡ Сгенерировать тест по утверждённому промпту", type="primary"):
            with st.spinner("LLM генерирует тест..."):
                test_content = generate_test_from_prompt(prompt_for_generation, is_api_mode=is_api_mode)
                validation = validate_test_structure(test_content)

            st.session_state.draft_test_content = validation.get("normalized_text", test_content)
            st.session_state.draft_validation = validation
            st.session_state.draft_test_editor_version = st.session_state.get("draft_test_editor_version", 0) + 1
            st.session_state.pop("llm_test_validation_report", None)
            st.rerun()


def render_draft_step(db, selected_doc, test_title, is_api_mode):
    if "draft_test_content" not in st.session_state:
        return

    st.write("---")
    st.subheader("Шаг 4. Проверка и доработка черновика теста")

    if "draft_test_editor_version" not in st.session_state:
        st.session_state.draft_test_editor_version = 0
    draft_editor_key = f"draft_test_editor_{st.session_state.draft_test_editor_version}"

    edited_content = st.text_area(
        "Текущий черновик теста (можно править вручную):",
        value=st.session_state.draft_test_content,
        height=380,
        key=draft_editor_key,
    )

    current_validation = validate_test_structure(edited_content)
    render_validation_messages(current_validation)

    col_manual_apply, col_llm_check = st.columns(2)
    with col_manual_apply:
        if st.button("Применить ручные правки к черновику"):
            normalized_manual = current_validation.get("normalized_text", normalize_test_format(edited_content))
            st.session_state.draft_test_content = normalized_manual
            st.session_state.draft_test_editor_version = st.session_state.get("draft_test_editor_version", 0) + 1
            st.session_state.pop("llm_test_validation_report", None)
            st.rerun()

    with col_llm_check:
        if st.button("Проверить тест на корректность оформления через ИИ"):
            with st.spinner("LLM проверяет оформление теста..."):
                llm_validation = validate_test_with_llm(edited_content, is_api_mode=is_api_mode)
            st.session_state.llm_test_validation_report = llm_validation.get("report", "")
            st.session_state.llm_test_validation_structural = llm_validation.get("python_validation", {})
            st.rerun()

    if "llm_test_validation_report" in st.session_state:
        st.markdown("#### Отчёт LLM-валидатора")
        st.info(st.session_state.llm_test_validation_report)

    render_refine_block(edited_content, is_api_mode)
    render_save_block(db, selected_doc, test_title, edited_content)


def render_refine_block(edited_content, is_api_mode):
    st.markdown("#### Доработка теста с помощью ИИ")
    refine_instruction = st.text_area(
        "Что нужно изменить, улучшить или добавить в этот тест?",
        placeholder="Например: добавь ещё 2 сложных вопроса и перенумеруй ключи ответов",
    )
    refine_audio = st.audio_input("Или надиктуйте изменения голосом:")

    if refine_audio:
        recognized_text = process_audio_stt(refine_audio.getvalue(), is_api_mode=is_api_mode)
        st.info(f"Распознанный текст: {recognized_text}")
        refine_instruction = recognized_text

    if st.button("Сгенерировать изменения", type="secondary"):
        if not refine_instruction.strip():
            st.error("Пожалуйста, введите инструкции для ИИ или запишите аудио.")
        else:
            with st.spinner("ИИ дорабатывает тест по вашему запросу..."):
                proposed = refine_test_action(
                    current_test=edited_content,
                    instruction=refine_instruction,
                    is_api_mode=is_api_mode,
                )
            st.session_state.proposed_test_content = proposed
            st.session_state.proposed_test_editor_version = st.session_state.get("proposed_test_editor_version", 0) + 1
            st.session_state.pop("llm_test_validation_report", None)
            st.rerun()

    if "proposed_test_content" in st.session_state:
        st.success("Изменения сгенерированы. Проверьте предложенный вариант ниже.")
        if "proposed_test_editor_version" not in st.session_state:
            st.session_state.proposed_test_editor_version = 0
        proposed_editor_key = f"proposed_test_editor_{st.session_state.proposed_test_editor_version}"

        proposed_content = st.text_area(
            "Предложенный ИИ вариант:",
            value=st.session_state.proposed_test_content,
            height=350,
            key=proposed_editor_key,
        )

        proposed_validation = validate_test_structure(proposed_content)
        if proposed_validation.get("ok"):
            st.success(f"Предложенный вариант корректен. Вопросов: {proposed_validation.get('questions_count')}.")
        else:
            st.warning("В предложенном варианте есть проблемы:")
            for err in proposed_validation.get("errors", []):
                st.write(f"- {err}")

        col_apply, col_cancel = st.columns(2)
        with col_apply:
            if st.button("Применить предложенный вариант", type="primary"):
                normalized_proposed = proposed_validation.get("normalized_text", normalize_test_format(proposed_content))
                st.session_state.draft_test_content = normalized_proposed
                st.session_state.draft_test_editor_version = st.session_state.get("draft_test_editor_version", 0) + 1
                st.session_state.pop("proposed_test_content", None)
                st.session_state.pop("llm_test_validation_report", None)
                st.rerun()
        with col_cancel:
            if st.button("Отклонить предложенный вариант"):
                st.session_state.pop("proposed_test_content", None)
                st.session_state.proposed_test_editor_version = st.session_state.get("proposed_test_editor_version", 0) + 1
                st.rerun()


def render_save_block(db, selected_doc, test_title, edited_content):
    st.write("---")
    st.markdown("### Шаг 5. Сохранение теста")
    if st.button("Утвердить и опубликовать тест для студентов"):
        title = st.session_state.get("draft_test_title", test_title)
        document_id = st.session_state.get("draft_doc_id", selected_doc.id)
        new_test, final_validation = save_published_test(db, title, edited_content, document_id)

        if not final_validation.get("ok"):
            st.error("Тест нельзя сохранить: структура нарушена.")
            for err in final_validation.get("errors", []):
                st.write(f"- {err}")
        elif new_test is None or parse_test_content(final_validation.get("normalized_text", edited_content)) is None:
            st.error("Ошибка парсера: тест формально прошёл проверку, но не отобразился как интерактивный тест.")
        else:
            st.success(f"Тест '{new_test.title}' успешно сохранен в PostgreSQL и доступен студентам!")
            reset_teacher_draft_state()
            st.rerun()


def render_results_tab(db):
    st.subheader("Журнал успеваемости студентов")
    results = get_all_results(db)
    if not results:
        st.info("Студенты еще не проходили тестирование.")
    else:
        st.dataframe(pd.DataFrame(build_results_dataframe_rows(db)))


def render_ml_radar_tab(db, is_api_mode: bool):
    st.subheader("Симулятор рисков успеваемости («What-If» анализ)")
    st.write("Инструмент предиктивного анализа на базе встроенной ML-модели `RandomForestClassifier`.")

    if not ml_model_exists():
        st.error(
            f"Файл обученной ML-модели не найден по пути `{MODEL_PATH}`! "
            f"Пожалуйста, запустите скрипт `train_ml_model.py` для генерации пайплайна."
        )
        return

    existing_students = get_existing_student_names(db)

    col_st, _ = st.columns([2, 2])
    with col_st:
        is_manual = st.checkbox("Ввести ФИО вручную (новый студент)")
        if is_manual or not existing_students:
            target_student = st.text_input("Введите ФИО анализируемого студента:", value="Петров В.В.")
        else:
            target_student = st.selectbox("Выберите студента из журнала успеваемости:", existing_students)

    st.write("---")
    st.markdown(f"### Заполнение карты факторов активности для: **{target_student}**")

    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Пол студента (gender):", ["M", "F"])
        topic = st.selectbox("Учебная дисциплина (Topic):", ["Transport", "IT", "Math", "Science"])
        absence = st.selectbox(
            "Количество пропусков занятий (StudentAbsenceDays):",
            ["Under-7 (Мало)", "Above-7 (Много)"],
        )
    with c2:
        raised_hands = st.slider("Сколько раз поднимал руку (raisedhands):", 0, 100, 45)
        visited_res = st.slider("Просмотр электронных учебников (VisITedResources):", 0, 100, 60)
    with c3:
        announcements = st.slider("Проверка объявлений системы (AnnouncementsView):", 0, 100, 30)
        discussion = st.slider("Активность на форумах/семинарах (Discussion):", 0, 100, 25)

    if st.button("Рассчитать прогноз успеваемости", type="primary"):
        student_results = get_student_results(db, target_student)
        avg_score_text = calculate_average_score(student_results)
        input_df = build_performance_input(gender, topic, absence, raised_hands, visited_res, announcements, discussion)

        try:
            st.session_state.ml_prediction = predict_student_performance(input_df)
            st.session_state.ml_avg_score = avg_score_text
            st.session_state.show_ai_advice = False
        except Exception as ex:
            st.error(f"Ошибка прогнозирования: {str(ex)}")

    if "ml_prediction" in st.session_state:
        render_ml_prediction_result(is_api_mode, absence, raised_hands, visited_res, discussion)


def render_ml_prediction_result(is_api_mode, absence, raised_hands, visited_res, discussion):
    prediction = st.session_state.ml_prediction
    avg_score_text = st.session_state.ml_avg_score

    st.write("---")
    st.subheader("Результаты комплексного моделирования")
    st.metric(label="Реальная накопленная успеваемость студента в БД:", value=avg_score_text)

    if prediction == "H":
        st.success("### Прогноз модели: Класс H (Высокая успеваемость)")
    elif prediction == "M":
        st.warning("### Прогноз модели: Класс M (Средняя успеваемость)")
    else:
        st.error("### Прогноз модели: Класс L (Низкая успеваемость / Зона Риска)")

    if st.button("Получить методические рекомендации от ИИ", type="secondary"):
        st.session_state.show_ai_advice = True

    if st.session_state.get("show_ai_advice"):
        with st.spinner("Гибридный ИИ формирует персональные рекомендации на основе прогноза ML и успеваемости из БД..."):
            llm = get_llm(is_api_mode=is_api_mode)

            prompt_analysis = f"""Ты — ИИ-Методист. Классификатор выдал прогноз: Класс {prediction}.
Успеваемость: {avg_score_text}. Активность: пропуски - {absence}, руки - {raised_hands}, ресурсы - {visited_res}, обсуждения - {discussion}.
Напиши для преподавателя краткий (из 3 пунктов) конкретный план действий по этому студенту."""

            ai_advice = llm.invoke([HumanMessage(content=prompt_analysis)]).content
            st.info(f"**Методические рекомендации ИИ-Консультанта:**\n\n{ai_advice}")
```
