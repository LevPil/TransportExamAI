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