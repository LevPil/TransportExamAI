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