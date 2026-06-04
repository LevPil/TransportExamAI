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
