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