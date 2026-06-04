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