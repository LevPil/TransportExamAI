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
