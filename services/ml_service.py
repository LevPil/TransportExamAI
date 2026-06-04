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
