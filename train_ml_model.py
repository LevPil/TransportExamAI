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