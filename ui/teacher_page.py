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
