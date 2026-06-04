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