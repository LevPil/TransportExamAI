import streamlit as st
from llm_provider import check_gigachat_connection
from speech_utils import check_salutespeech_connection

def render_sidebar():
    st.sidebar.title("Настройки Системы")
    role = st.sidebar.radio("Выберите роль пользователя:", ["Студент", "Преподаватель"])
    st.sidebar.divider()
    st.sidebar.subheader("Режим работы ИИ")
    engine_mode = st.sidebar.radio(
        "Архитектура нейросетей:",
        ["Локальный (LM Studio + Whisper)", "Облачный API (GigaChat + Sber)"],
    )
    is_api_mode = engine_mode == "Облачный API (GigaChat + Sber)"
    if is_api_mode:
        if st.sidebar.button("Проверить подключение к API"):
            with st.sidebar.status("Проверка токенов доступа...", expanded=True) as status:
                giga_ok, giga_msg = check_gigachat_connection()
                st.write(giga_msg)
                sber_ok, sber_msg = check_salutespeech_connection()
                st.write(sber_msg)
                if giga_ok and sber_ok:
                    status.update(label="API системы работают стабильно!", state="complete")
                else:
                    status.update(label="Обнаружены проблемы с API!", state="error")
    st.sidebar.divider()
    return role, is_api_mode