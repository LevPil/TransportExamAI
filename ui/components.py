import os
import streamlit as st
from speech_utils import generate_audio_tts

def render_tts_button(text: str, key: str, is_api_mode: bool, speaker: str = "xenia"):
    if st.button("🔊", key=key, help="Прослушать вопрос"):
        with st.spinner("..."):
            audio_path = generate_audio_tts(text, speaker=speaker, is_api_mode=is_api_mode)
            if audio_path:
                with open(audio_path, "rb") as f:
                    st.audio(f.read(), format="audio/wav")
                os.remove(audio_path)

def render_validation_messages(validation):
    if validation.get("ok"):
        st.success(f"Формат теста корректен. Найдено вопросов: {validation.get('questions_count')}.")
    else:
        st.warning("Python-валидатор нашёл проблемы в формате теста:")
        for err in validation.get("errors", []):
            st.write(f"- {err}")
    for warn in validation.get("warnings", []):
        st.info(f"Предупреждение: {warn}")