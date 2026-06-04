import streamlit as st

def init_session_state():
    defaults = {
        "messages": [],
        "test_completed": False,
        "current_test_id": None,
        "auto_trigger": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_teacher_draft_state():
    keys_to_delete = [
        "teacher_topics",
        "teacher_topics_raw_context",
        "selected_teacher_topic",
        "generation_prompt",
        "generation_context",
        "draft_test_title",
        "draft_test_content",
        "draft_doc_id",
        "draft_validation",
        "proposed_test_content",
        "llm_test_validation_report",
        "llm_test_validation_structural",
        "draft_test_editor",
        "draft_test_editor_version",
        "proposed_test_editor_version",
        "generation_prompt_editor",
    ]
    for key in keys_to_delete:
        st.session_state.pop(key, None)
    for key in list(st.session_state.keys()):
        if key.startswith("draft_test_editor_") or key.startswith("proposed_test_editor_"):
            del st.session_state[key]

def reset_student_test_state(active_test_id):
    st.session_state.current_test_id = active_test_id
    st.session_state.test_completed = False
    st.session_state.messages = []
    st.session_state.student_score = ""
    st.session_state.answers_report = ""
    st.session_state.detailed_results = []
    for key in list(st.session_state.keys()):
        if key.startswith("q_"):
            del st.session_state[key]