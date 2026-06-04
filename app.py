import streamlit as st
from config import APP_ICON, APP_LAYOUT, APP_TITLE
from database import SessionLocal
from state import init_session_state
from ui.sidebar import render_sidebar
from ui.student_page import render_student_page
from ui.teacher_page import render_teacher_page

st.set_page_config(page_title=APP_TITLE, layout=APP_LAYOUT, page_icon=APP_ICON)

@st.cache_resource
def get_db_session():
    return SessionLocal()

def main():
    init_session_state()
    db = get_db_session()
    role, is_api_mode = render_sidebar()

    if role == "Преподаватель":
        render_teacher_page(db=db, is_api_mode=is_api_mode)
    else:
        render_student_page(db=db, is_api_mode=is_api_mode)

if __name__ == "__main__":
    main()
