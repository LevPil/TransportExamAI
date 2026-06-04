import os
from langchain_openai import ChatOpenAI
from langchain_gigachat import GigaChat

def read_api_key_from_file(*filenames: str) -> str:
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    for filename in filenames:
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                key = f.read().strip()
                if key:
                    return key
    return ""

GIGACHAT_AUTH_KEY = (
    os.getenv("GIGACHAT_AUTH_KEY")
    or read_api_key_from_file("gigachat_key.txt")
)

def check_gigachat_connection():
    if not GIGACHAT_AUTH_KEY:
        return False, "GigaChat: ключ не найден. Создайте файл gigachat_key.txt в корне проекта."
    try:
        llm = GigaChat(credentials=GIGACHAT_AUTH_KEY, verify_ssl_certs=False, scope="GIGACHAT_API_PERS")
        llm.invoke("Привет")
        return True, "GigaChat: Успешное подключение"
    except Exception as e:
        return False, f"GigaChat: Ошибка ({str(e)})"

def get_llm(is_api_mode: bool = False, temperature: float = 0.3, max_tokens: int = 2048):
    if is_api_mode:
        if not GIGACHAT_AUTH_KEY:
            raise ValueError("GigaChat API-ключ не найден. Создайте файл gigachat_key.txt в корне проекта.")

        return GigaChat(
            credentials=GIGACHAT_AUTH_KEY,
            verify_ssl_certs=False,
            scope="GIGACHAT_API_PERS",
            temperature=temperature,
            max_tokens=max_tokens,
        )

    return ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        model="local-model",
        temperature=temperature,
        max_tokens=max_tokens,
    )