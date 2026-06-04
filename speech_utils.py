import io
import os
import tempfile
import uuid
import warnings
import wave
import requests
import torch
import urllib3
import streamlit as st
from transformers import pipeline

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="transformers")
urllib3.disable_warnings()


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


SALUTESPEECH_AUTH_KEY = (
        os.getenv("SALUTESPEECH_AUTH_KEY")
        or read_api_key_from_file("salutespeech_key.txt", "salutespeech.key", "api_audio_key.txt")
)


@st.cache_resource(show_spinner=False)
def load_speech_models():
    print("Инициализация локальных аудиомоделей (Whisper + Silero)...")

    whisper_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device="cpu")
    device = torch.device("cpu")
    local_file = "model.pt"

    if not os.path.isfile(local_file):
        torch.hub.download_url_to_file("https://models.silero.ai/models/tts/ru/v4_ru.pt", local_file)

    silero_model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
    silero_model.to(device)

    return whisper_pipe, silero_model


def get_salute_token():
    if not SALUTESPEECH_AUTH_KEY:
        return None

    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    headers = {
        "Authorization": f"Basic {SALUTESPEECH_AUTH_KEY}",
        "RqUID": str(uuid.uuid4()),
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {"scope": "SALUTE_SPEECH_PERS"}
    try:
        response = requests.post(url, headers=headers, data=data, verify=False)
        response.raise_for_status()
        return response.json().get("access_token")
    except Exception:
        return None


def check_salutespeech_connection():
    if not SALUTESPEECH_AUTH_KEY:
        return False, "SaluteSpeech: ключ не найден. Создайте файл salutespeech_key.txt в корне проекта."

    token = get_salute_token()
    if token:
        return True, "SaluteSpeech: Токен получен"

    return False, "SaluteSpeech: Ошибка авторизации или неверный ключ"


def process_audio_stt(audio_bytes, is_api_mode=False):
    if not audio_bytes:
        return ""

    if is_api_mode:
        token = get_salute_token()
        if token:
            try:
                with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
                    sample_rate = wav_file.getframerate()
                    sampwidth = wav_file.getsampwidth()
                    raw_frames = wav_file.readframes(wav_file.getnframes())

                content_type = f"audio/x-pcm;bit={sampwidth * 8};rate={sample_rate}"
                url = "https://smartspeech.sber.ru/rest/v1/speech:recognize"
                headers = {"Authorization": f"Bearer {token}", "Content-Type": content_type}

                response = requests.post(url, headers=headers, data=raw_frames, verify=False)
                response.raise_for_status()
                return response.json().get("result", [""])[0]
            except Exception as e:
                print(f"Ошибка SaluteSpeech STT: {e}")
                return f"Ошибка Сбера: {str(e)}"
        return "Ошибка Сбера: API-ключ SaluteSpeech не найден или токен не получен"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_filename = temp_audio.name

    try:
        whisper_pipe, _ = load_speech_models()
        result = whisper_pipe(temp_filename, generate_kwargs={"language": "russian"})
        return result["text"].strip()
    finally:
        os.remove(temp_filename)


def generate_audio_tts(text, speaker="xenia", is_api_mode=False):
    clean_text = text.replace("**", "").replace("*", "").replace("#", "")
    if len(clean_text) > 800:
        clean_text = clean_text[:800] + "... Текст длинный, читайте на экране."

    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_out.close()

    if is_api_mode:
        token = get_salute_token()
        if token:
            url = "https://smartspeech.sber.ru/rest/v1/text:synthesize?format=wav16&voice=Nec_24000"
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/text"}
            try:
                response = requests.post(url, headers=headers, data=clean_text.encode("utf-8"), verify=False)
                response.raise_for_status()
                with open(temp_out.name, "wb") as f:
                    f.write(response.content)
                return temp_out.name
            except Exception as e:
                print(f"Ошибка SaluteSpeech TTS: {e}")

    try:
        _, silero_model = load_speech_models()
        silero_model.save_wav(text=clean_text, speaker=speaker, sample_rate=48000, audio_path=temp_out.name)
        return temp_out.name
    except Exception as e:
        print(f"Ошибка Silero: {e}")
        return None