import json
import os
import warnings
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    category=UserWarning,
)

BASE_DIR = Path(__file__).resolve().parent
INVALID_PROXY_VALUES = {"http://127.0.0.1:9", "https://127.0.0.1:9", "127.0.0.1:9"}
PROXY_ENV_NAMES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "GIT_HTTP_PROXY",
    "GIT_HTTPS_PROXY",
    "git_http_proxy",
    "git_https_proxy",
)

load_dotenv(BASE_DIR / ".env")


def env_float(name: str, default: float = 0.0) -> float:
    try:
        return float((os.getenv(name) or "").strip() or default)
    except ValueError:
        return default


def sanitize_invalid_proxy_env() -> None:
    for name in PROXY_ENV_NAMES:
        value = os.getenv(name)
        if value and value.strip().lower().rstrip("/") in INVALID_PROXY_VALUES:
            os.environ.pop(name, None)


def load_clientes() -> list[dict[str, Any]]:
    with open(BASE_DIR / "clientes.json", encoding="utf-8") as file:
        return json.load(file)


def create_default_config() -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": "sessao1"}}


def set_session_thread(config: dict[str, Any], session: int) -> None:
    config["configurable"]["thread_id"] = f"sessao{session}"


def extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [extract_text_content(item) for item in content]
        return "\n".join(part for part in parts if part).strip()
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text.strip()
        nested_content = content.get("content")
        if nested_content is not None:
            return extract_text_content(nested_content)
    return ""


def last_ai_content(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = extract_text_content(message.content)
            if text:
                return text
            return str(message.content)
    return ""


sanitize_invalid_proxy_env()
