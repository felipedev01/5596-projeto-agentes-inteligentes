import json
import os
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    category=UserWarning,
)

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

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
SYSTEM_PROMPT = (

    "Responda qualquer pergunta do usuário"
)


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


sanitize_invalid_proxy_env()
GOOGLE_API_KEY = (os.getenv("GOOGLE_API_KEY") or "").strip()
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY ausente. Defina essa variavel no .env para usar o backend Google.")

with open(BASE_DIR / "clientes.json", encoding="utf-8") as file:
    clientes = json.load(file)


@tool
def buscar_saldo_por_nome(parte_nome: str):
    """Busca cliente por parte do nome e retorna cadastro e saldo de pontos."""
    nome = parte_nome.lower()
    return [
        {
            "nome": cliente["nome"],
            "numeroCadastro": cliente["numeroCadastro"],
            "saldoPontos": cliente["saldoPontos"],
        }
        for cliente in clientes
        if nome in cliente["nome"].lower()
    ]


llm = ChatGoogleGenerativeAI(
    model=(os.getenv("GOOGLE_MODEL") or "gemini-2.5-flash").strip(),
    google_api_key=GOOGLE_API_KEY,
    temperature=env_float("GOOGLE_TEMPERATURE"),
).bind_tools([buscar_saldo_por_nome])


def assistant(state: MessagesState):
    return {"messages": [llm.invoke([SystemMessage(content=SYSTEM_PROMPT), *state["messages"]])]}


workflow = StateGraph(MessagesState)
workflow.add_node("assistant", assistant)
workflow.add_node("tools", ToolNode([buscar_saldo_por_nome]))
workflow.add_edge(START, "assistant")
workflow.add_conditional_edges("assistant", tools_condition, {"tools": "tools", END: END})
workflow.add_edge("tools", "assistant")

agent = workflow.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "sessao1"}}


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


def last_ai_content(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = extract_text_content(message.content)
            if text:
                return text
            return str(message.content)
    return ""


if __name__ == "__main__":
    session = 1
    try:
        while True:
            user_input = input("Voce: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"sair", "exit", "quit"}:
                break
            if user_input.lower() == "/reset":
                session += 1
                config["configurable"]["thread_id"] = f"sessao{session}"
                continue
            try:
                result = agent.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
                print(f"Assistente: {last_ai_content(result['messages'])}")
            except Exception as exc:
                print(f"Erro: {exc}")
    except KeyboardInterrupt:
        print()
