import json
import os
import unicodedata
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    category=UserWarning,
)

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

INVALID_PROXY_VALUES = {
    "http://127.0.0.1:9",
    "https://127.0.0.1:9",
    "127.0.0.1:9",
}
PROXY_ENV_NAMES = [
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
]


def env_to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_to_float(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value.strip())
    except ValueError:
        return default


def sanitize_invalid_proxy_env() -> None:
    for env_name in PROXY_ENV_NAMES:
        current_value = os.getenv(env_name)
        if not current_value:
            continue
        normalized = current_value.strip().lower().rstrip("/")
        if normalized in INVALID_PROXY_VALUES:
            os.environ.pop(env_name, None)


sanitize_invalid_proxy_env()

RAG_ENABLED = env_to_bool(os.getenv("ENABLE_RAG"), default=False)
LANGSMITH_TRACING = env_to_bool(os.getenv("LANGSMITH_TRACING"), default=False)
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "agente2")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash").strip()
GOOGLE_TEMPERATURE = env_to_float(os.getenv("GOOGLE_TEMPERATURE"), default=0.0)

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY ausente. Defina essa variavel no .env para usar o backend Google.")

retriever = None
if RAG_ENABLED:
    from indexar import embeddings

    vectorstore = FAISS.load_local(
        str(BASE_DIR / "vectorstore"),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

with open(BASE_DIR / "clientes.json", "r", encoding="utf-8") as arquivo:
    clientes = json.load(arquivo)

llm = ChatGoogleGenerativeAI(
    model=GOOGLE_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=GOOGLE_TEMPERATURE,
)
SYSTEM_PROMPT = (
    "Voce e um assistente do PetShop Mundo Animalia. "
    "Responda de forma educada e curta, com o minimo de informacoes possivel. "
    "Use tools apenas quando a pergunta exigir consulta de dados de cliente, cadastro, "
    "saldo de pontos, politicas, endereco ou produtos. "
    "Para conversa geral, saudacoes e perguntas ambiguas, nao use tools."
)
SOCIAL_KEYWORDS = {
    "oi",
    "ola",
    "bom dia",
    "boa tarde",
    "boa noite",
    "tudo bem",
    "obrigado",
    "valeu",
}
BUSINESS_KEYWORDS = {
    "saldo",
    "ponto",
    "pontos",
    "cadastro",
    "numero",
    "fidelidade",
    "beneficio",
    "beneficios",
    "cliente",
}
RAG_KEYWORDS = {
    "endereco",
    "politica",
    "politicas",
    "produto",
    "produtos",
    "servico",
    "servicos",
    "preco",
    "precos",
    "desconto",
}


def format_chunks(chunks):
    return "\n\n".join(chunk.page_content for chunk in chunks)


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii").lower().strip()


def route_input_policy(state: MessagesState):
    messages = state.get("messages", [])
    if not messages:
        return "assistant_no_tools"

    last_message = messages[-1]
    content = normalize_text(getattr(last_message, "content", "") or "")
    if not content:
        return "assistant_no_tools"

    if any(keyword in content for keyword in SOCIAL_KEYWORDS):
        return "assistant_no_tools"

    keywords = set(BUSINESS_KEYWORDS)
    if RAG_ENABLED:
        keywords.update(RAG_KEYWORDS)

    if any(keyword in content for keyword in keywords):
        return "assistant_with_tools"

    # Ambiguo: prioriza nao usar tool.
    return "assistant_no_tools"


if RAG_ENABLED:

    @tool
    def busca_rag(query: str) -> str:
        """Busca informacoes do Pet Shop, como endereco, produtos, politicas e fidelidade."""
        chunks = retriever.invoke(query)
        return format_chunks(chunks)


@tool
def buscar_saldo_por_nome(parte_nome: str):
    """Busca saldos de pontos de um cliente por parte do nome (case-insensitive)."""
    parte_nome = parte_nome.lower()
    resultados = [
        {
            "nome": cliente["nome"],
            "numeroCadastro": cliente["numeroCadastro"],
            "saldoPontos": cliente["saldoPontos"],
        }
        for cliente in clientes
        if parte_nome in cliente["nome"].lower()
    ]
    return resultados


ferramentas = [buscar_saldo_por_nome]
if RAG_ENABLED:
    ferramentas.insert(0, busca_rag)

llm_with_tools = llm.bind_tools(ferramentas)


def assistant_no_tools(state: MessagesState):
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    response = llm.invoke(messages)
    return {"messages": [response]}


def assistant_with_tools(state: MessagesState):
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


tools_node = ToolNode(ferramentas)


def route_after_assistant(state: MessagesState):
    messages = state.get("messages", [])
    if not messages:
        return END
    last_message = messages[-1]
    if getattr(last_message, "tool_calls", None):
        return "tools_node"
    return END


workflow = StateGraph(MessagesState)
workflow.add_node("assistant_no_tools", assistant_no_tools)
workflow.add_node("assistant_with_tools", assistant_with_tools)
workflow.add_node("tools_node", tools_node)
workflow.add_conditional_edges(
    START,
    route_input_policy,
    {
        "assistant_no_tools": "assistant_no_tools",
        "assistant_with_tools": "assistant_with_tools",
    },
)
workflow.add_edge("assistant_no_tools", END)
workflow.add_conditional_edges(
    "assistant_with_tools",
    route_after_assistant,
    {"tools_node": "tools_node", END: END},
)
workflow.add_edge("tools_node", "assistant_with_tools")

memory = MemorySaver()
agent = workflow.compile(checkpointer=memory)

config = {
    "configurable": {"thread_id": "sessao1"},
    "run_name": "agente2_cli",
    "tags": ["agente2", "rag" if RAG_ENABLED else "no-rag"],
    "metadata": {
        "rag_enabled": RAG_ENABLED,
        "langsmith_tracing": LANGSMITH_TRACING,
        "langsmith_project": LANGSMITH_PROJECT,
        "llm_backend": "google",
        "llm_model": GOOGLE_MODEL,
    },
}


def extract_last_ai_content(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return str(message.content)
    return ""


def extract_tool_names(messages) -> list[str]:
    tool_names: list[str] = []
    for message in messages:
        message_type = getattr(message, "type", "")
        if message_type in {"tool", "tool_message"}:
            name = getattr(message, "name", None)
            if name and name not in tool_names:
                tool_names.append(name)
    return tool_names


if __name__ == "__main__":
    session_counter = 1
    try:
        while True:
            user_input = input("Voce: ").strip()
            if not user_input:
                continue

            normalized = user_input.lower()
            if normalized in {"sair", "exit", "quit"}:
                break

            if normalized == "/reset":
                session_counter += 1
                new_thread_id = f"sessao{session_counter}"
                config["configurable"]["thread_id"] = new_thread_id
                continue

            try:
                resposta = agent.invoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                )
                messages = resposta.get("messages", [])
                tool_names = extract_tool_names(messages)
                if tool_names:
                    if len(tool_names) == 1:
                        print(f"Consultando tool: {tool_names[0]}")
                    else:
                        print(f"Consultando tools: {', '.join(tool_names)}")
                content = extract_last_ai_content(messages)
                print(f"Assistente: {content}")
            except Exception as exc:
                print(f"Erro: {exc}")
    except KeyboardInterrupt:
        print()
