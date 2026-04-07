import os

from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agente2_support import env_float
from agente2_tools import TOOLS

SYSTEM_PROMPT = "Responda qualquer pergunta do usuario"

GOOGLE_API_KEY = (os.getenv("GOOGLE_API_KEY") or "").strip()
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY ausente. Defina essa variavel no .env para usar o backend Google.")

llm = ChatGoogleGenerativeAI(
    model=(os.getenv("GOOGLE_MODEL") or "gemini-2.5-flash").strip(),
    google_api_key=GOOGLE_API_KEY,
    temperature=env_float("GOOGLE_TEMPERATURE"),
).bind_tools(TOOLS)


def assistant(state: MessagesState):
    return {"messages": [llm.invoke([SystemMessage(content=SYSTEM_PROMPT), *state["messages"]])]}


workflow = StateGraph(MessagesState)
workflow.add_node("assistant", assistant)
workflow.add_node("tools", ToolNode(TOOLS))
workflow.add_edge(START, "assistant")
workflow.add_conditional_edges("assistant", tools_condition, {"tools": "tools", END: END})
workflow.add_edge("tools", "assistant")

agent = workflow.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    from agente2_cli import main

    main(agent)
