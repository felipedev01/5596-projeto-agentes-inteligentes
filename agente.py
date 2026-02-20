import os
from dotenv import load_dotenv
from rich import print
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from indexar import embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough

from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.prebuilt import ToolNode, tools_condition

import json

load_dotenv()

vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k":4})

llm = ChatGroq(model=os.getenv("GROQ_MODEL"), temperature=0)

def format_chunks(chunks):
    return "\n\n".join(chunk.page_content for chunk in chunks)

@tool
def busca_rag(query: str) -> str:
    """
    Busca de informações do Pet Shop, como:
    descrição, endereço, produtos, políticas, fidelidade, e dúvidas frequentes.
    """
    chunks = retriever.invoke(query)
    return format_chunks(chunks)

class Estado(TypedDict):
    messages: Annotated[list, add_messages]

with open("clientes.json", 'r', encoding='utf-8') as arquivo:
    clientes = json.load(arquivo)

@tool
def buscar_saldo_por_nome(parte_nome):
    """
    busca saldos de pontos de um cliente
    por parte do nome (case-insensitive).

    :param parte_nome: parte do nome a ser buscada
    :return: lista de dicionários com nome, numeroCadastro e saldoPontos
    """
    parte_nome = parte_nome.lower()

    resultados = [
        {
            "nome": cliente["nome"],
            "numeroCadastro": cliente["numeroCadastro"],
            "saldoPontos": cliente["saldoPontos"]
        }
        for cliente in clientes
        if parte_nome in cliente["nome"].lower()
    ]

    return resultados

ferramentas = [busca_rag, buscar_saldo_por_nome]

llm_com_ferramentas = llm.bind_tools(ferramentas)

def chamar_llm_com_ferramentas(estado: Estado) -> Estado:
    return { "messages": [ llm_com_ferramentas.invoke(estado["messages"]) ]}


builder = StateGraph(Estado)
builder.add_node("no_llm", chamar_llm_com_ferramentas)
builder.add_node("tools", ToolNode(ferramentas))

builder.add_edge(START, "no_llm")
builder.add_conditional_edges("no_llm", tools_condition)
builder.add_edge("tools", "no_llm")
builder.add_edge("no_llm", END)

graph=builder.compile()

# Gera uma imagem PNG usando o serviço online do Mermaid (não requer pygraphviz)
img_data = graph.get_graph().draw_mermaid_png(
    draw_method=MermaidDrawMethod.API
)

with open("graph.png", "wb") as f:
    f.write(img_data)

SYSTEM_PROMPT ="Você é um assistente do PetShop Mundo Animalia. Responda de forma educada e curta, com o mínimo de informações possível."
estado_global = Estado({ "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
        ] })

def chamar_grafo(texto):
    global estado_global
    estado_global["messages"].append(HumanMessage(content=texto))
    estado_global = graph.invoke(estado_global)
    return estado_global["messages"][-1].content

if __name__ == "__main__":
    print("ok")
    print(chamar_grafo("meu nome é Silva"))
    #print(chamar_grafo("qual endereço da loja?"))
    #print(chamar_grafo("qual o meu nome?"))
    print(chamar_grafo("quantos pontos eu tenho?"))
    print(chamar_grafo("Gabriela"))
