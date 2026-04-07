from langchain_core.tools import tool

from agente2_support import load_clientes

clientes = load_clientes()


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


TOOLS = [buscar_saldo_por_nome]
