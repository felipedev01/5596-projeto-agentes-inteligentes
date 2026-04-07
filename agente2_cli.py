from langchain_core.messages import HumanMessage

from agente2_support import create_default_config, last_ai_content, set_session_thread

EXIT_COMMANDS = {"sair", "exit", "quit"}
RESET_COMMAND = "/reset"


def main(agent) -> None:
    config = create_default_config()
    session = 1

    try:
        while True:
            user_input = input("Voce: ").strip()
            if not user_input:
                continue

            normalized = user_input.lower()
            if normalized in EXIT_COMMANDS:
                break

            if normalized == RESET_COMMAND:
                session += 1
                set_session_thread(config, session)
                continue

            try:
                result = agent.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
                print(f"Assistente: {last_ai_content(result['messages'])}")
            except Exception as exc:
                print(f"Erro: {exc}")
    except KeyboardInterrupt:
        print()
