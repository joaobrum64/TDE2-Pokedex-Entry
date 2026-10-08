# Ponto de integração do LLM local (Ollama), com RAG sobre a ficha do
# Pokémon. O guia completo está em LLM.md, na raiz do projeto.
#
# A rota POST /pokemon/{numero}/perguntar (backend/main.py) já monta o
# contexto e chama responder(). Para ligar o LLM, só esta função precisa
# ser implementada; o frontend já tem o campo de pergunta.


class LLMNaoConfigurado(Exception):
    # A rota devolve 501: a integração ainda não foi implementada.
    pass


class LLMIndisponivel(Exception):
    # A rota devolve 503: a integração existe, mas o LLM não respondeu
    # (por exemplo, o Ollama não está rodando ou o modelo não foi baixado).
    pass


def responder(pergunta, contexto, ficha):
    # pergunta: texto digitado pelo usuário.
    # contexto: texto em português com todos os dados do Pokémon, gerado
    #   por backend.pokedex.montar_contexto(ficha). É o "R" do RAG.
    # ficha: o mesmo dicionário devolvido por GET /pokemon/{numero}.
    #
    # Deve devolver a resposta do LLM como texto, ou levantar
    # LLMIndisponivel com uma mensagem em português se o LLM falhar.
    raise LLMNaoConfigurado(
        "O LLM local ainda não foi configurado."
    )
