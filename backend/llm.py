import json
import urllib.error
import urllib.request


OLLAMA_URL = "http://localhost:11434/api/chat"

MODELO_OLLAMA = "llama3.2:3b"

TEMPO_LIMITE_SEGUNDOS = 120

INSTRUCOES = (
    "Você é a Pokédex da Primeira Geração. Responda em português do Brasil, "
    "de forma curta e direta, usando APENAS os dados do Pokémon fornecidos "
    "abaixo. Se a resposta não estiver nesses dados, diga que a Pokédex não "
    "tem essa informação. Não invente dados de outras gerações."
)
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
    corpo = {
        "model": MODELO_OLLAMA,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    f"{INSTRUCOES}\n\n"
                    f"Dados do Pokémon:\n{contexto}"
                ),
            },
            {
                "role": "user",
                "content": pergunta,
            },
        ],
    }

    requisicao = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(corpo).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(
            requisicao,
            timeout=TEMPO_LIMITE_SEGUNDOS,
        ) as resposta:
            dados = json.loads(
                resposta.read().decode("utf-8")
            )

        texto = dados["message"]["content"].strip()

    # URLError: Ollama fechado ou modelo não baixado (HTTPError é uma
    # subclasse). ValueError/KeyError: resposta em formato inesperado.
    except (
        urllib.error.URLError,
        TimeoutError,
        ValueError,
        KeyError,
    ) as erro:
        raise LLMIndisponivel(
            "Não foi possível falar com o Ollama. Verifique se ele está "
            f"aberto e se o modelo {MODELO_OLLAMA} foi baixado."
        ) from erro

    if not texto:
        raise LLMIndisponivel(
            "O LLM não devolveu nenhuma resposta."
        )

    return texto
