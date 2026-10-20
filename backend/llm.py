import json
import re
import socket
import threading
import urllib.error
import urllib.request



# Integração com o LLM local (Ollama), com RAG sobre as fichas do
# dataset. Detalhes e decisões em LLM.md, na raiz do projeto.

OLLAMA = "http://localhost:11434"

MODELO_OLLAMA = "llama3.1:8b"

TEMPO_LIMITE_SEGUNDOS = 180

# Opções enviadas ao Ollama em toda pergunta.
OPCOES_MODELO = {
    # Moderada: dá o tom criativo da Pokédex sem soltar demais o
    # modelo (temperaturas altas inventam mais dados).
    "temperature": 0.4,
    # Janela de contexto: regras + fichas + histórico. O padrão do
    # Ollama é pequeno e cortaria o histórico sem avisar.
    "num_ctx": 8192,
    # Teto de tamanho da resposta, em tokens.
    "num_predict": 450,
}

# Mantém o modelo carregado na memória entre as perguntas.
MANTER_CARREGADO = "30m"

# Quantas mensagens anteriores da conversa (perguntas + respostas)
# são enviadas junto com a pergunta nova.
MENSAGENS_DE_HISTORICO = 12

MARCADOR_EXTERNO = "Registro externo:"

INSTRUCOES = f"""Você é a POKéDEX, a enciclopédia eletrônica de POKéMON criada pelo Professor Carvalho para a região de Kanto, como nos jogos Pokémon Red, Green, Blue e Yellow (Primeira Geração).

ESTILO
- Escreva em português do Brasil, como uma entrada da Pokédex: tom enciclopédico e descritivo, com detalhes sobre aparência, comportamento e habitat, e um toque de mistério. Escreva "POKéMON" como nos jogos.
- Responda primeiro, de forma direta, ao que foi perguntado; depois enriqueça com 2 ou 3 frases descritivas sobre o POKéMON (aparência, comportamento, o que a descrição da ficha conta). Use 1 a 3 parágrafos curtos. Respostas de uma linha só são aceitáveis apenas para recusas.
- Não use markdown: nada de asteriscos, títulos ou negrito.
- Quando perguntarem pelos golpes, liste TODOS os golpes da ficha, cada um em uma linha começando com "- ", com o nível (por exemplo "- Nível 9: Thunder Wave").
- A descrição da Pokédex nas fichas está em inglês: traduza e reescreva com as suas palavras. Cada POKéMON tem a sua própria descrição; nunca reaproveite a descrição de um em outro.

FONTES
- As FICHAS abaixo vêm do banco de dados do projeto e são a sua fonte principal. Números (altura, peso, stats, níveis de golpes), evoluções e vantagens ou fraquezas de tipo devem vir exatamente delas. Para perguntas de batalha, use as linhas "Tabela de tipos da Primeira Geração" e "causam dano dobrado em" das fichas.
- Você pode complementar com o seu próprio conhecimento da Primeira Geração (comportamento, lugares onde aparece, curiosidades, diferenças entre Red, Green, Blue e Yellow), mas SOMENTE em um parágrafo separado, no final, que comece exatamente com "{MARCADOR_EXTERNO}". Nesse parágrafo não invente números nem cite a Pokédex como fonte; fale como relatos de treinadores e pesquisadores. Nunca contradiga uma ficha.
- As fichas NÃO têm: lugares e rotas onde o POKéMON aparece, como ele evolui (nível, pedra ou troca), comportamento fora da descrição, e informações do anime. Qualquer coisa desses temas só pode aparecer dentro do parágrafo "{MARCADOR_EXTERNO}", nunca no texto principal. No texto principal, diga que esse dado não consta nos registros desta Pokédex.
- Para comparações (quem é mais alto, mais pesado, mais rápido, mais forte) e batalhas, a seção "RESUMO CALCULADO" já traz a resposta certa: use a conclusão dela (por exemplo "Maior Velocidade: ..." ou "Batalha: ... leva vantagem de tipo ...") sem refazer as contas, mas explique com as suas palavras, no estilo da Pokédex, em vez de copiar a linha.
- Se perguntarem por "Ataque Especial" ou "Defesa Especial", explique que na Primeira Geração existe um único stat Especial e informe o valor dele, que está na ficha.
- Se a pergunta pede algo que não está nas fichas e você não tem certeza, diga que esse registro não consta na Pokédex em vez de inventar. É melhor não ter o parágrafo "{MARCADOR_EXTERNO}" do que inventar um.
- Nunca copie o texto das fichas nem o formato delas (linhas como "Tipo: ..." ou "Altura: ..."): escreva como uma entrada da Pokédex.

UNIVERSO
- Só existem os 151 POKéMON de Kanto, do BULBASAUR (nº 1) ao MEW (nº 151). Quando a pergunta cita um deles, ele aparece na linha "CITADOS NA PERGUNTA ATUAL".
- Se perguntarem sobre um POKéMON e essa linha disser "nenhum", ele não é de Kanto: responda em uma ou duas frases que ele não consta nos registros de Kanto. Não descreva esse POKéMON, nem mesmo no "{MARCADOR_EXTERNO}": você não sabe nada sobre ele.
- Não existem os tipos Fada, Aço e Sombrio, nem habilidades, e existe um único stat Especial (não há Ataque Especial e Defesa Especial separados). Quando uma ficha mostrar golpes diferentes em outra versão (como o Yellow), você pode comentar a diferença.
- Se a pergunta não tiver relação com POKéMON, recuse com educação em uma frase, sem sair do personagem.

CONVERSA
- O usuário enviou uma imagem e o sistema detectou os POKéMON listados em "POKÉMON DETECTADOS NA IMAGEM DO USUÁRIO". "Ele", "esse" ou "este POKéMON" se referem a eles.
- Quando a pergunta for sobre outro POKéMON, responda sobre ele e termine comparando com o(s) POKéMON detectado(s) no que foi perguntado (altura, peso, stats, tipos).

EXEMPLOS DE TOM E FORMATO
Os exemplos abaixo são sobre o ONIX e servem só para mostrar o tom e o formato. Os dados deles não fazem parte desta conversa, e nenhuma frase deles deve ser repetida.

Pergunta: Fale sobre ele.
Resposta: ONIX, o POKéMON Serpente de Pedra. Do tipo Pedra e Terra, este colosso de 8,8 m e 210 kg é uma corrente de rochas que range a cada movimento. Conforme envelhece, as pedras de seu corpo endurecem até lembrar diamantes negros, e sua Defesa de 160 é a maior entre todos os POKéMON de rocha de Kanto.
{MARCADOR_EXTERNO} treinadores contam que BROCK, o líder do Ginásio de Pewter, confia em um ONIX para testar quem chega à cidade.
Pergunta: Como ele evolui?
Resposta: Os registros desta Pokédex indicam que o ONIX não possui evolução conhecida em Kanto. Sua imponência, porém, dispensa transformações: poucos POKéMON rivalizam com seu tamanho."""


class LLMNaoConfigurado(Exception):
    # Mantida por compatibilidade: a rota devolve 501 se ela aparecer.
    pass


class LLMIndisponivel(Exception):
    # A rota devolve 503: o LLM não respondeu (Ollama fechado, modelo
    # não baixado, tempo esgotado). A mensagem é mostrada ao usuário.
    pass


def chamar_ollama(caminho, corpo, tempo_limite):
    requisicao = urllib.request.Request(
        f"{OLLAMA}{caminho}",
        data=json.dumps(corpo).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(requisicao, timeout=tempo_limite) as resposta:
        return json.loads(resposta.read().decode("utf-8"))


def traduzir_erro(erro):
    # Mensagens diferentes para cada problema, para quem estiver
    # usando saber o que fazer.
    if isinstance(erro, urllib.error.HTTPError) and erro.code == 404:
        return (
            f"O modelo {MODELO_OLLAMA} não foi baixado. Rode "
            f"\"ollama pull {MODELO_OLLAMA}\" e tente de novo."
        )

    motivo = getattr(erro, "reason", erro)

    if isinstance(motivo, (TimeoutError, socket.timeout)):
        return (
            f"O LLM demorou mais de {TEMPO_LIMITE_SEGUNDOS} segundos para "
            "responder. Tente uma pergunta mais curta ou um modelo menor."
        )

    if isinstance(motivo, ConnectionRefusedError):
        return (
            "O Ollama não está aberto. Abra o Ollama e tente de novo."
        )

    return "Não foi possível obter uma resposta do Ollama."


def limpar_markdown(texto):
    # O site mostra texto puro. Mesmo pedindo sem markdown, os modelos
    # às vezes usam negrito, títulos ou listas com "*".
    texto = texto.replace("**", "").replace("__", "")
    texto = re.sub(r"^\s*#{1,6}\s*", "", texto, flags=re.MULTILINE)
    texto = re.sub(r"^(\s*)[*•]\s+", r"\1- ", texto, flags=re.MULTILINE)

    return texto.strip()


def montar_mensagens(pergunta, contexto, historico, lembrete=""):
    mensagens = [
        {
            "role": "system",
            "content": f"{INSTRUCOES}\n\nFICHAS\n{contexto}",
        }
    ]

    for mensagem in historico[-MENSAGENS_DE_HISTORICO:]:
        mensagens.append(
            {
                "role": "user" if mensagem["autor"] == "usuario" else "assistant",
                "content": mensagem["texto"],
            }
        )

    # O lembrete vai só para o modelo; o histórico guarda a pergunta
    # como o usuário escreveu.
    conteudo = f"{pergunta}\n\n{lembrete}" if lembrete else pergunta

    mensagens.append({"role": "user", "content": conteudo})

    return mensagens


def responder_em_partes(
    pergunta,
    contexto,
    historico,
    opcoes_extras=None,
    lembrete="",
):
    # Pede a resposta ao Ollama em streaming e devolve um iterador com os
    # pedaços de texto, na ordem em que o modelo os gera.
    # pergunta: texto digitado pelo usuário.
    # contexto: fichas do RAG, de Pokedex.montar_contexto_conversa().
    # historico: mensagens anteriores da conversa, em ordem
    #   ([{"autor": "usuario" | "ia", "texto": ...}]).
    # opcoes_extras: opções do Ollama que sobrepõem OPCOES_MODELO
    #   (usado por testar_llm.py para forçar CPU, por exemplo).
    # lembrete: texto colado na pergunta só para o modelo, de
    #   Pokedex.lembrete_para_pergunta().
    #
    # A conexão é aberta aqui mesmo, antes de devolver o iterador, para
    # que "Ollama fechado" ou "modelo não baixado" virem LLMIndisponivel
    # antes de a rota começar a responder.
    corpo = {
        "model": MODELO_OLLAMA,
        "stream": True,
        # Modelos com "modo de raciocínio" (como o qwen3) pensariam em
        # voz alta antes de responder, o que só deixa tudo mais lento.
        "think": False,
        "keep_alive": MANTER_CARREGADO,
        "options": {**OPCOES_MODELO, **(opcoes_extras or {})},
        "messages": montar_mensagens(pergunta, contexto, historico, lembrete),
    }

    requisicao = urllib.request.Request(
        f"{OLLAMA}/api/chat",
        data=json.dumps(corpo).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    try:
        resposta = urllib.request.urlopen(
            requisicao,
            timeout=TEMPO_LIMITE_SEGUNDOS,
        )

    except (urllib.error.URLError, TimeoutError, socket.timeout) as erro:
        raise LLMIndisponivel(traduzir_erro(erro)) from erro

    def partes():
        with resposta:
            try:
                # O Ollama manda uma linha JSON por pedaço de texto.
                for linha in resposta:
                    if not linha.strip():
                        continue

                    dados = json.loads(linha.decode("utf-8"))

                    if "error" in dados:
                        raise LLMIndisponivel(
                            f"O Ollama respondeu com erro: {dados['error']}"
                        )

                    pedaco = dados.get("message", {}).get("content", "")

                    if pedaco:
                        yield pedaco

                    if dados.get("done"):
                        return

            except (urllib.error.URLError, TimeoutError, socket.timeout) as erro:
                raise LLMIndisponivel(traduzir_erro(erro)) from erro

            except ValueError as erro:
                raise LLMIndisponivel(
                    "O Ollama devolveu uma resposta em formato inesperado."
                ) from erro

    return partes()


def responder(pergunta, contexto, historico, opcoes_extras=None, lembrete=""):
    # Mesma coisa que responder_em_partes, mas devolve a resposta
    # completa, já sem markdown (usado por testar_llm.py).
    texto = limpar_markdown(
        "".join(
            responder_em_partes(
                pergunta, contexto, historico, opcoes_extras, lembrete
            )
        )
    )

    if not texto:
        raise LLMIndisponivel("O LLM não devolveu nenhuma resposta.")

    return texto


def aquecer():
    # Carrega o modelo na memória em segundo plano quando a API sobe,
    # para a primeira pergunta não esperar o carregamento. Falhas são
    # ignoradas: o Ollama pode nem estar aberto ainda.
    def carregar():
        try:
            chamar_ollama(
                "/api/generate",
                {"model": MODELO_OLLAMA, "keep_alive": MANTER_CARREGADO},
                TEMPO_LIMITE_SEGUNDOS,
            )
        except Exception:
            pass

    threading.Thread(target=carregar, daemon=True).start()
