import json
import threading
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

from backend.conversas import Conversas
from backend.llm import (
    LLMIndisponivel,
    LLMNaoConfigurado,
    aquecer,
    limpar_markdown,
    responder_em_partes,
)
from backend.ml.pipeline import PipelinePokemon
from backend.pokedex import CSV_PATH, SPRITES_PATH, Pokedex


# Arquivos maiores que isso são recusados antes de abrir a imagem.
TAMANHO_MAXIMO_ARQUIVO_MB = 10

# Imagens maiores são reduzidas antes de passar pelos modelos (o
# detector trabalha em ~800 px de qualquer forma). As caixas voltam
# nas coordenadas da imagem original.
LADO_MAXIMO_PROCESSAMENTO = 1600

TAMANHO_MAXIMO_PERGUNTA = 500

MAXIMO_POKEMONS_POR_CONVERSA = 10


app = FastAPI(
    title="Pokédex Gen 1 API",
    description=(
        "Backend do projeto TDE2 de reconhecimento "
        "de Pokémon da Primeira Geração."
    ),
    version="0.6.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


pokedex = Pokedex()

# Conversas com o LLM, só na memória (somem quando o backend para).
conversas = Conversas()


print(
    "Carregando pipeline de Machine Learning..."
)

pipeline = PipelinePokemon()

print(
    f"Pipeline carregado em: "
    f"{pipeline.dispositivo}"
)

# Uma identificação por vez nos modelos: a GPU não ganha nada
# processando duas imagens em paralelo, e assim o uso de memória
# fica previsível. As outras rotas continuam respondendo.
trava_pipeline = threading.Lock()

# Carrega o modelo do Ollama em segundo plano, se ele estiver aberto.
aquecer()


classes_sem_dados = [
    classe
    for classe in pipeline.classificador.classes
    if pokedex.obter_ficha(classe) is None
]

if classes_sem_dados:
    raise RuntimeError(
        "Classes do modelo sem correspondência em "
        f"{CSV_PATH.name}: {classes_sem_dados}"
    )


app.mount(
    "/sprites",
    StaticFiles(
        directory=SPRITES_PATH
    ),
    name="sprites",
)


class Pergunta(BaseModel):
    pergunta: str = Field(
        min_length=1,
        max_length=TAMANHO_MAXIMO_PERGUNTA,
    )


class NovaConversa(BaseModel):
    numeros: list[int] = Field(
        min_length=1,
        max_length=MAXIMO_POKEMONS_POR_CONVERSA,
    )


def abrir_imagem(conteudo):
    try:
        imagem = Image.open(
            BytesIO(conteudo)
        )

        imagem.load()

    except (UnidentifiedImageError, Image.DecompressionBombError, OSError):
        raise HTTPException(
            status_code=400,
            detail=(
                "O arquivo enviado não é "
                "uma imagem válida."
            ),
        )

    # Fotos de celular guardam a rotação em um campo EXIF em vez de
    # girar os pixels; sem isso o modelo receberia a foto deitada.
    return ImageOps.exif_transpose(imagem)


def reduzir_imagem(imagem):
    # Devolve a imagem a processar e o fator para voltar as caixas
    # para o tamanho original.
    maior_lado = max(imagem.size)

    if maior_lado <= LADO_MAXIMO_PROCESSAMENTO:
        return imagem, 1.0

    escala = LADO_MAXIMO_PROCESSAMENTO / maior_lado

    reduzida = imagem.convert("RGB").resize(
        (
            max(1, round(imagem.width * escala)),
            max(1, round(imagem.height * escala)),
        ),
        Image.Resampling.LANCZOS,
    )

    return reduzida, 1 / escala


def referencia_previsao(previsao):
    return {
        **pokedex.referencia(previsao["pokemon"]),
        "confianca": previsao["confianca"],
    }


@app.get("/")
def root():
    return {
        "message": "API da Pokédex funcionando!"
    }


@app.get("/pokemon/{id_pokemon}")
def get_pokemon(
    id_pokemon: int,
):
    ficha = pokedex.obter_ficha(
        id_pokemon
    )

    if ficha is None:
        raise HTTPException(
            status_code=404,
            detail="Pokémon não encontrado.",
        )

    return ficha


def linha_json(dados):
    return json.dumps(dados, ensure_ascii=False) + "\n"


def obter_conversa(conversa_id):
    conversa = conversas.obter(conversa_id)

    if conversa is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Conversa não encontrada. O backend pode ter sido "
                "reiniciado: envie a imagem de novo."
            ),
        )

    return conversa


# Conversa para a busca manual (na identificação por imagem ela é
# criada automaticamente e vem em "conversa_id").
@app.post("/conversas")
def criar_conversa(
    corpo: NovaConversa,
):
    for numero in corpo.numeros:
        if pokedex.obter_ficha(numero) is None:
            raise HTTPException(
                status_code=404,
                detail=f"Pokémon {numero} não encontrado.",
            )

    return {
        "conversa_id": conversas.criar(corpo.numeros),
    }


@app.get("/conversas/{conversa_id}")
def ver_conversa(
    conversa_id: str,
):
    return obter_conversa(conversa_id)


# Perguntas ao LLM local (RAG): as fichas dos Pokémon detectados, dos
# citados na pergunta e das evoluções viram o contexto, e
# backend/llm.py gera a resposta levando em conta o histórico da
# conversa. Ver LLM.md. A resposta chega em streaming (o texto aparece
# aos poucos no site).
@app.post("/conversas/{conversa_id}/perguntar")
def perguntar(
    conversa_id: str,
    corpo: Pergunta,
):
    conversa = obter_conversa(conversa_id)

    pergunta = corpo.pergunta.strip()

    if not pergunta:
        raise HTTPException(
            status_code=400,
            detail="A pergunta está vazia.",
        )

    # Nomes citados nas últimas perguntas também entram no contexto,
    # para "e a altura dele?" continuar falando do mesmo Pokémon.
    perguntas_recentes = [
        mensagem["texto"]
        for mensagem in conversa["mensagens"]
        if mensagem["autor"] == "usuario"
    ][-2:]

    contexto = pokedex.montar_contexto_conversa(
        conversa["numeros"],
        [pergunta, *reversed(perguntas_recentes)],
    )

    try:
        partes = responder_em_partes(
            pergunta,
            contexto,
            conversa["mensagens"],
            lembrete=pokedex.lembrete_para_pergunta(
                conversa["numeros"],
                pergunta,
            ),
        )

    except LLMNaoConfigurado as erro:
        raise HTTPException(
            status_code=501,
            detail=str(erro),
        )

    except LLMIndisponivel as erro:
        raise HTTPException(
            status_code=503,
            detail=str(erro),
        )

    # A resposta vai sendo enviada enquanto o modelo escreve: uma linha
    # JSON por pedaço ({"tipo": "parte", "texto": ...}) e, no fim, a
    # resposta completa já limpa ({"tipo": "fim", "resposta": ...}) ou
    # um erro ({"tipo": "erro", "detail": ...}).
    def transmitir():
        acumulado = ""

        try:
            for pedaco in partes:
                acumulado += pedaco

                yield linha_json(
                    {"tipo": "parte", "texto": pedaco.replace("**", "")}
                )

        except LLMIndisponivel as erro:
            yield linha_json({"tipo": "erro", "detail": str(erro)})
            return

        resposta = limpar_markdown(acumulado)

        if not resposta:
            yield linha_json(
                {"tipo": "erro", "detail": "O LLM não devolveu nenhuma resposta."}
            )
            return

        conversas.registrar(conversa_id, pergunta, resposta)

        yield linha_json({"tipo": "fim", "resposta": resposta})

    return StreamingResponse(
        transmitir(),
        media_type="application/x-ndjson",
    )


# Rota síncrona (def, não async def): o FastAPI a executa em outra
# thread, então os modelos não travam o servidor enquanto rodam.
@app.post("/pokemon/identificar")
def identificar_pokemon(
    arquivo: UploadFile = File(...),
):
    try:
        limite_bytes = TAMANHO_MAXIMO_ARQUIVO_MB * 1024 * 1024

        conteudo = arquivo.file.read(
            limite_bytes + 1
        )

    finally:
        arquivo.file.close()

    if not conteudo:
        raise HTTPException(
            status_code=400,
            detail="O arquivo enviado está vazio.",
        )

    if len(conteudo) > limite_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                "O arquivo enviado é maior que "
                f"{TAMANHO_MAXIMO_ARQUIVO_MB} MB."
            ),
        )

    # A imagem só existe na memória durante esta requisição; nada
    # é gravado em disco.
    imagem = abrir_imagem(conteudo)

    imagem_processada, fator = reduzir_imagem(imagem)

    with trava_pipeline:
        resultados_ml = pipeline.processar(
            imagem_processada
        )

    pokemons_encontrados = []

    for resultado in resultados_ml:
        pokemons_encontrados.append(
            {
                "pokemon": pokedex.obter_ficha(
                    resultado["pokemon"]
                ),
                "confianca": resultado[
                    "confianca"
                ],
                "confianca_detector": resultado[
                    "confianca_detector"
                ],
                "box": [
                    round(valor * fator)
                    for valor in resultado["box"]
                ],
                "fallback": resultado[
                    "fallback"
                ],
                "top_5": [
                    referencia_previsao(previsao)
                    for previsao in resultado["top_5"]
                ],
            }
        )

    # Uma conversa com o LLM por imagem, já sabendo quais Pokémon
    # foram detectados.
    conversa_id = None

    if pokemons_encontrados:
        conversa_id = conversas.criar(
            [
                encontrado["pokemon"]["numero"]
                for encontrado in pokemons_encontrados
            ][:MAXIMO_POKEMONS_POR_CONVERSA]
        )

    return {
        "quantidade": len(
            pokemons_encontrados
        ),
        "conversa_id": conversa_id,
        "largura": imagem.width,
        "altura": imagem.height,
        "pokemons": pokemons_encontrados,
    }
