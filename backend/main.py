import threading
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

from backend.llm import LLMIndisponivel, LLMNaoConfigurado, responder
from backend.ml.pipeline import PipelinePokemon
from backend.pokedex import CSV_PATH, SPRITES_PATH, Pokedex, montar_contexto


# Arquivos maiores que isso são recusados antes de abrir a imagem.
TAMANHO_MAXIMO_ARQUIVO_MB = 10

# Imagens maiores são reduzidas antes de passar pelos modelos (o
# detector trabalha em ~800 px de qualquer forma). As caixas voltam
# nas coordenadas da imagem original.
LADO_MAXIMO_PROCESSAMENTO = 1600

TAMANHO_MAXIMO_PERGUNTA = 500


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


# Perguntas ao LLM local sobre um Pokémon (RAG): a ficha vira texto de
# contexto e backend/llm.py gera a resposta. Ver LLM.md. Também é
# síncrona, porque o LLM pode levar vários segundos para responder.
@app.post("/pokemon/{id_pokemon}/perguntar")
def perguntar_sobre_pokemon(
    id_pokemon: int,
    corpo: Pergunta,
):
    ficha = pokedex.obter_ficha(
        id_pokemon
    )

    if ficha is None:
        raise HTTPException(
            status_code=404,
            detail="Pokémon não encontrado.",
        )

    pergunta = corpo.pergunta.strip()

    if not pergunta:
        raise HTTPException(
            status_code=400,
            detail="A pergunta está vazia.",
        )

    try:
        resposta = responder(
            pergunta,
            montar_contexto(ficha),
            ficha,
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

    return {
        "numero": ficha["numero"],
        "nome": ficha["nome"],
        "pergunta": pergunta,
        "resposta": resposta,
    }


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

    return {
        "quantidade": len(
            pokemons_encontrados
        ),
        "largura": imagem.width,
        "altura": imagem.height,
        "pokemons": pokemons_encontrados,
    }
