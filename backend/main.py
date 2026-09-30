import csv
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from backend.ml.pipeline import PipelinePokemon


app = FastAPI(
    title="Pokédex Gen 1 API",
    description=(
        "Backend do projeto TDE2 de reconhecimento "
        "de Pokémon da Primeira Geração."
    ),
    version="0.5.0",
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


ROOT_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = (
    ROOT_DIR
    / "datasets"
    / "pokemon_gen1.csv"
)

SPRITES_PATH = (
    ROOT_DIR
    / "datasets"
    / "sprites"
)


def load_pokemons():
    fun_pokemons = {}

    with open(
        CSV_PATH,
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as arquivo:
        reader = csv.DictReader(
            arquivo
        )

        for pokemon in reader:
            id_pokemon = int(
                pokemon["pokedex_number"]
            )

            fun_pokemons[
                id_pokemon
            ] = pokemon

    return fun_pokemons


def normalizar_nome(nome):
    # As classes do modelo vêm dos nomes das pastas de treino
    # (ex.: "Farfetchd", "MrMime"), enquanto o CSV usa a grafia
    # oficial ("Farfetch'd", "Mr. Mime"). Os símbolos ♀ e ♂ são
    # mantidos para não confundir os dois Nidoran.
    return (
        nome
        .lower()
        .replace(" ", "")
        .replace(".", "")
        .replace("'", "")
    )


pokemons = load_pokemons()


pokemons_por_nome = {
    normalizar_nome(pokemon["name"]): pokemon
    for pokemon in pokemons.values()
}


print(
    "Carregando pipeline de Machine Learning..."
)

pipeline = PipelinePokemon()

print(
    f"Pipeline carregado em: "
    f"{pipeline.dispositivo}"
)


classes_sem_dados = [
    classe
    for classe in pipeline.classificador.classes
    if normalizar_nome(classe) not in pokemons_por_nome
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


@app.get("/")
def root():
    return {
        "message": "API da Pokédex funcionando!"
    }


@app.get("/pokemon/{id_pokemon}")
def get_pokemon(
    id_pokemon: int,
):
    pokemon = pokemons.get(
        id_pokemon
    )

    if pokemon is None:
        raise HTTPException(
            status_code=404,
            detail="Pokémon não encontrado.",
        )

    return pokemon


@app.post("/pokemon/identificar")
async def identificar_pokemon(
    arquivo: UploadFile = File(...),
):
    try:
        conteudo = await arquivo.read()

        if not conteudo:
            raise HTTPException(
                status_code=400,
                detail="O arquivo enviado está vazio.",
            )

        imagem = Image.open(
            BytesIO(conteudo)
        )

        imagem.load()

        resultados_ml = pipeline.processar(
            imagem
        )

        pokemons_encontrados = []

        for resultado in resultados_ml:
            nome_pokemon = resultado[
                "pokemon"
            ]

            pokemon = pokemons_por_nome[
                normalizar_nome(nome_pokemon)
            ]

            pokemons_encontrados.append(
                {
                    "pokemon": pokemon,
                    "confianca": resultado[
                        "confianca"
                    ],
                    "confianca_detector": resultado[
                        "confianca_detector"
                    ],
                    "box": list(
                        resultado["box"]
                    ),
                    "fallback": resultado[
                        "fallback"
                    ],
                    "top_5": resultado[
                        "top_5"
                    ],
                }
            )

        return {
            "quantidade": len(
                pokemons_encontrados
            ),
            "pokemons": pokemons_encontrados,
        }

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail=(
                "O arquivo enviado não é "
                "uma imagem válida."
            ),
        )

    finally:
        await arquivo.close()