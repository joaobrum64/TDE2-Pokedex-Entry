import csv
import json
import random
import shutil
from pathlib import Path

from PIL import Image

from backend.ml.analise_dataset import normalizar_nome
from backend.ml.preparar_dados import carregar_amostras


ROOT_DIR = Path(__file__).resolve().parent.parent.parent

DATASETS_PATH = ROOT_DIR / "datasets"

CSV_PATH = DATASETS_PATH / "pokemon_gen1.csv"
COREL_PATH = DATASETS_PATH / "naopokemons" / "Corel-5k"
SPRITES_TESTE_PATH = DATASETS_PATH / "deteccao" / "test"

OUTPUT_PATH = DATASETS_PATH / "avaliacao_pipeline"
MOSAICOS_PATH = OUTPUT_PATH / "mosaicos"
GABARITO_PATH = OUTPUT_PATH / "gabarito.json"

SEED = 42

QUANTIDADE_NENHUM = 500
QUANTIDADE_MOSAICOS_2 = 150
QUANTIDADE_MOSAICOS_3 = 150

ALTURA_MOSAICO = 384


def caminho_relativo(caminho):
    return caminho.relative_to(DATASETS_PATH).as_posix()


def gerar_unicos(amostras_teste, classes):
    # Imagens reais com um Pokémon, tiradas do conjunto de teste
    # do classificador (nunca usadas no treino).
    return [
        {
            "imagem": caminho_relativo(caminho),
            "categoria": "unico",
            "esperado": [classes[classe]],
        }
        for caminho, classe in amostras_teste
    ]


def gerar_nenhum(sorteio):
    # As versões "_aug" são cópias alteradas das originais.
    originais = sorted(
        arquivo
        for arquivo in COREL_PATH.iterdir()
        if arquivo.is_file() and "_aug" not in arquivo.name
    )

    escolhidas = sorteio.sample(
        originais,
        QUANTIDADE_NENHUM,
    )

    return [
        {
            "imagem": caminho_relativo(caminho),
            "categoria": "nenhum",
            "esperado": [],
        }
        for caminho in escolhidas
    ]


def montar_mosaico(caminhos):
    partes = []

    for caminho in caminhos:
        with Image.open(caminho) as imagem:
            imagem = imagem.convert("RGB")

        largura = max(
            1,
            round(
                imagem.width * ALTURA_MOSAICO / imagem.height
            ),
        )

        partes.append(
            imagem.resize(
                (largura, ALTURA_MOSAICO),
                Image.Resampling.LANCZOS,
            )
        )

    mosaico = Image.new(
        "RGB",
        (
            sum(parte.width for parte in partes),
            ALTURA_MOSAICO,
        ),
    )

    posicao = 0

    for parte in partes:
        mosaico.paste(parte, (posicao, 0))
        posicao += parte.width

    return mosaico


def gerar_mosaicos(amostras_teste, classes, sorteio):
    # Imagens reais de Pokémon diferentes lado a lado, para medir
    # o caso de vários Pokémon sem depender de sprites.
    disponiveis = list(amostras_teste)

    sorteio.shuffle(disponiveis)

    itens = []

    for tamanho, quantidade in (
        (2, QUANTIDADE_MOSAICOS_2),
        (3, QUANTIDADE_MOSAICOS_3),
    ):
        for indice in range(1, quantidade + 1):
            escolhidas = []

            while len(escolhidas) < tamanho:
                caminho, classe = disponiveis.pop()

                if classe not in [c for _, c in escolhidas]:
                    escolhidas.append((caminho, classe))

            caminho_mosaico = (
                MOSAICOS_PATH
                / f"mosaico_{tamanho}_{indice:04d}.jpg"
            )

            montar_mosaico(
                [caminho for caminho, _ in escolhidas]
            ).save(
                caminho_mosaico,
                quality=92,
            )

            itens.append(
                {
                    "imagem": caminho_relativo(caminho_mosaico),
                    "categoria": f"mosaico_{tamanho}",
                    "esperado": [
                        classes[classe]
                        for _, classe in escolhidas
                    ],
                }
            )

    return itens


def carregar_classes_por_numero(classes):
    classes_normalizadas = {
        normalizar_nome(classe): classe
        for classe in classes
    }

    classes_por_numero = {}

    with open(
        CSV_PATH,
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as arquivo:
        for pokemon in csv.DictReader(arquivo):
            classes_por_numero[
                int(pokemon["pokedex_number"])
            ] = classes_normalizadas[
                normalizar_nome(pokemon["name"])
            ]

    return classes_por_numero


def gerar_sprites(classes):
    # Cenas sintéticas de teste do detector: sprites colados sobre
    # fundos do Corel, algumas sem nenhum Pokémon.
    classes_por_numero = carregar_classes_por_numero(classes)

    itens = []

    for caminho_label in sorted(
        (SPRITES_TESTE_PATH / "labels").glob("*.json")
    ):
        with open(
            caminho_label,
            mode="r",
            encoding="utf-8",
        ) as arquivo:
            dados = json.load(arquivo)

        itens.append(
            {
                "imagem": caminho_relativo(
                    SPRITES_TESTE_PATH / "images" / dados["image"]
                ),
                "categoria": "sprites",
                "esperado": [
                    classes_por_numero[objeto["pokedex_id"]]
                    for objeto in dados["objects"]
                ],
            }
        )

    return itens


def main():
    print("=" * 60)
    print("GERADOR DO CONJUNTO DE AVALIACAO DO PIPELINE")
    print("=" * 60)

    sorteio = random.Random(SEED)

    classes, _, _, _, amostras_teste = carregar_amostras()

    if OUTPUT_PATH.exists():
        shutil.rmtree(OUTPUT_PATH)

    MOSAICOS_PATH.mkdir(parents=True)

    itens = (
        gerar_unicos(amostras_teste, classes)
        + gerar_nenhum(sorteio)
        + gerar_mosaicos(amostras_teste, classes, sorteio)
        + gerar_sprites(classes)
    )

    with open(
        GABARITO_PATH,
        mode="w",
        encoding="utf-8",
    ) as arquivo:
        json.dump(
            itens,
            arquivo,
            ensure_ascii=False,
            indent=0,
        )

    print()

    for categoria in sorted(
        {item["categoria"] for item in itens}
    ):
        quantidade = sum(
            1
            for item in itens
            if item["categoria"] == categoria
        )

        print(f"{categoria}: {quantidade} imagens")

    print()
    print(f"Total: {len(itens)} imagens")
    print(f"Gabarito: {GABARITO_PATH}")


if __name__ == "__main__":
    main()
