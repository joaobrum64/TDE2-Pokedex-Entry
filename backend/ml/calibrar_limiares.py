import json
import random
import sys
from collections import Counter
from pathlib import Path
from PIL import Image
from backend.ml.gerar_avaliacao_pipeline import (
    COREL_PATH,
    carregar_classes_por_numero,
    montar_mosaico,
)
from backend.ml.pipeline import (
    CONFIANCA_MINIMA_CLASSIFICADOR,
    CONFIANCA_MINIMA_FALLBACK,
    DETECTOR_PATH,
    SCORE_MINIMO_DETECTOR,
    PipelinePokemon,
)
from backend.ml.preparar_dados import carregar_amostras


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATASETS_PATH = ROOT_DIR / "datasets"
GABARITO_AVALIACAO_PATH = DATASETS_PATH / "avaliacao_pipeline" / "gabarito.json"
SPRITES_VALIDACAO_PATH = DATASETS_PATH / "deteccao" / "val"
OUTPUT_PATH = ROOT_DIR / "saidas"
SEED = 42
QUANTIDADE_NENHUM = 500
QUANTIDADE_MOSAICOS = 100
SCORE_MINIMO_BUSCA = 30.0
GRADE_DETECTOR = [30, 40, 50, 60, 70, 80, 90]
LIMIARES_ATUAIS = (
    round(SCORE_MINIMO_DETECTOR * 100),
    round(CONFIANCA_MINIMA_CLASSIFICADOR),
    round(CONFIANCA_MINIMA_FALLBACK),
)
GRADE_CLASSIFICADOR = list(range(10, 75, 5))
GRADE_FALLBACK = list(range(20, 95, 5))
FALSO_POSITIVO_MAXIMO = 3.0
QUANTIDADE_MELHORES = 10


def carregar_corel_avaliacao():
    with open(
        GABARITO_AVALIACAO_PATH,
        mode="r",
        encoding="utf-8",
    ) as arquivo:
        return [
            DATASETS_PATH / item["imagem"]
            for item in json.load(arquivo)
            if item["categoria"] == "nenhum"
        ]


def sortear_corel_calibracao(sorteio=None):

    if sorteio is None:
        sorteio = random.Random(SEED)

    usadas = set(carregar_corel_avaliacao())

    originais = sorted(
        arquivo
        for arquivo in COREL_PATH.iterdir()
        if arquivo.is_file()
        and "_aug" not in arquivo.name
        and arquivo not in usadas
    )

    return sorteio.sample(originais, QUANTIDADE_NENHUM)


def montar_conjunto(classes):
    sorteio = random.Random(SEED)

    _, _, _, validacao, _ = carregar_amostras()

    itens = [
        {
            "categoria": "unico",
            "imagem": lambda caminho=caminho: Image.open(caminho),
            "esperado": [classes[classe]],
        }
        for caminho, classe in validacao
    ]

    for caminho in sortear_corel_calibracao(sorteio):
        itens.append(
            {
                "categoria": "nenhum",
                "imagem": lambda caminho=caminho: Image.open(caminho),
                "esperado": [],
            }
        )

    disponiveis = list(validacao)
    sorteio.shuffle(disponiveis)

    for tamanho in (2, 3):
        for _ in range(QUANTIDADE_MOSAICOS):
            escolhidas = []

            while len(escolhidas) < tamanho:
                caminho, classe = disponiveis.pop()

                if classe not in [c for _, c in escolhidas]:
                    escolhidas.append((caminho, classe))

            itens.append(
                {
                    "categoria": "mosaico",
                    "imagem": lambda escolhidas=escolhidas: montar_mosaico(
                        [caminho for caminho, _ in escolhidas]
                    ),
                    "esperado": [classes[c] for _, c in escolhidas],
                }
            )

    classes_por_numero = carregar_classes_por_numero(classes)

    for caminho_label in sorted(
        (SPRITES_VALIDACAO_PATH / "labels").glob("*.json")
    ):
        with open(caminho_label, mode="r", encoding="utf-8") as arquivo:
            dados = json.load(arquivo)

        caminho = SPRITES_VALIDACAO_PATH / "images" / dados["image"]

        itens.append(
            {
                "categoria": "sprites",
                "imagem": lambda caminho=caminho: Image.open(caminho),
                "esperado": [
                    classes_por_numero[objeto["pokedex_id"]]
                    for objeto in dados["objects"]
                ],
            }
        )

    return itens


def coletar_saidas(pipeline, itens):

    for indice, item in enumerate(itens, start=1):
        imagem = item["imagem"]().convert("RGB")

        regioes = pipeline.detectar_regioes(
            imagem,
            score_minimo=SCORE_MINIMO_BUSCA / 100,
            remover_redundantes=False,
        )

        for regiao in regioes:
            melhor = pipeline.classificador.prever(
                imagem.crop(regiao["box"])
            )[0]

            regiao["pokemon"] = melhor["pokemon"]
            regiao["confianca"] = melhor["confianca"]

        melhor_inteira = pipeline.classificador.prever(imagem)[0]

        item["regioes"] = regioes
        item["inteira"] = (
            melhor_inteira["pokemon"],
            melhor_inteira["confianca"],
        )

        del item["imagem"]

        if indice % 300 == 0 or indice == len(itens):
            print(f"{indice}/{len(itens)}")


def simular(pipeline, itens, limiar_detector):

    return [
        pipeline.remover_boxes_redundantes(
            [
                regiao
                for regiao in item["regioes"]
                if regiao["score"] >= limiar_detector
            ]
        )
        for item in itens
    ]


def avaliar(
    pipeline,
    itens,
    regioes_por_item,
    limiar_classificador,
    limiar_fallback,
):
    exatas = Counter()
    total = Counter()

    for item, regioes in zip(itens, regioes_por_item):
        obtido = [
            regiao["pokemon"]
            for regiao in pipeline.unir_mesma_especie(
                [
                    regiao
                    for regiao in regioes
                    if regiao["confianca"] >= limiar_classificador
                ]
            )
        ]

        if not obtido and item["inteira"][1] >= limiar_fallback:
            obtido = [item["inteira"][0]]

        total[item["categoria"]] += 1

        if Counter(obtido) == Counter(item["esperado"]):
            exatas[item["categoria"]] += 1

    return {
        categoria: exatas[categoria] / total[categoria] * 100
        for categoria in total
    }


def main():
    rotulo = sys.argv[1] if len(sys.argv) > 1 else "atual"

    caminho_classificador = (
        ROOT_DIR / "modelos" / sys.argv[2]
        if len(sys.argv) > 2 and sys.argv[2] != "-"
        else None
    )

    caminho_detector = (
        ROOT_DIR / "modelos" / sys.argv[3]
        if len(sys.argv) > 3
        else DETECTOR_PATH
    )

    print("=" * 78)
    print("CALIBRACAO DOS LIMIARES DO PIPELINE")
    print("=" * 78)

    pipeline = PipelinePokemon(caminho_classificador, caminho_detector)

    itens = montar_conjunto(pipeline.classificador.classes)

    print()
    print(
        "Conjunto de calibracao: "
        + ", ".join(
            f"{categoria} {quantidade}"
            for categoria, quantidade in Counter(
                item["categoria"] for item in itens
            ).items()
        )
    )
    print()

    coletar_saidas(pipeline, itens)

    print()
    print("Testando combinacoes...")

    resultados = []

    for limiar_detector in GRADE_DETECTOR:
        regioes_por_item = simular(pipeline, itens, limiar_detector)

        for limiar_classificador in GRADE_CLASSIFICADOR:
            for limiar_fallback in GRADE_FALLBACK:
                acertos = avaliar(
                    pipeline,
                    itens,
                    regioes_por_item,
                    limiar_classificador,
                    limiar_fallback,
                )

                resultados.append(
                    {
                        "detector": limiar_detector,
                        "classificador": limiar_classificador,
                        "fallback": limiar_fallback,
                        **acertos,
                        "media": sum(acertos.values()) / len(acertos),
                    }
                )

    validos = [
        resultado
        for resultado in resultados
        if 100 - resultado["nenhum"] <= FALSO_POSITIVO_MAXIMO
    ]

    validos.sort(key=lambda resultado: resultado["media"], reverse=True)

    atual = next(
        resultado
        for resultado in resultados
        if (
            resultado["detector"],
            resultado["classificador"],
            resultado["fallback"],
        ) == LIMIARES_ATUAIS
    )

    def linha(resultado):
        return (
            f"{resultado['detector']:>6}"
            f"{resultado['classificador']:>8}"
            f"{resultado['fallback']:>9}"
            f"{resultado['unico']:>9.2f}"
            f"{resultado['nenhum']:>9.2f}"
            f"{resultado['mosaico']:>9.2f}"
            f"{resultado['sprites']:>9.2f}"
            f"{resultado['media']:>9.2f}"
        )

    cabecalho = (
        f"{'det.':>6}{'classif.':>8}{'fallback':>9}"
        f"{'unico':>9}{'nenhum':>9}{'mosaico':>9}{'sprites':>9}{'media':>9}"
    )

    print()
    print("Imagens exatas (%) por categoria")
    print()
    print("Limiares atuais (%d / %d / %d):" % LIMIARES_ATUAIS)
    print(cabecalho)
    print(linha(atual))

    print()
    print(
        f"Melhores com falso positivo <= {FALSO_POSITIVO_MAXIMO:.0f}% "
        "(nenhum >= "
        f"{100 - FALSO_POSITIVO_MAXIMO:.0f}%):"
    )
    print(cabecalho)

    for resultado in validos[:QUANTIDADE_MELHORES]:
        print(linha(resultado))

    OUTPUT_PATH.mkdir(exist_ok=True)

    caminho_saida = OUTPUT_PATH / f"calibracao_{rotulo}.json"

    with open(caminho_saida, mode="w", encoding="utf-8") as arquivo:
        json.dump(
            {"atual": atual, "melhores": validos[:50]},
            arquivo,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(f"Resultados salvos em: {caminho_saida}")


if __name__ == "__main__":
    main()
