import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

from modelo import ModeloPokemon


ROOT_DIR = Path(__file__).resolve().parent.parent.parent

DETECTOR_PATH = (
    ROOT_DIR
    / "modelos"
    / "pokemon_object_detector.pth"
)


SCORE_MINIMO_DETECTOR = 0.50

CONFIANCA_MINIMA_CLASSIFICADOR = 50.0

CONFIANCA_MINIMA_FALLBACK = 70.0

IOA_THRESHOLD = 0.80


def carregar_detector(dispositivo):
    checkpoint = torch.load(
        DETECTOR_PATH,
        map_location=dispositivo,
        weights_only=False,
    )

    modelo = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
    )

    entradas = (
        modelo
        .roi_heads
        .box_predictor
        .cls_score
        .in_features
    )

    modelo.roi_heads.box_predictor = FastRCNNPredictor(
        entradas,
        2,
    )

    modelo.load_state_dict(
        checkpoint["model_state_dict"]
    )

    modelo = modelo.to(
        dispositivo
    )

    modelo.eval()

    return modelo


def calcular_area(box):
    x1, y1, x2, y2 = box

    largura = max(
        0,
        x2 - x1,
    )

    altura = max(
        0,
        y2 - y1,
    )

    return largura * altura


def calcular_intersecao(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)

    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    largura = max(
        0,
        x2 - x1,
    )

    altura = max(
        0,
        y2 - y1,
    )

    return largura * altura


def calcular_ioa(box_menor, box_maior):
    area_menor = calcular_area(
        box_menor
    )

    if area_menor <= 0:
        return 0.0

    intersecao = calcular_intersecao(
        box_menor,
        box_maior,
    )

    return (
        intersecao
        / area_menor
    )


def remover_boxes_redundantes(deteccoes):
    deteccoes = sorted(
        deteccoes,
        key=lambda item: item["score"],
        reverse=True,
    )

    resultado = []

    for atual in deteccoes:
        box_atual = atual["box"]

        area_atual = calcular_area(
            box_atual
        )

        redundante = False

        for mantida in resultado:
            box_mantida = mantida["box"]

            area_mantida = calcular_area(
                box_mantida
            )

            if area_atual <= area_mantida:
                box_menor = box_atual
                box_maior = box_mantida
            else:
                box_menor = box_mantida
                box_maior = box_atual

            ioa = calcular_ioa(
                box_menor,
                box_maior,
            )

            if ioa >= IOA_THRESHOLD:
                redundante = True
                break

        if not redundante:
            resultado.append(
                atual
            )

    return resultado


def detectar_regioes(
    detector,
    imagem,
    dispositivo,
):
    tensor = F.to_tensor(
        imagem
    )

    tensor = tensor.to(
        dispositivo
    )

    with torch.no_grad():
        resultado = detector(
            [tensor]
        )[0]

    boxes = (
        resultado["boxes"]
        .detach()
        .cpu()
    )

    scores = (
        resultado["scores"]
        .detach()
        .cpu()
    )

    labels = (
        resultado["labels"]
        .detach()
        .cpu()
    )

    deteccoes = []

    for box, score, label in zip(
        boxes,
        scores,
        labels,
    ):
        if label.item() != 1:
            continue

        if score.item() < SCORE_MINIMO_DETECTOR:
            continue

        x1 = int(box[0].item())
        y1 = int(box[1].item())
        x2 = int(box[2].item())
        y2 = int(box[3].item())

        deteccoes.append(
            {
                "box": (
                    x1,
                    y1,
                    x2,
                    y2,
                ),
                "score": (
                    score.item()
                    * 100
                ),
            }
        )

    return remover_boxes_redundantes(
        deteccoes
    )


def classificar_regioes(
    imagem,
    deteccoes,
    classificador,
):
    resultados = []

    for deteccao in deteccoes:
        box = deteccao["box"]

        recorte = imagem.crop(
            box
        )

        previsoes = classificador.prever(
            recorte
        )

        melhor = previsoes[0]

        resultados.append(
            {
                "box": box,
                "confianca_detector": deteccao["score"],
                "pokemon": melhor["pokemon"],
                "confianca": melhor["confianca"],
                "top_5": previsoes,
            }
        )

    return resultados


def filtrar_classificacoes(resultados):
    validos = []

    for resultado in resultados:
        if (
            resultado["confianca"]
            >= CONFIANCA_MINIMA_CLASSIFICADOR
        ):
            validos.append(
                resultado
            )

    return validos


def tentar_fallback(
    imagem,
    classificador,
):
    previsoes = classificador.prever(
        imagem
    )

    melhor = previsoes[0]

    if (
        melhor["confianca"]
        < CONFIANCA_MINIMA_FALLBACK
    ):
        return None

    largura, altura = imagem.size

    return {
        "box": (
            0,
            0,
            largura,
            altura,
        ),
        "confianca_detector": None,
        "pokemon": melhor["pokemon"],
        "confianca": melhor["confianca"],
        "top_5": previsoes,
        "fallback": True,
    }


def processar_imagem(
    caminho_imagem,
):
    dispositivo = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("PIPELINE ML COMPLETO")
    print("=" * 60)

    print()
    print(
        f"Dispositivo: {dispositivo}"
    )

    print()
    print(
        "Carregando detector de objetos..."
    )

    detector = carregar_detector(
        dispositivo
    )

    print(
        "Carregando classificador..."
    )

    classificador = ModeloPokemon()

    print(
        "Modelos carregados."
    )

    with Image.open(
        caminho_imagem
    ) as imagem_original:
        imagem = imagem_original.convert(
            "RGB"
        )

    deteccoes = detectar_regioes(
        detector,
        imagem,
        dispositivo,
    )

    print()
    print(
        f"Regioes brutas detectadas: "
        f"{len(deteccoes)}"
    )

    resultados = classificar_regioes(
        imagem,
        deteccoes,
        classificador,
    )

    resultados_validos = (
        filtrar_classificacoes(
            resultados
        )
    )

    if len(resultados_validos) == 0:
        print()
        print(
            "Nenhuma regiao passou pelo "
            "threshold do classificador."
        )

        print(
            "Tentando classificacao da "
            "imagem completa..."
        )

        fallback = tentar_fallback(
            imagem,
            classificador,
        )

        if fallback is not None:
            resultados_validos = [
                fallback
            ]

    print()
    print("=" * 60)
    print("RESULTADO FINAL")
    print("=" * 60)

    if len(resultados_validos) == 0:
        print()
        print(
            "Nenhum Pokemon Gen 1 identificado."
        )

        return

    print()
    print(
        f"Pokemon validos encontrados: "
        f"{len(resultados_validos)}"
    )

    for indice, resultado in enumerate(
        resultados_validos,
        start=1,
    ):
        print()
        print(
            f"Pokemon {indice}"
        )

        print(
            f"Previsao: "
            f"{resultado['pokemon']}"
        )

        print(
            f"Confianca: "
            f"{resultado['confianca']:.2f}%"
        )

        print(
            f"Box: "
            f"{resultado['box']}"
        )

        if (
            resultado.get(
                "fallback",
                False,
            )
        ):
            print(
                "Origem: imagem completa "
                "(fallback)"
            )

        else:
            print(
                f"Detector: "
                f"{resultado['confianca_detector']:.2f}%"
            )

        print(
            "Top 5:"
        )

        for posicao, previsao in enumerate(
            resultado["top_5"],
            start=1,
        ):
            print(
                f"  {posicao}. "
                f"{previsao['pokemon']}: "
                f"{previsao['confianca']:.2f}%"
            )


def main():
    if len(sys.argv) < 2:
        print(
            "Uso:"
        )

        print(
            "python "
            "backend/ml/"
            "teste_pipeline_completo.py "
            "\"caminho_imagem\""
        )

        return

    caminho_imagem = Path(
        sys.argv[1]
    )

    if not caminho_imagem.exists():
        print(
            f"Imagem nao encontrada: "
            f"{caminho_imagem}"
        )

        return

    processar_imagem(
        caminho_imagem
    )


if __name__ == "__main__":
    main()