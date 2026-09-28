import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.models import resnet18
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


ROOT_DIR = Path(__file__).resolve().parent.parent.parent


DETECTOR_PATH = (
    ROOT_DIR
    / "modelos"
    / "pokemon_object_detector.pth"
)


CLASSIFICADOR_PATH = (
    ROOT_DIR
    / "modelos"
    / "pokemon_resnet18_v2.pth"
)


OUTPUT_PATH = (
    ROOT_DIR
    / "datasets"
    / "pipeline_multiplos"
)


SCORE_THRESHOLD_DETECTOR = 0.50

IOA_THRESHOLD = 0.80


TRANSFORM_CLASSIFICADOR = transforms.Compose([
    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ],
    ),
])


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

    quantidade_entradas = (
        modelo
        .roi_heads
        .box_predictor
        .cls_score
        .in_features
    )

    modelo.roi_heads.box_predictor = FastRCNNPredictor(
        quantidade_entradas,
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


def carregar_classificador(
    dispositivo,
):
    checkpoint = torch.load(
        CLASSIFICADOR_PATH,
        map_location=dispositivo,
        weights_only=False,
    )

    classes = checkpoint[
        "classes"
    ]

    quantidade_classes = checkpoint[
        "num_classes"
    ]

    modelo = resnet18(
        weights=None
    )

    quantidade_entradas = (
        modelo.fc.in_features
    )

    modelo.fc = nn.Linear(
        quantidade_entradas,
        quantidade_classes,
    )

    modelo.load_state_dict(
        checkpoint["model_state_dict"]
    )

    modelo = modelo.to(
        dispositivo
    )

    modelo.eval()

    return modelo, classes


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


def calcular_intersecao(
    box_a,
    box_b,
):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(
        ax1,
        bx1,
    )

    y1 = max(
        ay1,
        by1,
    )

    x2 = min(
        ax2,
        bx2,
    )

    y2 = min(
        ay2,
        by2,
    )

    largura = max(
        0,
        x2 - x1,
    )

    altura = max(
        0,
        y2 - y1,
    )

    return largura * altura


def calcular_ioa(
    box_menor,
    box_maior,
):
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


def remover_boxes_redundantes(
    deteccoes,
):
    deteccoes_ordenadas = sorted(
        deteccoes,
        key=lambda item: item["score"],
        reverse=True,
    )

    resultado = []

    for deteccao_atual in deteccoes_ordenadas:
        box_atual = deteccao_atual[
            "box"
        ]

        area_atual = calcular_area(
            box_atual
        )

        redundante = False

        for deteccao_existente in resultado:
            box_existente = (
                deteccao_existente[
                    "box"
                ]
            )

            area_existente = calcular_area(
                box_existente
            )

            if area_atual <= area_existente:
                box_menor = box_atual
                box_maior = box_existente

            else:
                box_menor = box_existente
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
                deteccao_atual
            )

    return resultado


def detectar_pokemons(
    modelo,
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
        resultado = modelo(
            [tensor]
        )[0]

    deteccoes = []

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

    for box, score, label in zip(
        boxes,
        scores,
        labels,
    ):
        if label.item() != 1:
            continue

        score_valor = (
            score.item()
        )

        if (
            score_valor
            < SCORE_THRESHOLD_DETECTOR
        ):
            continue

        x1 = int(
            box[0].item()
        )

        y1 = int(
            box[1].item()
        )

        x2 = int(
            box[2].item()
        )

        y2 = int(
            box[3].item()
        )

        deteccoes.append({
            "box": (
                x1,
                y1,
                x2,
                y2,
            ),

            "score": (
                score_valor * 100
            ),
        })

    return remover_boxes_redundantes(
        deteccoes
    )


def classificar_recorte(
    modelo,
    classes,
    recorte,
    dispositivo,
):
    recorte = recorte.convert(
        "RGB"
    )

    tensor = TRANSFORM_CLASSIFICADOR(
        recorte
    )

    tensor = tensor.unsqueeze(
        0
    )

    tensor = tensor.to(
        dispositivo
    )

    with torch.no_grad():
        saidas = modelo(
            tensor
        )

        probabilidades = torch.softmax(
            saidas,
            dim=1,
        )[0]

    top_probs, top_indices = torch.topk(
        probabilidades,
        k=3,
    )

    resultados = []

    for prob, indice in zip(
        top_probs,
        top_indices,
    ):
        resultados.append({
            "pokemon": classes[
                indice.item()
            ],

            "confianca": (
                prob.item()
                * 100
            ),
        })

    return resultados


def processar_imagem(
    caminho_imagem,
    detector,
    classificador,
    classes,
    dispositivo,
):
    with Image.open(
        caminho_imagem
    ) as imagem_original:
        imagem = imagem_original.convert(
            "RGB"
        )

    deteccoes = detectar_pokemons(
        detector,
        imagem,
        dispositivo,
    )

    resultados = []

    for indice, deteccao in enumerate(
        deteccoes,
        start=1,
    ):
        box = deteccao[
            "box"
        ]

        recorte = imagem.crop(
            box
        )

        classificacoes = classificar_recorte(
            classificador,
            classes,
            recorte,
            dispositivo,
        )

        resultados.append({
            "indice": indice,
            "box": box,

            "confianca_detector": (
                deteccao["score"]
            ),

            "classificacoes": (
                classificacoes
            ),
        })

    return imagem, resultados


def gerar_preview(
    imagem,
    resultados,
):
    copia = imagem.copy()

    desenho = ImageDraw.Draw(
        copia
    )

    for resultado in resultados:
        x1, y1, x2, y2 = resultado[
            "box"
        ]

        melhor = resultado[
            "classificacoes"
        ][0]

        nome = melhor[
            "pokemon"
        ]

        confianca = melhor[
            "confianca"
        ]

        texto = (
            f"{nome} "
            f"{confianca:.1f}%"
        )

        desenho.rectangle(
            (
                x1,
                y1,
                x2,
                y2,
            ),
            outline="red",
            width=4,
        )

        desenho.text(
            (
                x1 + 5,
                y1 + 5,
            ),
            texto,
            fill="red",
        )

    return copia


def main():
    if len(sys.argv) < 2:
        print("Uso:")

        print(
            "python "
            "backend/ml/"
            "teste_pipeline_multiplos.py "
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

    dispositivo = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("PIPELINE DE MULTIPLOS POKEMON")
    print("=" * 60)

    print()
    print(
        f"Dispositivo: "
        f"{dispositivo}"
    )

    print()
    print(
        "Carregando detector..."
    )

    detector = carregar_detector(
        dispositivo
    )

    print(
        "Carregando classificador..."
    )

    (
        classificador,
        classes,
    ) = carregar_classificador(
        dispositivo
    )

    print(
        "Modelos carregados."
    )

    (
        imagem,
        resultados,
    ) = processar_imagem(
        caminho_imagem,
        detector,
        classificador,
        classes,
        dispositivo,
    )

    print()
    print("=" * 60)
    print("RESULTADO")
    print("=" * 60)

    print()
    print(
        f"Regioes detectadas: "
        f"{len(resultados)}"
    )

    if len(resultados) == 0:
        print()
        print(
            "Nenhum Pokemon detectado."
        )

    for resultado in resultados:
        print()
        print(
            f"Pokemon "
            f"{resultado['indice']}"
        )

        print(
            f"Box: "
            f"{resultado['box']}"
        )

        print(
            f"Detector: "
            f"{resultado['confianca_detector']:.2f}%"
        )

        print(
            "Classificacao:"
        )

        for posicao, classificacao in enumerate(
            resultado[
                "classificacoes"
            ],
            start=1,
        ):
            print(
                f"  {posicao}. "
                f"{classificacao['pokemon']}: "
                f"{classificacao['confianca']:.2f}%"
            )

    OUTPUT_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    preview = gerar_preview(
        imagem,
        resultados,
    )

    caminho_saida = (
        OUTPUT_PATH
        / (
            caminho_imagem.stem
            + "_pipeline.jpg"
        )
    )

    preview.save(
        caminho_saida,
        quality=95,
    )

    print()
    print(
        f"Preview salvo em: "
        f"{caminho_saida}"
    )

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()