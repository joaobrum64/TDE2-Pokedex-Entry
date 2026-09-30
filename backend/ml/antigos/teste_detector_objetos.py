import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent

MODELO_PATH = (
    ROOT_DIR
    / "modelos"
    / "pokemon_object_detector.pth"
)

OUTPUT_PATH = (
    ROOT_DIR
    / "datasets"
    / "deteccao_externa"
)

SCORE_THRESHOLD = 0.50

COR_CAIXA = "red"
COR_TEXTO = "white"
ESPESSURA_CAIXA = 4


def carregar_modelo(dispositivo):
    if not MODELO_PATH.exists():
        raise FileNotFoundError(
            f"Detector não encontrado: {MODELO_PATH}"
        )

    checkpoint = torch.load(
        MODELO_PATH,
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


def preparar_imagem(caminho_imagem):
    with Image.open(caminho_imagem) as imagem_original:
        imagem_pil = imagem_original.convert("RGB")

    imagem_tensor = F.to_tensor(
        imagem_pil
    )

    return imagem_pil, imagem_tensor


def detectar(
    modelo,
    imagem_tensor,
    dispositivo,
):
    imagem_gpu = imagem_tensor.to(
        dispositivo
    )

    with torch.no_grad():
        resultados = modelo(
            [imagem_gpu]
        )

    resultado = resultados[0]

    boxes = resultado[
        "boxes"
    ].detach().cpu()

    scores = resultado[
        "scores"
    ].detach().cpu()

    labels = resultado[
        "labels"
    ].detach().cpu()

    deteccoes = []

    for box, score, label in zip(
        boxes,
        scores,
        labels,
    ):
        if label.item() != 1:
            continue

        if score.item() < SCORE_THRESHOLD:
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

    return deteccoes


def desenhar_deteccoes(
    imagem,
    deteccoes,
):
    imagem_saida = imagem.copy()

    desenho = ImageDraw.Draw(
        imagem_saida
    )

    for indice, deteccao in enumerate(
        deteccoes,
        start=1,
    ):
        x1, y1, x2, y2 = deteccao[
            "box"
        ]

        score = deteccao[
            "score"
        ]

        desenho.rectangle(
            (
                x1,
                y1,
                x2,
                y2,
            ),
            outline=COR_CAIXA,
            width=ESPESSURA_CAIXA,
        )

        texto = (
            f"Pokemon {indice} - "
            f"{score:.2f}%"
        )

        caixa_texto = desenho.textbbox(
            (0, 0),
            texto,
        )

        largura_texto = (
            caixa_texto[2]
            - caixa_texto[0]
        )

        altura_texto = (
            caixa_texto[3]
            - caixa_texto[1]
        )

        margem = 4

        texto_x = x1

        texto_y = max(
            0,
            y1
            - altura_texto
            - margem * 2,
        )

        desenho.rectangle(
            (
                texto_x,
                texto_y,
                texto_x
                + largura_texto
                + margem * 2,
                texto_y
                + altura_texto
                + margem * 2,
            ),
            fill=COR_CAIXA,
        )

        desenho.text(
            (
                texto_x + margem,
                texto_y + margem,
            ),
            texto,
            fill=COR_TEXTO,
        )

    return imagem_saida


def salvar_recortes(
    imagem,
    deteccoes,
    nome_base,
):
    pasta_recortes = (
        OUTPUT_PATH
        / f"{nome_base}_recortes"
    )

    pasta_recortes.mkdir(
        parents=True,
        exist_ok=True,
    )

    for indice, deteccao in enumerate(
        deteccoes,
        start=1,
    ):
        x1, y1, x2, y2 = deteccao[
            "box"
        ]

        recorte = imagem.crop(
            (
                x1,
                y1,
                x2,
                y2,
            )
        )

        caminho_recorte = (
            pasta_recortes
            / f"pokemon_{indice}.png"
        )

        recorte.save(
            caminho_recorte
        )


def main():
    if len(sys.argv) < 2:
        print("Uso:")
        print(
            "python -m "
            "backend.ml.antigos.teste_detector_objetos "
            "\"caminho_da_imagem\""
        )
        return

    caminho_imagem = Path(
        sys.argv[1]
    )

    if not caminho_imagem.exists():
        print(
            f"Imagem não encontrada: "
            f"{caminho_imagem}"
        )
        return

    dispositivo = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("TESTE EXTERNO DO DETECTOR DE OBJETOS")
    print("=" * 60)

    print()
    print(
        f"Imagem: {caminho_imagem}"
    )

    print(
        f"Dispositivo: {dispositivo}"
    )

    print()
    print("Carregando detector...")

    modelo = carregar_modelo(
        dispositivo
    )

    print("Detector carregado.")

    imagem_pil, imagem_tensor = preparar_imagem(
        caminho_imagem
    )

    deteccoes = detectar(
        modelo,
        imagem_tensor,
        dispositivo,
    )

    print()
    print("=" * 60)
    print("RESULTADO")
    print("=" * 60)

    print()
    print(
        f"Deteccoes encontradas: "
        f"{len(deteccoes)}"
    )

    if len(deteccoes) == 0:
        print()
        print(
            "Nenhum objeto Pokemon detectado."
        )

    else:
        for indice, deteccao in enumerate(
            deteccoes,
            start=1,
        ):
            print()
            print(
                f"Pokemon {indice}"
            )

            print(
                f"Score: "
                f"{deteccao['score']:.2f}%"
            )

            print(
                f"Box: "
                f"{deteccao['box']}"
            )

    OUTPUT_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    nome_base = caminho_imagem.stem

    imagem_saida = desenhar_deteccoes(
        imagem_pil,
        deteccoes,
    )

    caminho_saida = (
        OUTPUT_PATH
        / f"{nome_base}_deteccoes.jpg"
    )

    imagem_saida.save(
        caminho_saida,
        quality=95,
    )

    if len(deteccoes) > 0:
        salvar_recortes(
            imagem_pil,
            deteccoes,
            nome_base,
        )

    print()
    print("=" * 60)
    print("ARQUIVOS GERADOS")
    print("=" * 60)

    print()
    print(
        f"Preview: {caminho_saida}"
    )

    if len(deteccoes) > 0:
        print(
            f"Recortes: "
            f"{OUTPUT_PATH / (nome_base + '_recortes')}"
        )

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()