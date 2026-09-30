from pathlib import Path

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from backend.ml.preparar_dados_deteccao import criar_dataloaders


ROOT_DIR = Path(__file__).resolve().parent.parent.parent

MODELO_PATH = (
    ROOT_DIR
    / "modelos"
    / "pokemon_object_detector.pth"
)

SCORE_THRESHOLD = 0.50
IOU_THRESHOLD = 0.50


def calcular_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1_intersecao = max(ax1, bx1)
    y1_intersecao = max(ay1, by1)

    x2_intersecao = min(ax2, bx2)
    y2_intersecao = min(ay2, by2)

    largura_intersecao = max(
        0.0,
        x2_intersecao - x1_intersecao,
    )

    altura_intersecao = max(
        0.0,
        y2_intersecao - y1_intersecao,
    )

    area_intersecao = (
        largura_intersecao
        * altura_intersecao
    )

    area_a = max(
        0.0,
        ax2 - ax1,
    ) * max(
        0.0,
        ay2 - ay1,
    )

    area_b = max(
        0.0,
        bx2 - bx1,
    ) * max(
        0.0,
        by2 - by1,
    )

    area_uniao = (
        area_a
        + area_b
        - area_intersecao
    )

    if area_uniao <= 0:
        return 0.0

    return (
        area_intersecao
        / area_uniao
    )


def carregar_modelo(dispositivo):
    if not MODELO_PATH.exists():
        raise FileNotFoundError(
            f"Detector não encontrado: {MODELO_PATH}"
        )

    print("Carregando detector...")

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

    return modelo, checkpoint


def filtrar_deteccoes(resultado):
    boxes = resultado["boxes"].detach().cpu()

    scores = resultado["scores"].detach().cpu()

    labels = resultado["labels"].detach().cpu()

    boxes_filtradas = []

    scores_filtrados = []

    for box, score, label in zip(
        boxes,
        scores,
        labels,
    ):
        if (
            label.item() == 1
            and score.item() >= SCORE_THRESHOLD
        ):
            boxes_filtradas.append(
                box.tolist()
            )

            scores_filtrados.append(
                score.item()
            )

    return (
        boxes_filtradas,
        scores_filtrados,
    )


def associar_boxes(
    boxes_reais,
    boxes_previstas,
):
    pares = []

    for indice_real, box_real in enumerate(
        boxes_reais
    ):
        for indice_previsto, box_prevista in enumerate(
            boxes_previstas
        ):
            iou = calcular_iou(
                box_real,
                box_prevista,
            )

            pares.append(
                (
                    iou,
                    indice_real,
                    indice_previsto,
                )
            )

    pares.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    reais_usadas = set()
    previstas_usadas = set()

    matches = []

    for (
        iou,
        indice_real,
        indice_previsto,
    ) in pares:
        if iou < IOU_THRESHOLD:
            break

        if indice_real in reais_usadas:
            continue

        if indice_previsto in previstas_usadas:
            continue

        reais_usadas.add(
            indice_real
        )

        previstas_usadas.add(
            indice_previsto
        )

        matches.append(
            (
                indice_real,
                indice_previsto,
                iou,
            )
        )

    true_positives = len(
        matches
    )

    false_negatives = (
        len(boxes_reais)
        - true_positives
    )

    false_positives = (
        len(boxes_previstas)
        - true_positives
    )

    return (
        matches,
        true_positives,
        false_positives,
        false_negatives,
    )


def avaliar():
    dispositivo = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("AVALIACAO DO DETECTOR DE OBJETOS")
    print("=" * 60)

    print()
    print(
        f"Dispositivo: {dispositivo}"
    )

    if dispositivo.type == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print()

    modelo, checkpoint = carregar_modelo(
        dispositivo
    )

    print(
        f"Arquitetura: "
        f"{checkpoint['architecture']}"
    )

    print(
        f"Loss de validacao salva: "
        f"{checkpoint['validation_loss']:.4f}"
    )

    print()
    print("Preparando conjunto de teste...")

    (
        _,
        _,
        loader_teste,
    ) = criar_dataloaders()

    print(
        f"Imagens de teste: "
        f"{len(loader_teste.dataset)}"
    )

    total_imagens = 0

    total_objetos_reais = 0
    total_deteccoes = 0

    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    todos_ious = []
    todos_scores = []

    imagens_contagem_correta = 0

    imagens_vazias = 0
    imagens_vazias_corretas = 0

    imagens_com_pokemon = 0
    imagens_com_pokemon_detectado = 0

    with torch.no_grad():

        for indice_batch, (
            imagens,
            targets,
        ) in enumerate(
            loader_teste,
            start=1,
        ):
            imagens_gpu = [
                imagem.to(
                    dispositivo
                )
                for imagem in imagens
            ]

            resultados = modelo(
                imagens_gpu
            )

            for (
                imagem,
                target,
                resultado,
            ) in zip(
                imagens,
                targets,
                resultados,
            ):
                total_imagens += 1

                boxes_reais = (
                    target["boxes"]
                    .cpu()
                    .tolist()
                )

                (
                    boxes_previstas,
                    scores_previstos,
                ) = filtrar_deteccoes(
                    resultado
                )

                quantidade_real = len(
                    boxes_reais
                )

                quantidade_prevista = len(
                    boxes_previstas
                )

                total_objetos_reais += (
                    quantidade_real
                )

                total_deteccoes += (
                    quantidade_prevista
                )

                if (
                    quantidade_real
                    == quantidade_prevista
                ):
                    imagens_contagem_correta += 1

                if quantidade_real == 0:

                    imagens_vazias += 1

                    if quantidade_prevista == 0:
                        imagens_vazias_corretas += 1

                else:

                    imagens_com_pokemon += 1

                    if quantidade_prevista > 0:
                        imagens_com_pokemon_detectado += 1

                (
                    matches,
                    true_positives,
                    false_positives,
                    false_negatives,
                ) = associar_boxes(
                    boxes_reais,
                    boxes_previstas,
                )

                total_true_positives += (
                    true_positives
                )

                total_false_positives += (
                    false_positives
                )

                total_false_negatives += (
                    false_negatives
                )

                for (
                    _,
                    indice_previsto,
                    iou,
                ) in matches:
                    todos_ious.append(
                        iou
                    )

                    todos_scores.append(
                        scores_previstos[
                            indice_previsto
                        ]
                    )

            if indice_batch % 10 == 0:
                imagens_processadas = min(
                    indice_batch
                    * loader_teste.batch_size,
                    len(loader_teste.dataset),
                )

                print(
                    f"Processadas: "
                    f"{imagens_processadas}/"
                    f"{len(loader_teste.dataset)}"
                )

    if (
        total_true_positives
        + total_false_positives
        > 0
    ):
        precision = (
            total_true_positives
            / (
                total_true_positives
                + total_false_positives
            )
        ) * 100

    else:
        precision = 0.0

    if (
        total_true_positives
        + total_false_negatives
        > 0
    ):
        recall = (
            total_true_positives
            / (
                total_true_positives
                + total_false_negatives
            )
        ) * 100

    else:
        recall = 0.0

    if precision + recall > 0:
        f1 = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
        )

    else:
        f1 = 0.0

    if len(todos_ious) > 0:
        iou_medio = (
            sum(todos_ious)
            / len(todos_ious)
        ) * 100
    else:
        iou_medio = 0.0

    if len(todos_scores) > 0:
        score_medio = (
            sum(todos_scores)
            / len(todos_scores)
        ) * 100
    else:
        score_medio = 0.0

    taxa_contagem_correta = (
        imagens_contagem_correta
        / total_imagens
    ) * 100

    if imagens_vazias > 0:
        taxa_vazias_corretas = (
            imagens_vazias_corretas
            / imagens_vazias
        ) * 100
    else:
        taxa_vazias_corretas = 0.0

    if imagens_com_pokemon > 0:
        taxa_presenca_detectada = (
            imagens_com_pokemon_detectado
            / imagens_com_pokemon
        ) * 100
    else:
        taxa_presenca_detectada = 0.0

    print()
    print("=" * 60)
    print("RESULTADO GERAL")
    print("=" * 60)

    print()
    print(
        f"Imagens avaliadas: "
        f"{total_imagens}"
    )

    print(
        f"Objetos reais: "
        f"{total_objetos_reais}"
    )

    print(
        f"Deteccoes produzidas: "
        f"{total_deteccoes}"
    )

    print()
    print(
        f"True Positives: "
        f"{total_true_positives}"
    )

    print(
        f"False Positives: "
        f"{total_false_positives}"
    )

    print(
        f"False Negatives: "
        f"{total_false_negatives}"
    )

    print()
    print(
        f"Precision: "
        f"{precision:.2f}%"
    )

    print(
        f"Recall: "
        f"{recall:.2f}%"
    )

    print(
        f"F1 Score: "
        f"{f1:.2f}%"
    )

    print()
    print(
        f"IoU medio dos matches: "
        f"{iou_medio:.2f}%"
    )

    print(
        f"Score medio das deteccoes corretas: "
        f"{score_medio:.2f}%"
    )

    print()
    print("=" * 60)
    print("CONTAGEM DE POKEMON")
    print("=" * 60)

    print()
    print(
        f"Imagens com quantidade exata "
        f"de Pokemon detectada: "
        f"{imagens_contagem_correta}/"
        f"{total_imagens}"
    )

    print(
        f"Taxa de contagem correta: "
        f"{taxa_contagem_correta:.2f}%"
    )

    print()
    print("=" * 60)
    print("IMAGENS SEM POKEMON")
    print("=" * 60)

    print()
    print(
        f"Imagens vazias: "
        f"{imagens_vazias}"
    )

    print(
        f"Vazias corretamente sem deteccao: "
        f"{imagens_vazias_corretas}"
    )

    print(
        f"Taxa de rejeicao correta: "
        f"{taxa_vazias_corretas:.2f}%"
    )

    print()
    print("=" * 60)
    print("IMAGENS COM POKEMON")
    print("=" * 60)

    print()
    print(
        f"Imagens com Pokemon: "
        f"{imagens_com_pokemon}"
    )

    print(
        f"Imagens onde pelo menos um "
        f"Pokemon foi detectado: "
        f"{imagens_com_pokemon_detectado}"
    )

    print(
        f"Taxa de presenca detectada: "
        f"{taxa_presenca_detectada:.2f}%"
    )

    print()
    print("=" * 60)
    print("CONFIGURACAO")
    print("=" * 60)

    print()
    print(
        f"Score threshold: "
        f"{SCORE_THRESHOLD}"
    )

    print(
        f"IoU threshold: "
        f"{IOU_THRESHOLD}"
    )

    print()
    print("=" * 60)
    print("AVALIACAO CONCLUIDA")
    print("=" * 60)


if __name__ == "__main__":
    avaliar()