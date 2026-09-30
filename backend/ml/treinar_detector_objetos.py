from pathlib import Path

import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from backend.ml.preparar_dados_deteccao import criar_dataloaders


ROOT_DIR = Path(__file__).resolve().parent.parent.parent

MODELOS_PATH = ROOT_DIR / "modelos"

MODELO_PATH = (
    MODELOS_PATH
    / "pokemon_object_detector.pth"
)


NUM_EPOCAS = 3

TAXA_APRENDIZADO = 0.005


def criar_modelo(dispositivo):
    print()
    print("Carregando Faster R-CNN pre-treinada...")

    pesos = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

    modelo = fasterrcnn_resnet50_fpn(
        weights=pesos
    )

    quantidade_entradas = (
        modelo.roi_heads.box_predictor.cls_score.in_features
    )

    # 2 classes:
    # 0 = background
    # 1 = pokemon
    modelo.roi_heads.box_predictor = FastRCNNPredictor(
        quantidade_entradas,
        2,
    )

    modelo = modelo.to(
        dispositivo
    )

    return modelo


def mover_targets_para_dispositivo(
    targets,
    dispositivo,
):
    targets_gpu = []

    for target in targets:
        novo_target = {}

        for chave, valor in target.items():
            novo_target[chave] = valor.to(
                dispositivo
            )

        targets_gpu.append(
            novo_target
        )

    return targets_gpu


def treinar_uma_epoca(
    modelo,
    loader,
    otimizador,
    dispositivo,
):
    modelo.train()

    loss_total = 0.0
    quantidade_batches = 0

    for indice_batch, (
        imagens,
        targets,
    ) in enumerate(
        loader,
        start=1,
    ):
        imagens = [
            imagem.to(
                dispositivo
            )
            for imagem in imagens
        ]

        targets = mover_targets_para_dispositivo(
            targets,
            dispositivo,
        )

        otimizador.zero_grad()

        losses = modelo(
            imagens,
            targets,
        )

        loss = sum(
            valor
            for valor in losses.values()
        )

        loss.backward()

        otimizador.step()

        loss_total += loss.item()
        quantidade_batches += 1

        if indice_batch % 50 == 0:
            print(
                f"Batch "
                f"{indice_batch}/"
                f"{len(loader)} "
                f"- Loss: "
                f"{loss.item():.4f}"
            )

    loss_medio = (
        loss_total
        / quantidade_batches
    )

    return loss_medio


def calcular_loss_validacao(
    modelo,
    loader,
    dispositivo,
):
    # O Faster R-CNN retorna losses
    # somente em modo de treino.
    #
    # Não calculamos gradientes durante
    # a validação.
    modelo.train()

    loss_total = 0.0
    quantidade_batches = 0

    with torch.no_grad():
        for imagens, targets in loader:
            imagens = [
                imagem.to(
                    dispositivo
                )
                for imagem in imagens
            ]

            targets = mover_targets_para_dispositivo(
                targets,
                dispositivo,
            )

            losses = modelo(
                imagens,
                targets,
            )

            loss = sum(
                valor
                for valor in losses.values()
            )

            loss_total += loss.item()
            quantidade_batches += 1

    loss_medio = (
        loss_total
        / quantidade_batches
    )

    return loss_medio


def salvar_modelo(
    modelo,
    melhor_loss_validacao,
):
    MODELOS_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_state_dict": modelo.state_dict(),
        "architecture": "fasterrcnn_resnet50_fpn",
        "num_classes": 2,
        "classes": [
            "background",
            "pokemon",
        ],
        "validation_loss": melhor_loss_validacao,
    }

    torch.save(
        checkpoint,
        MODELO_PATH,
    )


def treinar_detector():
    dispositivo = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("TREINAMENTO DO DETECTOR DE OBJETOS")
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
    print("Preparando datasets...")

    (
        loader_treino,
        loader_validacao,
        loader_teste,
    ) = criar_dataloaders()

    print()
    print(
        f"Treino: "
        f"{len(loader_treino.dataset)} imagens"
    )

    print(
        f"Validacao: "
        f"{len(loader_validacao.dataset)} imagens"
    )

    print(
        f"Teste: "
        f"{len(loader_teste.dataset)} imagens"
    )

    modelo = criar_modelo(
        dispositivo
    )

    parametros = [
        parametro
        for parametro in modelo.parameters()
        if parametro.requires_grad
    ]

    otimizador = torch.optim.SGD(
        parametros,
        lr=TAXA_APRENDIZADO,
        momentum=0.9,
        weight_decay=0.0005,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        otimizador,
        step_size=2,
        gamma=0.1,
    )

    melhor_loss_validacao = float(
        "inf"
    )

    print()
    print("=" * 60)
    print("INICIO DO TREINAMENTO")
    print("=" * 60)

    for epoca in range(
        1,
        NUM_EPOCAS + 1,
    ):
        print()
        print(
            f"Epoca "
            f"{epoca}/"
            f"{NUM_EPOCAS}"
        )

        print()
        print("Treinando...")

        loss_treino = treinar_uma_epoca(
            modelo,
            loader_treino,
            otimizador,
            dispositivo,
        )

        print()
        print("Validando...")

        loss_validacao = calcular_loss_validacao(
            modelo,
            loader_validacao,
            dispositivo,
        )

        print()
        print(
            f"Treino - Loss: "
            f"{loss_treino:.4f}"
        )

        print(
            f"Validacao - Loss: "
            f"{loss_validacao:.4f}"
        )

        if (
            loss_validacao
            < melhor_loss_validacao
        ):
            melhor_loss_validacao = (
                loss_validacao
            )

            salvar_modelo(
                modelo,
                melhor_loss_validacao,
            )

            print(
                "Melhor detector salvo."
            )

        scheduler.step()

        print(
            f"Learning rate atual: "
            f"{scheduler.get_last_lr()}"
        )

    print()
    print("=" * 60)
    print("TREINAMENTO CONCLUIDO")
    print("=" * 60)

    print()
    print(
        f"Melhor loss de validacao: "
        f"{melhor_loss_validacao:.4f}"
    )

    print(
        f"Detector salvo em: "
        f"{MODELO_PATH}"
    )

    print()
    print(
        "O conjunto de teste NAO foi "
        "utilizado no treinamento."
    )


if __name__ == "__main__":
    treinar_detector()