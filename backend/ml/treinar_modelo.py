from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import ResNet18_Weights, resnet18

from preparar_dados import testar_dataloaders


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODELOS_PATH = ROOT_DIR / "modelos"

MODELO_PATH = MODELOS_PATH / "pokemon_resnet18.pth"

NUM_EPOCAS = 5
TAXA_APRENDIZADO = 0.001


def criar_modelo(quantidade_classes, dispositivo):
    print()
    print("Carregando ResNet18 pré-treinada...")

    pesos = ResNet18_Weights.DEFAULT

    modelo = resnet18(weights=pesos)

    quantidade_entradas = modelo.fc.in_features

    modelo.fc = nn.Linear(
        quantidade_entradas,
        quantidade_classes
    )

    modelo = modelo.to(dispositivo)

    return modelo


def treinar_uma_epoca(
    modelo,
    loader,
    criterio,
    otimizador,
    dispositivo
):
    modelo.train()

    loss_total = 0.0
    quantidade_corretos = 0
    quantidade_total = 0

    for imagens, rotulos in loader:
        imagens = imagens.to(dispositivo)
        rotulos = rotulos.to(dispositivo)

        otimizador.zero_grad()

        saidas = modelo(imagens)

        loss = criterio(saidas, rotulos)

        loss.backward()

        otimizador.step()

        loss_total += loss.item() * imagens.size(0)

        _, previsoes = torch.max(
            saidas,
            dim=1
        )

        quantidade_total += rotulos.size(0)

        quantidade_corretos += (
            previsoes == rotulos
        ).sum().item()

    loss_medio = (
        loss_total
        / quantidade_total
    )

    accuracy = (
        quantidade_corretos
        / quantidade_total
    ) * 100

    return loss_medio, accuracy


def validar(
    modelo,
    loader,
    criterio,
    dispositivo
):
    modelo.eval()

    loss_total = 0.0
    quantidade_corretos = 0
    quantidade_total = 0

    with torch.no_grad():
        for imagens, rotulos in loader:
            imagens = imagens.to(dispositivo)
            rotulos = rotulos.to(dispositivo)

            saidas = modelo(imagens)

            loss = criterio(
                saidas,
                rotulos
            )

            loss_total += (
                loss.item()
                * imagens.size(0)
            )

            _, previsoes = torch.max(
                saidas,
                dim=1
            )

            quantidade_total += (
                rotulos.size(0)
            )

            quantidade_corretos += (
                previsoes == rotulos
            ).sum().item()

    loss_medio = (
        loss_total
        / quantidade_total
    )

    accuracy = (
        quantidade_corretos
        / quantidade_total
    ) * 100

    return loss_medio, accuracy


def salvar_modelo(
    modelo,
    classes,
    melhor_accuracy
):
    MODELOS_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {
        "model_state_dict": modelo.state_dict(),
        "classes": classes,
        "num_classes": len(classes),
        "architecture": "resnet18",
        "validation_accuracy": melhor_accuracy,
    }

    torch.save(
        checkpoint,
        MODELO_PATH
    )


def treinar_modelo():
    dispositivo = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("TREINAMENTO DO CLASSIFICADOR")
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

    (
        classes,
        class_to_idx,
        loader_treino,
        loader_validacao,
        loader_teste,
    ) = testar_dataloaders()

    quantidade_classes = len(classes)

    print()
    print(
        f"Quantidade de classes: "
        f"{quantidade_classes}"
    )

    modelo = criar_modelo(
        quantidade_classes,
        dispositivo
    )

    criterio = nn.CrossEntropyLoss()

    otimizador = optim.Adam(
        modelo.parameters(),
        lr=TAXA_APRENDIZADO
    )

    melhor_accuracy = 0.0

    print()
    print("=" * 60)
    print("INÍCIO DO TREINAMENTO")
    print("=" * 60)

    for epoca in range(
        1,
        NUM_EPOCAS + 1
    ):
        print()
        print(
            f"Época {epoca}/{NUM_EPOCAS}"
        )

        loss_treino, accuracy_treino = (
            treinar_uma_epoca(
                modelo,
                loader_treino,
                criterio,
                otimizador,
                dispositivo
            )
        )

        (
            loss_validacao,
            accuracy_validacao
        ) = validar(
            modelo,
            loader_validacao,
            criterio,
            dispositivo
        )

        print(
            f"Treino - "
            f"Loss: {loss_treino:.4f} | "
            f"Accuracy: {accuracy_treino:.2f}%"
        )

        print(
            f"Validação - "
            f"Loss: {loss_validacao:.4f} | "
            f"Accuracy: {accuracy_validacao:.2f}%"
        )

        if accuracy_validacao > melhor_accuracy:
            melhor_accuracy = (
                accuracy_validacao
            )

            salvar_modelo(
                modelo,
                classes,
                melhor_accuracy
            )

            print(
                "Melhor modelo salvo."
            )

    print()
    print("=" * 60)
    print("TREINAMENTO CONCLUÍDO")
    print("=" * 60)

    print()
    print(
        f"Melhor accuracy de validação: "
        f"{melhor_accuracy:.2f}%"
    )

    print(
        f"Modelo salvo em: "
        f"{MODELO_PATH}"
    )

    print()
    print(
        "O conjunto de teste NÃO foi "
        "utilizado durante o treinamento."
    )


if __name__ == "__main__":
    treinar_modelo()
