from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet18

from preparar_dados import testar_dataloaders


ROOT_DIR = Path(__file__).resolve().parent.parent.parent

MODELO_PATH = (
    ROOT_DIR
    / "modelos"
    / "pokemon_resnet18_v2.pth"
)


def carregar_modelo(dispositivo):
    if not MODELO_PATH.exists():
        raise FileNotFoundError(
            f"Modelo nao encontrado em: {MODELO_PATH}"
        )

    print("Carregando modelo v2...")

    checkpoint = torch.load(
        MODELO_PATH,
        map_location=dispositivo,
        weights_only=False,
    )

    classes = checkpoint["classes"]

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

    return (
        modelo,
        classes,
        checkpoint,
    )


def avaliar_modelo(
    modelo,
    loader_teste,
    classes,
    dispositivo,
):
    total = 0
    corretos = 0

    total_por_classe = Counter()
    corretos_por_classe = Counter()

    confusoes = Counter()

    with torch.no_grad():
        for imagens, rotulos in loader_teste:
            imagens = imagens.to(
                dispositivo
            )

            rotulos = rotulos.to(
                dispositivo
            )

            saidas = modelo(
                imagens
            )

            previsoes = torch.argmax(
                saidas,
                dim=1,
            )

            total += rotulos.size(0)

            corretos += (
                previsoes == rotulos
            ).sum().item()

            rotulos_cpu = (
                rotulos
                .cpu()
                .tolist()
            )

            previsoes_cpu = (
                previsoes
                .cpu()
                .tolist()
            )

            for rotulo_real, previsao in zip(
                rotulos_cpu,
                previsoes_cpu,
            ):
                nome_real = classes[
                    rotulo_real
                ]

                nome_previsto = classes[
                    previsao
                ]

                total_por_classe[
                    nome_real
                ] += 1

                if rotulo_real == previsao:
                    corretos_por_classe[
                        nome_real
                    ] += 1

                else:
                    confusoes[
                        (
                            nome_real,
                            nome_previsto,
                        )
                    ] += 1

    accuracy_geral = (
        corretos
        / total
    ) * 100

    accuracy_por_classe = []

    for classe in classes:
        quantidade = total_por_classe[
            classe
        ]

        acertos = corretos_por_classe[
            classe
        ]

        if quantidade > 0:
            accuracy = (
                acertos
                / quantidade
            ) * 100

        else:
            accuracy = 0.0

        accuracy_por_classe.append(
            (
                classe,
                accuracy,
                acertos,
                quantidade,
            )
        )

    return (
        accuracy_geral,
        corretos,
        total,
        accuracy_por_classe,
        confusoes,
    )


def main():
    dispositivo = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("AVALIACAO DO CLASSIFICADOR V2")
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

    (
        modelo,
        classes_modelo,
        checkpoint,
    ) = carregar_modelo(
        dispositivo
    )

    print(
        f"Arquitetura: "
        f"{checkpoint['architecture']}"
    )

    print(
        f"Epoca salva: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Melhor validacao: "
        f"{checkpoint['validation_accuracy']:.2f}%"
    )

    print()
    print(
        "Preparando conjunto de teste..."
    )

    (
        classes_dataset,
        class_to_idx,
        loader_treino,
        loader_validacao,
        loader_teste,
    ) = testar_dataloaders()

    if classes_modelo != classes_dataset:
        raise ValueError(
            "A ordem das classes do modelo "
            "nao corresponde ao dataset."
        )

    print()
    print("=" * 60)
    print("EXECUTANDO TESTE")
    print("=" * 60)

    (
        accuracy_geral,
        corretos,
        total,
        accuracy_por_classe,
        confusoes,
    ) = avaliar_modelo(
        modelo,
        loader_teste,
        classes_modelo,
        dispositivo,
    )

    print()
    print(
        f"Acertos: "
        f"{corretos}/{total}"
    )

    print(
        f"Accuracy de teste: "
        f"{accuracy_geral:.2f}%"
    )

    ranking = sorted(
        accuracy_por_classe,
        key=lambda item: item[1],
        reverse=True,
    )

    print()
    print("=" * 60)
    print("10 MELHORES CLASSES")
    print("=" * 60)

    for (
        classe,
        accuracy,
        acertos,
        quantidade,
    ) in ranking[:10]:
        print(
            f"{classe}: "
            f"{accuracy:.2f}% "
            f"({acertos}/{quantidade})"
        )

    print()
    print("=" * 60)
    print("10 PIORES CLASSES")
    print("=" * 60)

    piores_classes = sorted(
        accuracy_por_classe,
        key=lambda item: item[1],
    )

    for (
        classe,
        accuracy,
        acertos,
        quantidade,
    ) in piores_classes[:10]:
        print(
            f"{classe}: "
            f"{accuracy:.2f}% "
            f"({acertos}/{quantidade})"
        )

    print()
    print("=" * 60)
    print("15 CONFUSOES MAIS FREQUENTES")
    print("=" * 60)

    if len(confusoes) == 0:
        print()
        print(
            "Nenhuma confusao encontrada."
        )

    else:
        print()

        confusoes_frequentes = (
            confusoes.most_common(15)
        )

        for (
            par,
            quantidade,
        ) in confusoes_frequentes:
            nome_real = par[0]
            nome_previsto = par[1]

            print(
                f"{nome_real} -> "
                f"{nome_previsto}: "
                f"{quantidade}"
            )

    print()
    print("=" * 60)
    print("COMPARACAO V1 X V2")
    print("=" * 60)

    print()
    print(
        "V1 - Validacao: 67.49%"
    )

    print(
        "V1 - Teste: 66.44%"
    )

    print()

    print(
        f"V2 - Validacao: "
        f"{checkpoint['validation_accuracy']:.2f}%"
    )

    print(
        f"V2 - Teste: "
        f"{accuracy_geral:.2f}%"
    )

    diferenca_teste = (
        accuracy_geral
        - 66.44
    )

    print()

    print(
        f"Diferenca V2 - V1 no teste: "
        f"{diferenca_teste:+.2f} "
        f"pontos percentuais"
    )

    print()
    print("=" * 60)
    print("AVALIACAO CONCLUIDA")
    print("=" * 60)


if __name__ == "__main__":
    main()