from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet18

from backend.ml.preparar_dados import testar_dataloaders


ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODELO_PATH = ROOT_DIR / "modelos" / "pokemon_resnet18.pth"


def carregar_modelo(dispositivo):
    if not MODELO_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em: {MODELO_PATH}"
        )

    print("Carregando modelo salvo...")

    checkpoint = torch.load(
        MODELO_PATH,
        map_location=dispositivo,
        weights_only=False
    )

    classes = checkpoint["classes"]
    quantidade_classes = checkpoint["num_classes"]

    modelo = resnet18(weights=None)

    quantidade_entradas = modelo.fc.in_features

    modelo.fc = nn.Linear(
        quantidade_entradas,
        quantidade_classes
    )

    modelo.load_state_dict(
        checkpoint["model_state_dict"]
    )

    modelo = modelo.to(dispositivo)

    modelo.eval()

    return modelo, classes, checkpoint


def avaliar_modelo(
    modelo,
    loader_teste,
    classes,
    dispositivo
):
    total = 0
    corretos = 0

    total_por_classe = Counter()
    corretos_por_classe = Counter()

    confusoes = Counter()

    with torch.no_grad():
        for imagens, rotulos in loader_teste:
            imagens = imagens.to(dispositivo)
            rotulos = rotulos.to(dispositivo)

            saidas = modelo(imagens)

            _, previsoes = torch.max(
                saidas,
                dim=1
            )

            total += rotulos.size(0)

            corretos += (
                previsoes == rotulos
            ).sum().item()

            for rotulo_real, previsao in zip(
                rotulos.cpu().tolist(),
                previsoes.cpu().tolist()
            ):
                nome_real = classes[rotulo_real]
                nome_previsto = classes[previsao]

                total_por_classe[nome_real] += 1

                if rotulo_real == previsao:
                    corretos_por_classe[nome_real] += 1
                else:
                    confusoes[
                        (nome_real, nome_previsto)
                    ] += 1

    accuracy_geral = (
        corretos / total
    ) * 100

    accuracy_por_classe = []

    for classe in classes:
        quantidade_total = total_por_classe[classe]

        if quantidade_total == 0:
            accuracy = 0.0
        else:
            accuracy = (
                corretos_por_classe[classe]
                / quantidade_total
            ) * 100

        accuracy_por_classe.append(
            (
                classe,
                accuracy,
                corretos_por_classe[classe],
                quantidade_total
            )
        )

    return (
        accuracy_geral,
        corretos,
        total,
        accuracy_por_classe,
        confusoes
    )


def main():
    dispositivo = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("AVALIAÇÃO DO MODELO")
    print("=" * 60)

    print()
    print(f"Dispositivo: {dispositivo}")

    if dispositivo.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print()

    modelo, classes_modelo, checkpoint = (
        carregar_modelo(dispositivo)
    )

    print(
        f"Arquitetura salva: "
        f"{checkpoint['architecture']}"
    )

    print(
        f"Classes no modelo: "
        f"{checkpoint['num_classes']}"
    )

    print(
        f"Melhor accuracy de validação salva: "
        f"{checkpoint['validation_accuracy']:.2f}%"
    )

    print()
    print("Preparando conjunto de teste...")

    (
        classes_dataset,
        class_to_idx,
        loader_treino,
        loader_validacao,
        loader_teste,
    ) = testar_dataloaders()

    if classes_modelo != classes_dataset:
        raise ValueError(
            "A ordem das classes do modelo é diferente "
            "da ordem das classes do dataset."
        )

    print()
    print("=" * 60)
    print("EXECUTANDO CONJUNTO DE TESTE")
    print("=" * 60)

    (
        accuracy_geral,
        corretos,
        total,
        accuracy_por_classe,
        confusoes
    ) = avaliar_modelo(
        modelo,
        loader_teste,
        classes_modelo,
        dispositivo
    )

    print()
    print(
        f"Acertos: {corretos}/{total}"
    )

    print(
        f"Accuracy de teste: "
        f"{accuracy_geral:.2f}%"
    )

    ranking = sorted(
        accuracy_por_classe,
        key=lambda item: item[1],
        reverse=True
    )

    print()
    print("=" * 60)
    print("10 MELHORES CLASSES")
    print("=" * 60)

    for (
        classe,
        accuracy,
        acertos,
        quantidade
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

    for (
        classe,
        accuracy,
        acertos,
        quantidade
    ) in ranking[-10:]:
        print(
            f"{classe}: "
            f"{accuracy:.2f}% "
            f"({acertos}/{quantidade})"
        )

    print()
    print("=" * 60)
    print("15 CONFUSÕES MAIS FREQUENTES")
    print("=" * 60)

    if not confusoes:
        print("Nenhuma confusão encontrada.")
    else:
        for (
            nome_real,
            nome_previsto
        ), quantidade in confusoes.most_common(15):

            print(
                f"{nome_real} -> "
                f"{nome_previsto}: "
                f"{quantidade}"
            )

    print()
    print("=" * 60)
    print("AVALIAÇÃO CONCLUÍDA")
    print("=" * 60)


if __name__ == "__main__":
    main()