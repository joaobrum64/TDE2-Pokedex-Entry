from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet18

from preparar_dados_binario import preparar_dados_binarios


ROOT_DIR = Path(__file__).resolve().parent.parent.parent

MODELO_PATH = (
    ROOT_DIR
    / "modelos"
    / "pokemon_detector_v2.pth"
)


def carregar_detector(dispositivo):
    if not MODELO_PATH.exists():
        raise FileNotFoundError(
            f"Detector não encontrado em: {MODELO_PATH}"
        )

    print("Carregando detector...")

    checkpoint = torch.load(
        MODELO_PATH,
        map_location=dispositivo,
        weights_only=False,
    )

    modelo = resnet18(
        weights=None
    )

    quantidade_entradas = modelo.fc.in_features

    modelo.fc = nn.Linear(
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


def avaliar_detector(
    modelo,
    loader_teste,
    dispositivo,
):
    total = 0
    corretos = 0

    # Classe real 1 = Pokémon
    pokemon_total = 0
    pokemon_corretos = 0
    pokemon_rejeitados = 0

    # Classe real 0 = não-Pokémon
    nao_pokemon_total = 0
    nao_pokemon_corretos = 0
    nao_pokemon_aceitos = 0

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

            probabilidades = torch.softmax(
                saidas,
                dim=1,
            )

            _, previsoes = torch.max(
                probabilidades,
                dim=1,
            )

            total += rotulos.size(0)

            corretos += (
                previsoes == rotulos
            ).sum().item()

            for rotulo_real, previsao in zip(
                rotulos.cpu().tolist(),
                previsoes.cpu().tolist(),
            ):

                if rotulo_real == 1:
                    pokemon_total += 1

                    if previsao == 1:
                        pokemon_corretos += 1

                    else:
                        pokemon_rejeitados += 1

                else:
                    nao_pokemon_total += 1

                    if previsao == 0:
                        nao_pokemon_corretos += 1

                    else:
                        nao_pokemon_aceitos += 1

    accuracy_geral = (
        corretos
        / total
    ) * 100

    taxa_pokemon = (
        pokemon_corretos
        / pokemon_total
    ) * 100

    taxa_rejeicao = (
        nao_pokemon_corretos
        / nao_pokemon_total
    ) * 100

    return {
        "total": total,
        "corretos": corretos,
        "accuracy_geral": accuracy_geral,

        "pokemon_total": pokemon_total,
        "pokemon_corretos": pokemon_corretos,
        "pokemon_rejeitados": pokemon_rejeitados,
        "taxa_pokemon": taxa_pokemon,

        "nao_pokemon_total": nao_pokemon_total,
        "nao_pokemon_corretos": nao_pokemon_corretos,
        "nao_pokemon_aceitos": nao_pokemon_aceitos,
        "taxa_rejeicao": taxa_rejeicao,
    }


def main():
    dispositivo = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("AVALIAÇÃO DO DETECTOR BINÁRIO")
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

    modelo, checkpoint = carregar_detector(
        dispositivo
    )

    print(
        f"Arquitetura: "
        f"{checkpoint['architecture']}"
    )

    print(
        f"Classes: "
        f"{checkpoint['classes']}"
    )

    print(
        f"Melhor accuracy de validação salva: "
        f"{checkpoint['validation_accuracy']:.2f}%"
    )

    print()
    print("Preparando conjunto de teste...")

    (
        loader_treino,
        loader_validacao,
        loader_teste,
    ) = preparar_dados_binarios()

    print()
    print("=" * 60)
    print("EXECUTANDO TESTE")
    print("=" * 60)

    resultados = avaliar_detector(
        modelo,
        loader_teste,
        dispositivo,
    )

    print()
    print(
        f"Accuracy geral: "
        f"{resultados['accuracy_geral']:.2f}%"
    )

    print(
        f"Acertos: "
        f"{resultados['corretos']}/"
        f"{resultados['total']}"
    )

    print()
    print("=" * 60)
    print("POKÉMON GEN 1")
    print("=" * 60)

    print()
    print(
        f"Total: "
        f"{resultados['pokemon_total']}"
    )

    print(
        f"Aceitos corretamente: "
        f"{resultados['pokemon_corretos']}"
    )

    print(
        f"Rejeitados incorretamente: "
        f"{resultados['pokemon_rejeitados']}"
    )

    print(
        f"Taxa de aceitação: "
        f"{resultados['taxa_pokemon']:.2f}%"
    )

    print()
    print("=" * 60)
    print("NÃO-POKÉMON")
    print("=" * 60)

    print()
    print(
        f"Total: "
        f"{resultados['nao_pokemon_total']}"
    )

    print(
        f"Rejeitados corretamente: "
        f"{resultados['nao_pokemon_corretos']}"
    )

    print(
        f"Aceitos incorretamente: "
        f"{resultados['nao_pokemon_aceitos']}"
    )

    print(
        f"Taxa de rejeição: "
        f"{resultados['taxa_rejeicao']:.2f}%"
    )

    print()
    print("=" * 60)
    print("COMPARAÇÃO COM BASELINE")
    print("=" * 60)

    print()
    print(
        "Threshold simples anterior: 86.72%"
    )

    print(
        f"Detector binário: "
        f"{resultados['accuracy_geral']:.2f}%"
    )

    print()
    print("=" * 60)
    print("AVALIAÇÃO CONCLUÍDA")
    print("=" * 60)


if __name__ == "__main__":
    main()