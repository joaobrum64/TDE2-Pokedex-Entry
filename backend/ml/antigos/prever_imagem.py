import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent

MODELO_PATH = (
    ROOT_DIR
    / "modelos"
    / "pokemon_resnet18.pth"
)


transform_imagem = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def carregar_modelo(dispositivo):
    if not MODELO_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em: {MODELO_PATH}"
        )

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

    return modelo, classes


def preparar_imagem(caminho_imagem):
    with Image.open(caminho_imagem) as imagem:
        imagem = imagem.convert("RGB")

    imagem = transform_imagem(imagem)

    imagem = imagem.unsqueeze(0)

    return imagem


def prever_imagem(
    modelo,
    classes,
    caminho_imagem,
    dispositivo
):
    imagem = preparar_imagem(
        caminho_imagem
    )

    imagem = imagem.to(dispositivo)

    with torch.no_grad():
        saidas = modelo(imagem)

        probabilidades = torch.softmax(
            saidas,
            dim=1
        )

    probabilidades = probabilidades[0]

    top_probabilidades, top_indices = torch.topk(
        probabilidades,
        k=5
    )

    resultados = []

    for probabilidade, indice in zip(
        top_probabilidades,
        top_indices
    ):
        nome_pokemon = classes[
            indice.item()
        ]

        confianca = (
            probabilidade.item()
            * 100
        )

        resultados.append(
            (
                nome_pokemon,
                confianca
            )
        )

    return resultados


def main():
    if len(sys.argv) < 2:
        print(
            "Uso:"
        )

        print(
            'python -m backend.ml.antigos.prever_imagem '
            '"caminho_da_imagem"'
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
    print("PREVISÃO DE IMAGEM")
    print("=" * 60)

    print()
    print(
        f"Imagem: {caminho_imagem}"
    )

    print(
        f"Dispositivo: {dispositivo}"
    )

    print()
    print("Carregando modelo...")

    modelo, classes = carregar_modelo(
        dispositivo
    )

    print("Modelo carregado.")

    resultados = prever_imagem(
        modelo,
        classes,
        caminho_imagem,
        dispositivo
    )

    melhor_pokemon, melhor_confianca = (
        resultados[0]
    )

    print()
    print("=" * 60)
    print("RESULTADO")
    print("=" * 60)

    print()
    print(
        f"Pokémon previsto: "
        f"{melhor_pokemon}"
    )

    print(
        f"Confiança: "
        f"{melhor_confianca:.2f}%"
    )

    print()
    print("Top 5 previsões:")

    for posicao, (
        pokemon,
        confianca
    ) in enumerate(
        resultados,
        start=1
    ):
        print(
            f"{posicao}. "
            f"{pokemon}: "
            f"{confianca:.2f}%"
        )

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()