import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


ROOT_DIR = Path(__file__).resolve().parent.parent.parent

MODELO_PATH = (
    ROOT_DIR
    / "modelos"
    / "pokemon_detector_v2.pth"
)


TRANSFORM_IMAGEM = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def carregar_detector(dispositivo):
    if not MODELO_PATH.exists():
        raise FileNotFoundError(
            f"Detector não encontrado em: {MODELO_PATH}"
        )

    checkpoint = torch.load(
        MODELO_PATH,
        map_location=dispositivo,
        weights_only=False,
    )

    modelo = resnet18(
        weights=None
    )

    quantidade_entradas = (
        modelo.fc.in_features
    )

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

    return modelo


def preparar_imagem(caminho_imagem):
    with Image.open(caminho_imagem) as imagem:
        imagem = imagem.convert("RGB")

    imagem = TRANSFORM_IMAGEM(
        imagem
    )

    imagem = imagem.unsqueeze(0)

    return imagem


def detectar(
    modelo,
    caminho_imagem,
    dispositivo,
):
    imagem = preparar_imagem(
        caminho_imagem
    )

    imagem = imagem.to(
        dispositivo
    )

    with torch.no_grad():
        saidas = modelo(
            imagem
        )

        probabilidades = torch.softmax(
            saidas,
            dim=1,
        )

    probabilidade_nao_pokemon = (
        probabilidades[0][0].item()
        * 100
    )

    probabilidade_pokemon = (
        probabilidades[0][1].item()
        * 100
    )

    classe_prevista = torch.argmax(
        probabilidades,
        dim=1,
    ).item()

    return {
        "classe": classe_prevista,
        "pokemon": probabilidade_pokemon,
        "nao_pokemon": probabilidade_nao_pokemon,
    }


def main():
    if len(sys.argv) < 2:
        print("Uso:")
        print(
            'python backend/ml/teste_detector.py '
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
    print("TESTE DO DETECTOR BINÁRIO")
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

    modelo = carregar_detector(
        dispositivo
    )

    print("Detector carregado.")

    resultado = detectar(
        modelo,
        caminho_imagem,
        dispositivo,
    )

    print()
    print("=" * 60)
    print("RESULTADO")
    print("=" * 60)

    print()

    if resultado["classe"] == 1:
        print(
            "Resultado: Pokémon Gen 1"
        )

        print(
            f"Confiança: "
            f"{resultado['pokemon']:.2f}%"
        )

    else:
        print(
            "Resultado: Não-Pokémon Gen 1"
        )

        print(
            f"Confiança: "
            f"{resultado['nao_pokemon']:.2f}%"
        )

    print()
    print("Probabilidades:")

    print(
        f"Pokémon Gen 1: "
        f"{resultado['pokemon']:.2f}%"
    )

    print(
        f"Não-Pokémon Gen 1: "
        f"{resultado['nao_pokemon']:.2f}%"
    )

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()