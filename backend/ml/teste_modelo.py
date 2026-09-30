from pathlib import Path

from PIL import Image

from backend.ml.modelo import ModeloPokemon


ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def testar_modelo():
    print("=" * 60)
    print("TESTE DO MÓDULO DE INFERÊNCIA")
    print("=" * 60)

    print()
    print("Carregando modelo...")

    modelo_pokemon = ModeloPokemon()

    print("Modelo carregado.")

    print()
    print(
        f"Dispositivo: "
        f"{modelo_pokemon.dispositivo}"
    )

    caminho_imagem = input(
        "\nDigite o caminho da imagem: "
    )

    caminho_imagem = Path(
        caminho_imagem.strip('"')
    )

    if not caminho_imagem.exists():
        print()
        print("Imagem não encontrada.")
        return

    with Image.open(caminho_imagem) as imagem:
        resultados = modelo_pokemon.prever(
            imagem
        )

    print()
    print("=" * 60)
    print("RESULTADO")
    print("=" * 60)

    print()

    for posicao, resultado in enumerate(
        resultados,
        start=1,
    ):
        print(
            f"{posicao}. "
            f"{resultado['pokemon']}: "
            f"{resultado['confianca']:.2f}%"
        )


if __name__ == "__main__":
    testar_modelo()