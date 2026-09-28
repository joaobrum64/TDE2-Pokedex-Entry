import sys
from pathlib import Path

from PIL import Image

from pipeline import PipelinePokemon


def main():
    if len(sys.argv) < 2:
        print("Uso:")
        print('python -m backend.ml.teste_pipeline "caminho_imagem"')
        return

    caminho_imagem = Path(sys.argv[1])

    if not caminho_imagem.exists():
        print(f"Imagem nao encontrada: {caminho_imagem}")
        return

    print("Carregando pipeline...")

    pipeline = PipelinePokemon()

    print(f"Pipeline carregado em: {pipeline.dispositivo}")

    with Image.open(caminho_imagem) as imagem:
        resultados = pipeline.processar(imagem)

    print()
    print(f"Resultados: {len(resultados)}")

    for indice, resultado in enumerate(resultados, start=1):
        print()
        print(f"Pokemon {indice}")
        print(f"Nome: {resultado['pokemon']}")
        print(f"Confianca: {resultado['confianca']:.2f}%")
        print(f"Box: {resultado['box']}")
        print(f"Fallback: {resultado['fallback']}")


if __name__ == "__main__":
    main()