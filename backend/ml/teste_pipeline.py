import sys
from pathlib import Path

from PIL import Image, ImageDraw

from backend.ml.pipeline import PipelinePokemon


ROOT_DIR = Path(__file__).resolve().parent.parent.parent

OUTPUT_PATH = ROOT_DIR / "saidas"

COR_CAIXA = "red"
COR_TEXTO = "white"
ESPESSURA_CAIXA = 4


def desenhar_resultados(imagem, resultados):
    imagem = imagem.convert("RGB")

    desenho = ImageDraw.Draw(imagem)

    for resultado in resultados:
        x1, y1, x2, y2 = resultado["box"]

        desenho.rectangle(
            (x1, y1, x2, y2),
            outline=COR_CAIXA,
            width=ESPESSURA_CAIXA,
        )

        texto = (
            f"{resultado['pokemon']} "
            f"{resultado['confianca']:.1f}%"
        )

        caixa_texto = desenho.textbbox(
            (x1, y1),
            texto,
        )

        desenho.rectangle(
            caixa_texto,
            fill=COR_CAIXA,
        )

        desenho.text(
            (x1, y1),
            texto,
            fill=COR_TEXTO,
        )

    return imagem


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
        imagem.load()

    resultados = pipeline.processar(imagem)

    print()
    print(f"Resultados: {len(resultados)}")

    for indice, resultado in enumerate(resultados, start=1):
        confianca_detector = resultado["confianca_detector"]

        print()
        print(f"Pokemon {indice}")
        print(f"Nome: {resultado['pokemon']}")
        print(f"Confianca: {resultado['confianca']:.2f}%")

        if confianca_detector is not None:
            print(f"Confianca do detector: {confianca_detector:.2f}%")

        print(f"Box: {resultado['box']}")
        print(f"Fallback: {resultado['fallback']}")

    OUTPUT_PATH.mkdir(exist_ok=True)

    caminho_saida = (
        OUTPUT_PATH
        / f"{caminho_imagem.stem}_pipeline.jpg"
    )

    desenhar_resultados(
        imagem,
        resultados,
    ).save(caminho_saida)

    print()
    print(f"Imagem com as caixas salva em: {caminho_saida}")


if __name__ == "__main__":
    main()
