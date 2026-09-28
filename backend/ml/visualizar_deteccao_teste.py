import json
from pathlib import Path

from PIL import Image, ImageDraw


ROOT_DIR = Path(__file__).resolve().parent.parent.parent

DATASET_PATH = ROOT_DIR / "datasets" / "deteccao_teste"
IMAGES_PATH = DATASET_PATH / "images"
LABELS_PATH = DATASET_PATH / "labels"
PREVIEW_PATH = DATASET_PATH / "preview"

COR_CAIXA = "red"
COR_TEXTO = "white"
COR_FUNDO_TEXTO = "red"

ESPESSURA_CAIXA = 3
MARGEM_TEXTO = 4


def carregar_label(caminho_label):
    with open(
        caminho_label,
        mode="r",
        encoding="utf-8",
    ) as arquivo:
        return json.load(arquivo)


def desenhar_caixa(
    desenho,
    bbox,
    texto,
):
    x1 = int(bbox["x1"])
    y1 = int(bbox["y1"])
    x2 = int(bbox["x2"])
    y2 = int(bbox["y2"])

    desenho.rectangle(
        (x1, y1, x2, y2),
        outline=COR_CAIXA,
        width=ESPESSURA_CAIXA,
    )

    caixa_texto = desenho.textbbox(
        (0, 0),
        texto,
    )

    largura_texto = (
        caixa_texto[2] - caixa_texto[0]
    )

    altura_texto = (
        caixa_texto[3] - caixa_texto[1]
    )

    largura_fundo = (
        largura_texto
        + MARGEM_TEXTO * 2
    )

    altura_fundo = (
        altura_texto
        + MARGEM_TEXTO * 2
    )

    texto_x = x1

    texto_y = max(
        0,
        y1 - altura_fundo,
    )

    desenho.rectangle(
        (
            texto_x,
            texto_y,
            texto_x + largura_fundo,
            texto_y + altura_fundo,
        ),
        fill=COR_FUNDO_TEXTO,
    )

    desenho.text(
        (
            texto_x + MARGEM_TEXTO,
            texto_y + MARGEM_TEXTO,
        ),
        texto,
        fill=COR_TEXTO,
    )


def gerar_preview(caminho_label):
    dados = carregar_label(
        caminho_label
    )

    nome_imagem = dados["image"]

    caminho_imagem = (
        IMAGES_PATH / nome_imagem
    )

    if not caminho_imagem.exists():
        print(
            f"Imagem não encontrada: "
            f"{caminho_imagem}"
        )

        return False

    with Image.open(
        caminho_imagem
    ) as imagem_original:
        imagem = imagem_original.convert(
            "RGB"
        )

    desenho = ImageDraw.Draw(
        imagem
    )

    objetos = dados.get(
        "objects",
        [],
    )

    for objeto in objetos:
        pokedex_id = objeto[
            "pokedex_id"
        ]

        bbox = objeto[
            "bbox"
        ]

        texto = (
            f"Pokemon #{pokedex_id}"
        )

        desenhar_caixa(
            desenho,
            bbox,
            texto,
        )

    caminho_saida = (
        PREVIEW_PATH / nome_imagem
    )

    imagem.save(
        caminho_saida,
        quality=95,
    )

    return True


def main():
    print("=" * 60)
    print(
        "VISUALIZACAO DAS BOUNDING BOXES"
    )
    print("=" * 60)

    if not IMAGES_PATH.exists():
        raise FileNotFoundError(
            "Pasta de imagens não encontrada: "
            f"{IMAGES_PATH}"
        )

    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            "Pasta de labels não encontrada: "
            f"{LABELS_PATH}"
        )

    PREVIEW_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels = sorted(
        LABELS_PATH.glob("*.json")
    )

    if len(labels) == 0:
        raise RuntimeError(
            "Nenhum arquivo JSON foi encontrado."
        )

    print()
    print(
        f"Labels encontrados: {len(labels)}"
    )

    print()
    print("Gerando previews...")
    print()

    quantidade_gerada = 0

    for indice, caminho_label in enumerate(
        labels,
        start=1,
    ):
        sucesso = gerar_preview(
            caminho_label
        )

        if sucesso:
            quantidade_gerada += 1

        print(
            f"{indice:02d}/"
            f"{len(labels):02d} "
            f"- {caminho_label.name}"
        )

    print()
    print("=" * 60)
    print("PREVIEWS CONCLUIDOS")
    print("=" * 60)

    print()
    print(
        f"Previews gerados: "
        f"{quantidade_gerada}"
    )

    print(
        f"Pasta de saida: "
        f"{PREVIEW_PATH}"
    )


if __name__ == "__main__":
    main()