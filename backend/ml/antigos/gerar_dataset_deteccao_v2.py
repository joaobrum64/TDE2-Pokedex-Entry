import json
import random
import shutil
from pathlib import Path
from PIL import Image, ImageChops, ImageFilter
from backend.ml.calibrar_limiares import (
    carregar_corel_avaliacao,
    sortear_corel_calibracao,
)
from backend.ml.gerar_dataset_deteccao import (
    ALTURA_IMAGEM,
    LARGURA_IMAGEM,
    FUNDOS_PATH,
    caixas_se_sobrepoem,
    carregar_sprites,
    preparar_fundo,
    preparar_sprite,
)
from backend.ml.preparar_dados import carregar_amostras


ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_PATH = ROOT_DIR / "datasets" / "deteccao_v2"
SEED = 42
QUANTIDADES = {
    "train": 4000,
    "val": 400,
    "test": 200,
}
TIPOS_DE_CENA = [
    ("vazia", 0.15),
    ("sprites", 0.15),
    ("recortes", 0.25),
    ("fotos", 0.30),
    ("foto_inteira", 0.15),
]
MIN_OBJETOS = 1
MAX_OBJETOS = 4
TAMANHO_MIN_OBJETO = 120
TAMANHO_MAX_OBJETO = 360
PROBABILIDADE_DISTRATOR = 0.5
TOLERANCIA_FUNDO = 24
FRACAO_MINIMA_BORDA_LISA = 0.95
AREA_MINIMA_RECORTE = 0.05
AREA_MAXIMA_RECORTE = 0.90


def numero_base_corel(caminho):

    return caminho.stem.split("_")[0]


def carregar_fundos_permitidos():

    reservadas = {
        numero_base_corel(caminho)
        for caminho in carregar_corel_avaliacao() + sortear_corel_calibracao()
    }

    return sorted(
        arquivo
        for arquivo in FUNDOS_PATH.iterdir()
        if arquivo.is_file()
        and numero_base_corel(arquivo) not in reservadas
    )


def extrair_recorte(caminho):

    with Image.open(caminho) as imagem:
        imagem = imagem.convert("RGB")

    imagem.thumbnail((512, 512))

    largura, altura = imagem.size
    cor_fundo = imagem.getpixel((0, 0))

    diferenca = ImageChops.difference(
        imagem,
        Image.new("RGB", imagem.size, cor_fundo),
    ).convert("L")

    mascara = diferenca.point(
        lambda valor: 255 if valor > TOLERANCIA_FUNDO else 0
    )

    borda = (
        [mascara.getpixel((x, 0)) for x in range(largura)]
        + [mascara.getpixel((x, altura - 1)) for x in range(largura)]
        + [mascara.getpixel((0, y)) for y in range(altura)]
        + [mascara.getpixel((largura - 1, y)) for y in range(altura)]
    )

    if borda.count(0) / len(borda) < FRACAO_MINIMA_BORDA_LISA:
        return None

    mascara = mascara.filter(ImageFilter.MaxFilter(5)).filter(
        ImageFilter.MinFilter(5)
    )

    caixa = mascara.getbbox()

    if caixa is None:
        return None

    area = (caixa[2] - caixa[0]) * (caixa[3] - caixa[1])

    if not (
        AREA_MINIMA_RECORTE
        <= area / (largura * altura)
        <= AREA_MAXIMA_RECORTE
    ):
        return None

    recorte = imagem.convert("RGBA")
    recorte.putalpha(mascara)

    return recorte.crop(caixa)


def redimensionar(imagem, tamanho):
    escala = tamanho / max(imagem.size)

    return imagem.resize(
        (
            max(1, int(imagem.width * escala)),
            max(1, int(imagem.height * escala)),
        ),
        Image.Resampling.LANCZOS,
    )


def encontrar_posicao(sorteio, largura, altura, caixas):
    for _ in range(100):
        x1 = sorteio.randint(0, LARGURA_IMAGEM - largura)
        y1 = sorteio.randint(0, ALTURA_IMAGEM - altura)

        caixa = (x1, y1, x1 + largura, y1 + altura)

        if not any(
            caixas_se_sobrepoem(caixa, existente)
            for existente in caixas
        ):
            return caixa

    return None


def colar_objetos(sorteio, fundo, objetos):

    caixas_ocupadas = []
    caixas_pokemon = []

    for imagem, marcado in objetos:
        imagem = redimensionar(
            imagem,
            sorteio.randint(TAMANHO_MIN_OBJETO, TAMANHO_MAX_OBJETO),
        )

        caixa = encontrar_posicao(
            sorteio,
            imagem.width,
            imagem.height,
            caixas_ocupadas,
        )

        if caixa is None:
            continue

        mascara = imagem if imagem.mode == "RGBA" else None

        fundo.paste(imagem, caixa[:2], mascara)

        caixas_ocupadas.append(caixa)

        if marcado:
            caixas_pokemon.append(caixa)

    return caixas_pokemon


def abrir_rgb(caminho):
    with Image.open(caminho) as imagem:
        return imagem.convert("RGB")


def distratores(sorteio, fundos):
    if sorteio.random() >= PROBABILIDADE_DISTRATOR:
        return []

    return [
        (abrir_rgb(sorteio.choice(fundos)), False)
        for _ in range(sorteio.randint(1, 2))
    ]


def gerar_cena(sorteio, tipo, fontes, fundos):
    fundo = preparar_fundo(sorteio.choice(fundos))

    quantidade = sorteio.randint(MIN_OBJETOS, MAX_OBJETOS)

    if tipo == "vazia":
        caixas = colar_objetos(
            sorteio,
            fundo,
            distratores(sorteio, fundos),
        )

    elif tipo == "sprites":
        caixas = colar_objetos(
            sorteio,
            fundo,
            [
                (preparar_sprite(caminho, TAMANHO_MAX_OBJETO), True)
                for _, caminho in sorteio.sample(fontes["sprites"], quantidade)
            ],
        )

    elif tipo == "recortes":
        objetos = []

        while len(objetos) < quantidade:
            recorte = extrair_recorte(sorteio.choice(fontes["recortes"]))

            if recorte is not None:
                objetos.append((recorte, True))

        caixas = colar_objetos(sorteio, fundo, objetos)

    elif tipo == "fotos":

        objetos = [
            (abrir_rgb(caminho), True)
            for caminho in sorteio.sample(fontes["fotos"], quantidade)
        ] + distratores(sorteio, fundos)

        sorteio.shuffle(objetos)

        caixas = colar_objetos(sorteio, fundo, objetos)

    else:

        fundo = abrir_rgb(sorteio.choice(fontes["fotos"])).resize(
            (LARGURA_IMAGEM, ALTURA_IMAGEM),
            Image.Resampling.LANCZOS,
        )

        caixas = [(0, 0, LARGURA_IMAGEM, ALTURA_IMAGEM)]

    return fundo, caixas


def preparar_fontes(amostras):
    caminhos = [caminho for caminho, _ in amostras]

    print(f"  Procurando imagens com fundo liso em {len(caminhos)}...")

    recortes = [
        caminho
        for caminho in caminhos
        if extrair_recorte(caminho) is not None
    ]

    print(f"  Imagens que viram recorte: {len(recortes)}")

    return {
        "sprites": carregar_sprites(),
        "recortes": recortes,
        "fotos": caminhos,
    }


def gerar_split(nome, quantidade, fontes, fundos, sorteio):
    images_path = OUTPUT_PATH / nome / "images"
    labels_path = OUTPUT_PATH / nome / "labels"

    images_path.mkdir(parents=True)
    labels_path.mkdir(parents=True)

    tipos = [tipo for tipo, _ in TIPOS_DE_CENA]
    pesos = [peso for _, peso in TIPOS_DE_CENA]

    contagem = {tipo: 0 for tipo in tipos}
    total_objetos = 0

    for indice in range(1, quantidade + 1):
        tipo = sorteio.choices(tipos, weights=pesos)[0]

        imagem, caixas = gerar_cena(sorteio, tipo, fontes, fundos)

        nome_base = f"{nome}_{indice:05d}"

        imagem.save(images_path / f"{nome_base}.jpg", quality=92)

        with open(
            labels_path / f"{nome_base}.json",
            mode="w",
            encoding="utf-8",
        ) as arquivo:
            json.dump(
                {
                    "image": f"{nome_base}.jpg",
                    "width": LARGURA_IMAGEM,
                    "height": ALTURA_IMAGEM,
                    "scene_type": tipo,
                    "objects": [
                        {
                            "class_id": 1,
                            "class_name": "pokemon",
                            "bbox": {
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2,
                            },
                        }
                        for x1, y1, x2, y2 in caixas
                    ],
                },
                arquivo,
                ensure_ascii=False,
            )

        contagem[tipo] += 1
        total_objetos += len(caixas)

        if indice % 500 == 0 or indice == quantidade:
            print(f"  {nome}: {indice}/{quantidade}")

    print(
        f"  {nome} - cenas por tipo: {contagem} | "
        f"Pokémon marcados: {total_objetos}"
    )


def main():
    print("=" * 60)
    print("GERADOR DO DATASET DE DETECCAO V2")
    print("=" * 60)

    sorteio = random.Random(SEED)

    _, _, treino, validacao, teste = carregar_amostras()

    fundos = carregar_fundos_permitidos()

    print()
    print(f"Fundos do Corel permitidos: {len(fundos)}")

    if OUTPUT_PATH.exists():
        shutil.rmtree(OUTPUT_PATH)

    for nome, amostras in (
        ("train", treino),
        ("val", validacao),
        ("test", teste),
    ):
        print()
        print(f"Preparando {nome}...")

        fontes = preparar_fontes(amostras)

        gerar_split(
            nome,
            QUANTIDADES[nome],
            fontes,
            fundos,
            sorteio,
        )

    print()
    print(f"Dataset gerado em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
