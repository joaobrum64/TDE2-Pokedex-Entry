import json
import random
from pathlib import Path

from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent.parent.parent

SPRITES_PATH = (
    ROOT_DIR
    / "datasets"
    / "sprites"
)

FUNDOS_PATH = (
    ROOT_DIR
    / "datasets"
    / "naopokemons"
    / "Corel-5k"
)

OUTPUT_PATH = (
    ROOT_DIR
    / "datasets"
    / "deteccao_teste"
)

IMAGES_PATH = (
    OUTPUT_PATH
    / "images"
)

LABELS_PATH = (
    OUTPUT_PATH
    / "labels"
)


EXTENSOES_ACEITAS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


SEED = 42

QUANTIDADE_IMAGENS = 20

LARGURA_IMAGEM = 640
ALTURA_IMAGEM = 640

MIN_POKEMONS = 1
MAX_POKEMONS = 4

TAMANHO_MIN_SPRITE = 100
TAMANHO_MAX_SPRITE = 220


random.seed(SEED)


def carregar_sprites():
    sprites = []

    for arquivo in SPRITES_PATH.iterdir():

        if (
            arquivo.is_file()
            and arquivo.suffix.lower()
            in EXTENSOES_ACEITAS
        ):

            try:
                pokedex_id = int(
                    arquivo.stem
                )

                sprites.append(
                    (
                        pokedex_id,
                        arquivo,
                    )
                )

            except ValueError:
                continue

    sprites.sort(
        key=lambda item: item[0]
    )

    if len(sprites) != 151:
        print(
            f"ATENÇÃO: foram encontrados "
            f"{len(sprites)} sprites."
        )

    return sprites


def carregar_fundos():
    fundos = [
        arquivo
        for arquivo in FUNDOS_PATH.iterdir()
        if (
            arquivo.is_file()
            and arquivo.suffix.lower()
            in EXTENSOES_ACEITAS
        )
    ]

    if not fundos:
        raise RuntimeError(
            "Nenhuma imagem de fundo encontrada."
        )

    return fundos


def preparar_fundo(caminho):
    with Image.open(caminho) as imagem:
        fundo = imagem.convert(
            "RGB"
        )

    fundo = fundo.resize(
        (
            LARGURA_IMAGEM,
            ALTURA_IMAGEM,
        )
    )

    return fundo


def preparar_sprite(
    caminho,
    tamanho,
):
    with Image.open(caminho) as imagem:
        sprite = imagem.convert(
            "RGBA"
        )

    largura_original, altura_original = (
        sprite.size
    )

    maior_dimensao = max(
        largura_original,
        altura_original,
    )

    escala = (
        tamanho
        / maior_dimensao
    )

    largura_nova = max(
        1,
        int(
            largura_original
            * escala
        )
    )

    altura_nova = max(
        1,
        int(
            altura_original
            * escala
        )
    )

    sprite = sprite.resize(
        (
            largura_nova,
            altura_nova,
        ),
        Image.Resampling.LANCZOS,
    )

    return sprite


def caixas_se_sobrepoem(
    caixa_a,
    caixa_b,
):
    ax1, ay1, ax2, ay2 = caixa_a
    bx1, by1, bx2, by2 = caixa_b

    return not (
        ax2 <= bx1
        or bx2 <= ax1
        or ay2 <= by1
        or by2 <= ay1
    )


def encontrar_posicao(
    largura_sprite,
    altura_sprite,
    caixas_existentes,
):
    max_tentativas = 100

    for _ in range(
        max_tentativas
    ):

        x1 = random.randint(
            0,
            LARGURA_IMAGEM
            - largura_sprite,
        )

        y1 = random.randint(
            0,
            ALTURA_IMAGEM
            - altura_sprite,
        )

        x2 = (
            x1
            + largura_sprite
        )

        y2 = (
            y1
            + altura_sprite
        )

        nova_caixa = (
            x1,
            y1,
            x2,
            y2,
        )

        sobreposicao = any(
            caixas_se_sobrepoem(
                nova_caixa,
                caixa,
            )
            for caixa
            in caixas_existentes
        )

        if not sobreposicao:
            return nova_caixa

    return None


def gerar_imagem(
    indice,
    sprites,
    fundos,
):
    fundo_path = random.choice(
        fundos
    )

    imagem = preparar_fundo(
        fundo_path
    )

    quantidade_pokemons = (
        random.randint(
            MIN_POKEMONS,
            MAX_POKEMONS,
        )
    )

    sprites_escolhidos = (
        random.sample(
            sprites,
            quantidade_pokemons,
        )
    )

    objetos = []
    caixas_existentes = []

    for (
        pokedex_id,
        sprite_path,
    ) in sprites_escolhidos:

        tamanho = random.randint(
            TAMANHO_MIN_SPRITE,
            TAMANHO_MAX_SPRITE,
        )

        sprite = preparar_sprite(
            sprite_path,
            tamanho,
        )

        largura_sprite, altura_sprite = (
            sprite.size
        )

        caixa = encontrar_posicao(
            largura_sprite,
            altura_sprite,
            caixas_existentes,
        )

        if caixa is None:
            continue

        x1, y1, x2, y2 = caixa

        imagem.paste(
            sprite,
            (
                x1,
                y1,
            ),
            sprite,
        )

        caixas_existentes.append(
            caixa
        )

        objetos.append({
            "pokedex_id": pokedex_id,

            "sprite": sprite_path.name,

            "bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            },
        })

    nome_base = (
        f"cena_{indice:04d}"
    )

    caminho_imagem = (
        IMAGES_PATH
        / f"{nome_base}.jpg"
    )

    caminho_label = (
        LABELS_PATH
        / f"{nome_base}.json"
    )

    imagem.save(
        caminho_imagem,
        quality=95,
    )

    label = {
        "image": caminho_imagem.name,

        "width": LARGURA_IMAGEM,

        "height": ALTURA_IMAGEM,

        "background": fundo_path.name,

        "objects": objetos,
    }

    with open(
        caminho_label,
        mode="w",
        encoding="utf-8",
    ) as arquivo:
        json.dump(
            label,
            arquivo,
            ensure_ascii=False,
            indent=2,
        )

    return (
        caminho_imagem,
        caminho_label,
        len(objetos),
    )


def main():
    print("=" * 60)
    print("GERADOR DE DATASET DE DETECÇÃO")
    print("=" * 60)

    if not SPRITES_PATH.exists():
        raise FileNotFoundError(
            f"Pasta de sprites não encontrada: "
            f"{SPRITES_PATH}"
        )

    if not FUNDOS_PATH.exists():
        raise FileNotFoundError(
            f"Pasta Corel não encontrada: "
            f"{FUNDOS_PATH}"
        )

    IMAGES_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    LABELS_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    sprites = carregar_sprites()

    fundos = carregar_fundos()

    print()
    print(
        f"Sprites encontrados: "
        f"{len(sprites)}"
    )

    print(
        f"Fundos encontrados: "
        f"{len(fundos)}"
    )

    print()
    print(
        f"Gerando "
        f"{QUANTIDADE_IMAGENS} cenas..."
    )

    total_objetos = 0

    for indice in range(
        1,
        QUANTIDADE_IMAGENS + 1,
    ):
        (
            caminho_imagem,
            caminho_label,
            quantidade_objetos,
        ) = gerar_imagem(
            indice,
            sprites,
            fundos,
        )

        total_objetos += (
            quantidade_objetos
        )

        print(
            f"{indice:02d}/"
            f"{QUANTIDADE_IMAGENS} "
            f"- "
            f"{caminho_imagem.name} "
            f"- "
            f"{quantidade_objetos} Pokémon"
        )

    print()
    print("=" * 60)
    print("GERAÇÃO CONCLUÍDA")
    print("=" * 60)

    print()
    print(
        f"Cenas criadas: "
        f"{QUANTIDADE_IMAGENS}"
    )

    print(
        f"Objetos inseridos: "
        f"{total_objetos}"
    )

    print()
    print(
        f"Imagens: "
        f"{IMAGES_PATH}"
    )

    print(
        f"Labels: "
        f"{LABELS_PATH}"
    )


if __name__ == "__main__":
    main()