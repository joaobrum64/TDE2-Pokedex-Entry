import json
import random
import shutil
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
    / "deteccao"
)


EXTENSOES_ACEITAS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


SEED = 42

LARGURA_IMAGEM = 640
ALTURA_IMAGEM = 640


# Quantidade de cenas
QUANTIDADE_TREINO = 2400
QUANTIDADE_VALIDACAO = 300
QUANTIDADE_TESTE = 300


# 20% das imagens não terão nenhum Pokémon.
PROBABILIDADE_SEM_POKEMON = 0.20

MIN_POKEMONS = 1
MAX_POKEMONS = 4

TAMANHO_MIN_SPRITE = 90
TAMANHO_MAX_SPRITE = 230


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
        raise RuntimeError(
            "Era esperado encontrar 151 sprites, "
            f"mas foram encontrados {len(sprites)}."
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
            "Nenhuma imagem de fundo foi encontrada."
        )

    return fundos


def preparar_fundo(caminho):
    with Image.open(caminho) as imagem:
        imagem = imagem.convert("RGB")

    largura_original, altura_original = imagem.size

    proporcao = max(
        LARGURA_IMAGEM / largura_original,
        ALTURA_IMAGEM / altura_original,
    )

    nova_largura = int(
        largura_original * proporcao
    )

    nova_altura = int(
        altura_original * proporcao
    )

    imagem = imagem.resize(
        (
            nova_largura,
            nova_altura,
        ),
        Image.Resampling.LANCZOS,
    )

    esquerda = (
        nova_largura - LARGURA_IMAGEM
    ) // 2

    topo = (
        nova_altura - ALTURA_IMAGEM
    ) // 2

    imagem = imagem.crop(
        (
            esquerda,
            topo,
            esquerda + LARGURA_IMAGEM,
            topo + ALTURA_IMAGEM,
        )
    )

    return imagem


def preparar_sprite(
    caminho,
    tamanho,
):
    with Image.open(caminho) as imagem:
        sprite = imagem.convert("RGBA")

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
            largura_original * escala
        ),
    )

    altura_nova = max(
        1,
        int(
            altura_original * escala
        ),
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
    for _ in range(100):

        x1 = random.randint(
            0,
            LARGURA_IMAGEM - largura_sprite,
        )

        y1 = random.randint(
            0,
            ALTURA_IMAGEM - altura_sprite,
        )

        x2 = (
            x1 + largura_sprite
        )

        y2 = (
            y1 + altura_sprite
        )

        nova_caixa = (
            x1,
            y1,
            x2,
            y2,
        )

        sobrepoe = any(
            caixas_se_sobrepoem(
                nova_caixa,
                caixa_existente,
            )
            for caixa_existente
            in caixas_existentes
        )

        if not sobrepoe:
            return nova_caixa

    return None


def definir_quantidade_pokemons():
    if (
        random.random()
        < PROBABILIDADE_SEM_POKEMON
    ):
        return 0

    return random.randint(
        MIN_POKEMONS,
        MAX_POKEMONS,
    )


def gerar_cena(
    indice,
    split,
    sprites,
    fundos,
    images_path,
    labels_path,
):
    caminho_fundo = random.choice(
        fundos
    )

    imagem = preparar_fundo(
        caminho_fundo
    )

    quantidade_pokemons = (
        definir_quantidade_pokemons()
    )

    objetos = []
    caixas_existentes = []

    if quantidade_pokemons > 0:

        sprites_escolhidos = random.sample(
            sprites,
            quantidade_pokemons,
        )

        for (
            pokedex_id,
            caminho_sprite,
        ) in sprites_escolhidos:

            tamanho = random.randint(
                TAMANHO_MIN_SPRITE,
                TAMANHO_MAX_SPRITE,
            )

            sprite = preparar_sprite(
                caminho_sprite,
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

            objetos.append(
                {
                    "class_id": 1,
                    "class_name": "pokemon",
                    "pokedex_id": pokedex_id,
                    "bbox": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    },
                }
            )

    nome_base = (
        f"{split}_{indice:05d}"
    )

    nome_imagem = (
        f"{nome_base}.jpg"
    )

    caminho_imagem = (
        images_path
        / nome_imagem
    )

    caminho_label = (
        labels_path
        / f"{nome_base}.json"
    )

    imagem.save(
        caminho_imagem,
        quality=95,
    )

    dados = {
        "image": nome_imagem,
        "width": LARGURA_IMAGEM,
        "height": ALTURA_IMAGEM,
        "objects": objetos,
    }

    with open(
        caminho_label,
        mode="w",
        encoding="utf-8",
    ) as arquivo:

        json.dump(
            dados,
            arquivo,
            ensure_ascii=False,
            indent=2,
        )

    return len(
        objetos
    )


def preparar_pastas():
    if OUTPUT_PATH.exists():

        print(
            "Removendo dataset de detecção anterior..."
        )

        shutil.rmtree(
            OUTPUT_PATH
        )

    for split in (
        "train",
        "val",
        "test",
    ):
        (
            OUTPUT_PATH
            / split
            / "images"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            OUTPUT_PATH
            / split
            / "labels"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )


def gerar_split(
    nome_split,
    quantidade,
    sprites,
    fundos,
):
    images_path = (
        OUTPUT_PATH
        / nome_split
        / "images"
    )

    labels_path = (
        OUTPUT_PATH
        / nome_split
        / "labels"
    )

    total_objetos = 0

    quantidade_vazias = 0

    print()
    print(
        f"Gerando {nome_split}: "
        f"{quantidade} cenas"
    )

    for indice in range(
        1,
        quantidade + 1,
    ):
        quantidade_objetos = gerar_cena(
            indice,
            nome_split,
            sprites,
            fundos,
            images_path,
            labels_path,
        )

        total_objetos += (
            quantidade_objetos
        )

        if quantidade_objetos == 0:
            quantidade_vazias += 1

        if (
            indice % 100 == 0
            or indice == quantidade
        ):
            print(
                f"{indice}/"
                f"{quantidade}"
            )

    print(
        f"{nome_split} - "
        f"objetos: {total_objetos}"
    )

    print(
        f"{nome_split} - "
        f"imagens sem Pokémon: "
        f"{quantidade_vazias}"
    )


def main():
    print("=" * 60)
    print("GERADOR DO DATASET DE DETECCAO")
    print("=" * 60)

    if not SPRITES_PATH.exists():
        raise FileNotFoundError(
            "Pasta de sprites não encontrada: "
            f"{SPRITES_PATH}"
        )

    if not FUNDOS_PATH.exists():
        raise FileNotFoundError(
            "Pasta de fundos não encontrada: "
            f"{FUNDOS_PATH}"
        )

    preparar_pastas()

    sprites = carregar_sprites()

    fundos = carregar_fundos()

    print()
    print(
        f"Sprites: {len(sprites)}"
    )

    print(
        f"Fundos: {len(fundos)}"
    )

    gerar_split(
        "train",
        QUANTIDADE_TREINO,
        sprites,
        fundos,
    )

    gerar_split(
        "val",
        QUANTIDADE_VALIDACAO,
        sprites,
        fundos,
    )

    gerar_split(
        "test",
        QUANTIDADE_TESTE,
        sprites,
        fundos,
    )

    print()
    print("=" * 60)
    print("DATASET GERADO")
    print("=" * 60)

    print()
    print(
        f"Pasta: {OUTPUT_PATH}"
    )

    print()
    print(
        f"Treino: "
        f"{QUANTIDADE_TREINO}"
    )

    print(
        f"Validacao: "
        f"{QUANTIDADE_VALIDACAO}"
    )

    print(
        f"Teste: "
        f"{QUANTIDADE_TESTE}"
    )


if __name__ == "__main__":
    main()