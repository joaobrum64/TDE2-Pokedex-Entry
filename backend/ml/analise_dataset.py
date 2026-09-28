import csv
from collections import Counter
from pathlib import Path
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent.parent.parent

DATASET_PATH = ROOT_DIR / "datasets" / "pokemons"
CSV_PATH = ROOT_DIR / "datasets" / "pokemon_gen1.csv"

EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
}


def normalizar_nome(nome):
    return (
        nome
        .lower()
        .replace(" ", "")
        .replace(".", "")
        .replace("'", "")
        .replace("-", "")
        .replace("♀", "f")
        .replace("♂", "m")
    )


def carregar_nomes_esperados():
    nomes = {}

    with open(
        CSV_PATH,
        mode="r",
        encoding="utf-8-sig",
        newline=""
    ) as arquivo:

        reader = csv.DictReader(arquivo)

        for pokemon in reader:
            nome = pokemon["name"]

            nomes[normalizar_nome(nome)] = nome

    return nomes


def analisar_dataset():
    if not DATASET_PATH.exists():
        print("ERRO: pasta do dataset não encontrada.")
        print(f"Caminho esperado: {DATASET_PATH}")
        return

    nomes_esperados = carregar_nomes_esperados()

    pastas = sorted(
        pasta
        for pasta in DATASET_PATH.iterdir()
        if pasta.is_dir()
    )

    pastas_pokemon = []

    pastas_desconhecidas = []

    for pasta in pastas:
        nome_normalizado = normalizar_nome(pasta.name)

        if nome_normalizado in nomes_esperados:
            pastas_pokemon.append(pasta)
        else:
            pastas_desconhecidas.append(pasta)

    nomes_encontrados = {
        normalizar_nome(pasta.name)
        for pasta in pastas_pokemon
    }

    pokemons_faltando = []

    for nome_normalizado, nome_original in nomes_esperados.items():
        if nome_normalizado not in nomes_encontrados:
            pokemons_faltando.append(nome_original)

    quantidade_classes = len(pastas_pokemon)

    total_imagens = 0

    imagens_por_classe = {}

    extensoes = Counter()
    dimensoes = Counter()

    arquivos_invalidos = []

    print("=" * 60)
    print("ANÁLISE DO DATASET")
    print("=" * 60)

    for pasta in pastas_pokemon:
        quantidade_imagens = 0

        for arquivo in pasta.iterdir():
            if not arquivo.is_file():
                continue

            extensao = arquivo.suffix.lower()

            if extensao not in EXTENSIONS:
                continue

            extensoes[extensao] += 1

            try:
                with Image.open(arquivo) as imagem:
                    imagem.verify()

                with Image.open(arquivo) as imagem:
                    largura, altura = imagem.size

                dimensoes[(largura, altura)] += 1

                quantidade_imagens += 1
                total_imagens += 1

            except Exception as erro:
                arquivos_invalidos.append(
                    (arquivo, str(erro))
                )

        imagens_por_classe[pasta.name] = quantidade_imagens

    print()
    print(f"Classes de Pokémon encontradas: {quantidade_classes}")
    print(f"Total de imagens válidas: {total_imagens}")

    if quantidade_classes == 151:
        print("Quantidade de classes: OK")
    else:
        print("ATENÇÃO: eram esperadas 151 classes.")

    print()
    print("=" * 60)
    print("POKÉMON FALTANDO")
    print("=" * 60)

    if not pokemons_faltando:
        print("Nenhum Pokémon faltando.")
    else:
        for nome in pokemons_faltando:
            print(nome)

    print()
    print("=" * 60)
    print("PASTAS DESCONHECIDAS")
    print("=" * 60)

    if not pastas_desconhecidas:
        print("Nenhuma pasta desconhecida.")
    else:
        for pasta in pastas_desconhecidas:
            print(pasta.name)

    print()
    print("=" * 60)
    print("IMAGENS POR CLASSE")
    print("=" * 60)

    for nome, quantidade in sorted(
        imagens_por_classe.items()
    ):
        print(f"{nome}: {quantidade}")

    if imagens_por_classe:
        menor_classe = min(
            imagens_por_classe,
            key=imagens_por_classe.get
        )

        maior_classe = max(
            imagens_por_classe,
            key=imagens_por_classe.get
        )

        print()
        print("=" * 60)
        print("BALANCEAMENTO")
        print("=" * 60)

        print(
            f"Menor classe: "
            f"{menor_classe} "
            f"({imagens_por_classe[menor_classe]} imagens)"
        )

        print(
            f"Maior classe: "
            f"{maior_classe} "
            f"({imagens_por_classe[maior_classe]} imagens)"
        )

    print()
    print("=" * 60)
    print("EXTENSÕES")
    print("=" * 60)

    for extensao, quantidade in extensoes.most_common():
        print(f"{extensao}: {quantidade}")

    print()
    print("=" * 60)
    print("DIMENSÕES MAIS COMUNS")
    print("=" * 60)

    for dimensao, quantidade in dimensoes.most_common(10):
        largura, altura = dimensao

        print(
            f"{largura}x{altura}: "
            f"{quantidade} imagens"
        )

    print()
    print(
        f"Quantidade total de resoluções diferentes: "
        f"{len(dimensoes)}"
    )

    print()
    print("=" * 60)
    print("ARQUIVOS INVÁLIDOS")
    print("=" * 60)

    if not arquivos_invalidos:
        print("Nenhum arquivo de imagem inválido encontrado.")
    else:
        print(
            f"Foram encontrados "
            f"{len(arquivos_invalidos)} arquivos inválidos:"
        )

        for arquivo, erro in arquivos_invalidos:
            print()
            print(f"Arquivo: {arquivo}")
            print(f"Erro: {erro}")

    print()
    print("=" * 60)
    print("FIM DA ANÁLISE")
    print("=" * 60)


if __name__ == "__main__":
    analisar_dataset()