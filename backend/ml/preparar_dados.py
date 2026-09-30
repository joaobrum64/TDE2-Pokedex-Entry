import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = ROOT_DIR / "datasets" / "pokemons"
DIVISAO_PATH = ROOT_DIR / "datasets" / "divisao_pokemons.json"

EXTENSOES_ACEITAS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
}

SEED = 42
BATCH_SIZE = 32

PORCENTAGEM_TREINO = 0.70
PORCENTAGEM_VALIDACAO = 0.15
PORCENTAGEM_TESTE = 0.15


random.seed(SEED)
torch.manual_seed(SEED)


transform_treino = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1,
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


transform_avaliacao = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class PokemonDataset(Dataset):
    def __init__(
        self,
        amostras,
        transform=None
    ):
        self.amostras = amostras
        self.transform = transform

    def __len__(self):
        return len(self.amostras)

    def __getitem__(self, indice):
        caminho_imagem, classe = self.amostras[indice]

        with Image.open(caminho_imagem) as imagem:
            imagem = imagem.convert("RGB")

        if self.transform:
            imagem = self.transform(imagem)

        return imagem, classe


def listar_imagens(nome_classe):
    pasta_classe = DATASET_PATH / nome_classe

    return [
        arquivo
        for arquivo in pasta_classe.iterdir()
        if (
            arquivo.is_file()
            and arquivo.suffix.lower() in EXTENSOES_ACEITAS
        )
    ]


def salvar_divisao(treino, validacao, teste):
    divisao = {}

    for nome_conjunto, amostras in (
        ("treino", treino),
        ("validacao", validacao),
        ("teste", teste),
    ):
        divisao[nome_conjunto] = [
            caminho.relative_to(DATASET_PATH).as_posix()
            for caminho, _ in amostras
        ]

    with open(
        DIVISAO_PATH,
        mode="w",
        encoding="utf-8",
    ) as arquivo:
        json.dump(
            divisao,
            arquivo,
            ensure_ascii=False,
            indent=0,
        )


def carregar_divisao(classes, class_to_idx):
    with open(
        DIVISAO_PATH,
        mode="r",
        encoding="utf-8",
    ) as arquivo:
        divisao = json.load(arquivo)

    conjuntos = []
    caminhos_na_divisao = set()
    quantidade_ausentes = 0

    for nome_conjunto in ("treino", "validacao", "teste"):
        amostras = []

        for caminho_relativo in divisao[nome_conjunto]:
            caminho = DATASET_PATH / caminho_relativo

            caminhos_na_divisao.add(caminho)

            # Imagens removidas do dataset depois da divisão
            # simplesmente saem do conjunto em que estavam.
            if not caminho.exists():
                quantidade_ausentes += 1
                continue

            amostras.append(
                (caminho, class_to_idx[caminho.parent.name])
            )

        conjuntos.append(amostras)

    quantidade_fora = sum(
        1
        for nome_classe in classes
        for caminho in listar_imagens(nome_classe)
        if caminho not in caminhos_na_divisao
    )

    if quantidade_ausentes > 0:
        print(
            f"Aviso: {quantidade_ausentes} imagens da divisão "
            "não existem mais no dataset e foram ignoradas."
        )

    if quantidade_fora > 0:
        print(
            f"Aviso: {quantidade_fora} imagens do dataset não "
            f"estão em {DIVISAO_PATH.name} e foram ignoradas."
        )

    return conjuntos


def carregar_amostras():
    classes = sorted([
        pasta.name
        for pasta in DATASET_PATH.iterdir()
        if pasta.is_dir()
    ])

    class_to_idx = {
        nome: indice
        for indice, nome in enumerate(classes)
    }

    # A divisão fica gravada em arquivo para que todos os modelos
    # sejam treinados e avaliados exatamente nas mesmas imagens.
    # Ela só é sorteada quando o arquivo ainda não existe.
    if DIVISAO_PATH.exists():
        treino, validacao, teste = carregar_divisao(
            classes,
            class_to_idx,
        )

    else:
        treino, validacao, teste = sortear_divisao(
            classes,
            class_to_idx,
        )

        salvar_divisao(
            treino,
            validacao,
            teste,
        )

    return (
        classes,
        class_to_idx,
        treino,
        validacao,
        teste,
    )


def sortear_divisao(classes, class_to_idx):
    treino = []
    validacao = []
    teste = []

    for nome_classe in classes:
        imagens = listar_imagens(nome_classe)

        random.shuffle(imagens)

        quantidade = len(imagens)

        quantidade_treino = int(
            quantidade * PORCENTAGEM_TREINO
        )

        quantidade_validacao = int(
            quantidade * PORCENTAGEM_VALIDACAO
        )

        indice_validacao = quantidade_treino

        indice_teste = (
            quantidade_treino
            + quantidade_validacao
        )

        classe = class_to_idx[nome_classe]

        treino.extend([
            (imagem, classe)
            for imagem in imagens[:indice_validacao]
        ])

        validacao.extend([
            (imagem, classe)
            for imagem in imagens[
                indice_validacao:indice_teste
            ]
        ])

        teste.extend([
            (imagem, classe)
            for imagem in imagens[indice_teste:]
        ])

    random.shuffle(treino)
    random.shuffle(validacao)
    random.shuffle(teste)

    return (
        treino,
        validacao,
        teste,
    )


def testar_dataloaders():
    dispositivo = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    (
        classes,
        class_to_idx,
        amostras_treino,
        amostras_validacao,
        amostras_teste,
    ) = carregar_amostras()

    dataset_treino = PokemonDataset(
        amostras_treino,
        transform=transform_treino,
    )

    dataset_validacao = PokemonDataset(
        amostras_validacao,
        transform=transform_avaliacao,
    )

    dataset_teste = PokemonDataset(
        amostras_teste,
        transform=transform_avaliacao,
    )

    loader_treino = DataLoader(
        dataset_treino,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    loader_validacao = DataLoader(
        dataset_validacao,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    loader_teste = DataLoader(
        dataset_teste,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("=" * 60)
    print("PREPARAÇÃO DOS DADOS")
    print("=" * 60)

    print()
    print(f"Dispositivo: {dispositivo}")

    print()
    print(f"Quantidade de classes: {len(classes)}")

    print()
    print(f"Imagens de treino: {len(dataset_treino)}")
    print(
        f"Imagens de validação: "
        f"{len(dataset_validacao)}"
    )
    print(f"Imagens de teste: {len(dataset_teste)}")

    print()
    print("Primeiras 10 classes:")

    for indice, classe in enumerate(classes[:10]):
        print(
            f"{indice}: {classe}"
        )

    print()
    print("Últimas 10 classes:")

    inicio = len(classes) - 10

    for indice, classe in enumerate(
        classes[-10:],
        start=inicio
    ):
        print(
            f"{indice}: {classe}"
        )

    print()
    print("=" * 60)
    print("TESTANDO UM BATCH")
    print("=" * 60)

    imagens, rotulos = next(
        iter(loader_treino)
    )

    print()
    print(
        f"Formato das imagens antes da GPU: "
        f"{imagens.shape}"
    )

    print(
        f"Formato dos rótulos: "
        f"{rotulos.shape}"
    )

    imagens = imagens.to(dispositivo)
    rotulos = rotulos.to(dispositivo)

    print()
    print(
        f"Dispositivo das imagens: "
        f"{imagens.device}"
    )

    print(
        f"Dispositivo dos rótulos: "
        f"{rotulos.device}"
    )

    print()
    print(
        f"Tipo dos dados das imagens: "
        f"{imagens.dtype}"
    )

    print()
    print(
        f"Menor valor do batch: "
        f"{imagens.min().item():.4f}"
    )

    print(
        f"Maior valor do batch: "
        f"{imagens.max().item():.4f}"
    )

    print()
    print("=" * 60)
    print("DATALOADERS FUNCIONANDO")
    print("=" * 60)

    return (
        classes,
        class_to_idx,
        loader_treino,
        loader_validacao,
        loader_teste,
    )


if __name__ == "__main__":
    testar_dataloaders()