import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from preparar_dados import carregar_amostras


ROOT_DIR = Path(__file__).resolve().parent.parent.parent

COREL_PATH = (
    ROOT_DIR
    / "datasets"
    / "naopokemons"
    / "Corel-5k"
)

SEED = 42
BATCH_SIZE = 32

EXTENSOES_ACEITAS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}

random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# TRANSFORMACOES
# ============================================================

transform_treino_binario = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        10
    ),

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


transform_avaliacao_binario = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# DATASET BINARIO
# ============================================================

class DatasetBinario(Dataset):
    def __init__(
        self,
        imagens_pokemon,
        imagens_nao_pokemon,
        transform=None,
    ):
        self.transform = transform

        self.amostras = []

        # Classe 1 = Pokemon Gen 1
        for caminho in imagens_pokemon:
            self.amostras.append(
                {
                    "caminho": caminho,
                    "classe": 1,
                }
            )

        # Classe 0 = Nao-Pokemon Gen 1
        for caminho in imagens_nao_pokemon:
            self.amostras.append(
                {
                    "caminho": caminho,
                    "classe": 0,
                }
            )

        random.shuffle(
            self.amostras
        )


    def __len__(self):
        return len(
            self.amostras
        )


    def __getitem__(
        self,
        indice,
    ):
        amostra = self.amostras[
            indice
        ]

        caminho = amostra[
            "caminho"
        ]

        with Image.open(
            caminho
        ) as imagem:
            imagem = imagem.convert(
                "RGB"
            )

        if self.transform:
            imagem = self.transform(
                imagem
            )

        classe = amostra[
            "classe"
        ]

        return imagem, classe


# ============================================================
# COREL 5K
# ============================================================

def carregar_imagens_corel():
    if not COREL_PATH.exists():
        raise FileNotFoundError(
            "Pasta Corel-5k não encontrada: "
            f"{COREL_PATH}"
        )

    imagens = [
        arquivo
        for arquivo in COREL_PATH.iterdir()
        if (
            arquivo.is_file()
            and arquivo.suffix.lower()
            in EXTENSOES_ACEITAS
        )
    ]

    imagens = sorted(
        imagens
    )

    if not imagens:
        raise RuntimeError(
            "Nenhuma imagem válida encontrada "
            "no Corel-5k."
        )

    return imagens


def selecionar_imagens_corel(
    imagens_corel,
    quantidade_treino,
    quantidade_validacao,
    quantidade_teste,
):
    quantidade_necessaria = (
        quantidade_treino
        + quantidade_validacao
        + quantidade_teste
    )

    if len(imagens_corel) < quantidade_necessaria:
        raise RuntimeError(
            "Não existem imagens Corel suficientes. "
            f"Necessárias: {quantidade_necessaria}. "
            f"Disponíveis: {len(imagens_corel)}."
        )

    imagens = list(
        imagens_corel
    )

    gerador = random.Random(
        SEED
    )

    gerador.shuffle(
        imagens
    )

    inicio_validacao = (
        quantidade_treino
    )

    inicio_teste = (
        quantidade_treino
        + quantidade_validacao
    )

    fim_teste = (
        inicio_teste
        + quantidade_teste
    )

    corel_treino = imagens[
        :inicio_validacao
    ]

    corel_validacao = imagens[
        inicio_validacao:
        inicio_teste
    ]

    corel_teste = imagens[
        inicio_teste:
        fim_teste
    ]

    return (
        corel_treino,
        corel_validacao,
        corel_teste,
    )


# ============================================================
# PREPARACAO DOS DADOS
# ============================================================

def preparar_dados_binarios():
    print(
        "Preparando imagens de Pokemon..."
    )

    (
        classes,
        class_to_idx,
        amostras_treino,
        amostras_validacao,
        amostras_teste,
    ) = carregar_amostras()


    imagens_pokemon_treino = [
        caminho
        for caminho, _ in amostras_treino
    ]

    imagens_pokemon_validacao = [
        caminho
        for caminho, _ in amostras_validacao
    ]

    imagens_pokemon_teste = [
        caminho
        for caminho, _ in amostras_teste
    ]


    quantidade_treino = len(
        imagens_pokemon_treino
    )

    quantidade_validacao = len(
        imagens_pokemon_validacao
    )

    quantidade_teste = len(
        imagens_pokemon_teste
    )


    print(
        "Preparando Corel-5k..."
    )

    imagens_corel = carregar_imagens_corel()


    (
        corel_treino,
        corel_validacao,
        corel_teste,
    ) = selecionar_imagens_corel(
        imagens_corel,
        quantidade_treino,
        quantidade_validacao,
        quantidade_teste,
    )


    print()
    print(
        f"Corel-5k disponível: "
        f"{len(imagens_corel)}"
    )

    print()
    print(
        f"Pokemon treino: "
        f"{quantidade_treino}"
    )

    print(
        f"Corel treino: "
        f"{len(corel_treino)}"
    )

    print()
    print(
        f"Pokemon validacao: "
        f"{quantidade_validacao}"
    )

    print(
        f"Corel validacao: "
        f"{len(corel_validacao)}"
    )

    print()
    print(
        f"Pokemon teste: "
        f"{quantidade_teste}"
    )

    print(
        f"Corel teste: "
        f"{len(corel_teste)}"
    )


    # ========================================================
    # DATASETS
    # ========================================================

    dataset_treino = DatasetBinario(
        imagens_pokemon=imagens_pokemon_treino,
        imagens_nao_pokemon=corel_treino,
        transform=transform_treino_binario,
    )


    dataset_validacao = DatasetBinario(
        imagens_pokemon=imagens_pokemon_validacao,
        imagens_nao_pokemon=corel_validacao,
        transform=transform_avaliacao_binario,
    )


    dataset_teste = DatasetBinario(
        imagens_pokemon=imagens_pokemon_teste,
        imagens_nao_pokemon=corel_teste,
        transform=transform_avaliacao_binario,
    )


    # ========================================================
    # DATALOADERS
    # ========================================================

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


    return (
        loader_treino,
        loader_validacao,
        loader_teste,
    )


# ============================================================
# TESTE
# ============================================================

def testar_dados_binarios():
    dispositivo = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print("=" * 60)
    print("DATASET BINARIO V2 BALANCEADO")
    print("=" * 60)


    print()
    print(
        f"Dispositivo: "
        f"{dispositivo}"
    )


    (
        loader_treino,
        loader_validacao,
        loader_teste,
    ) = preparar_dados_binarios()


    print()
    print("=" * 60)
    print("QUANTIDADES FINAIS")
    print("=" * 60)


    print()
    print(
        f"Amostras de treino: "
        f"{len(loader_treino.dataset)}"
    )

    print(
        f"Amostras de validacao: "
        f"{len(loader_validacao.dataset)}"
    )

    print(
        f"Amostras de teste: "
        f"{len(loader_teste.dataset)}"
    )


    print()
    print("=" * 60)
    print("TESTANDO BATCH")
    print("=" * 60)


    imagens, rotulos = next(
        iter(
            loader_treino
        )
    )


    quantidade_nao_pokemon = (
        rotulos == 0
    ).sum().item()

    quantidade_pokemon = (
        rotulos == 1
    ).sum().item()


    print()
    print(
        f"Formato das imagens: "
        f"{imagens.shape}"
    )

    print(
        f"Formato dos rotulos: "
        f"{rotulos.shape}"
    )


    print()
    print(
        f"Nao-Pokemon no batch: "
        f"{quantidade_nao_pokemon}"
    )

    print(
        f"Pokemon no batch: "
        f"{quantidade_pokemon}"
    )


    imagens = imagens.to(
        dispositivo
    )

    rotulos = rotulos.to(
        dispositivo
    )


    print()
    print(
        f"Imagens em: "
        f"{imagens.device}"
    )

    print(
        f"Rotulos em: "
        f"{rotulos.device}"
    )


    print()
    print("=" * 60)
    print("DATASET BINARIO BALANCEADO FUNCIONANDO")
    print("=" * 60)


if __name__ == "__main__":
    testar_dados_binarios()