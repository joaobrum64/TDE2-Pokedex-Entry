import json
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = ROOT_DIR / "datasets" / "deteccao"
BATCH_SIZE = 4


class PokemonDetectionDataset(Dataset):
    def __init__(self, split, dataset_path=DATASET_PATH):
        self.split = split

        self.images_path = dataset_path / split / "images"
        self.labels_path = dataset_path / split / "labels"

        if not self.images_path.exists():
            raise FileNotFoundError(
                f"Pasta de imagens não encontrada: {self.images_path}"
            )

        if not self.labels_path.exists():
            raise FileNotFoundError(
                f"Pasta de labels não encontrada: {self.labels_path}"
            )

        self.labels = sorted(
            self.labels_path.glob("*.json")
        )

        if len(self.labels) == 0:
            raise RuntimeError(
                f"Nenhum label encontrado em: {self.labels_path}"
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, indice):
        caminho_label = self.labels[indice]

        with open(
            caminho_label,
            mode="r",
            encoding="utf-8",
        ) as arquivo:
            dados = json.load(arquivo)

        caminho_imagem = (
            self.images_path / dados["image"]
        )

        if not caminho_imagem.exists():
            raise FileNotFoundError(
                f"Imagem não encontrada: {caminho_imagem}"
            )

        with Image.open(caminho_imagem) as imagem_original:
            imagem = imagem_original.convert("RGB")

        imagem = F.to_tensor(imagem)

        caixas = []
        labels = []

        objetos = dados.get(
            "objects",
            [],
        )

        for objeto in objetos:
            bbox = objeto["bbox"]

            x1 = float(bbox["x1"])
            y1 = float(bbox["y1"])
            x2 = float(bbox["x2"])
            y2 = float(bbox["y2"])

            if x2 <= x1 or y2 <= y1:
                continue

            caixas.append(
                [
                    x1,
                    y1,
                    x2,
                    y2,
                ]
            )

            labels.append(1)

        if len(caixas) > 0:
            caixas_tensor = torch.tensor(
                caixas,
                dtype=torch.float32,
            )

            labels_tensor = torch.tensor(
                labels,
                dtype=torch.int64,
            )

            larguras = (
                caixas_tensor[:, 2]
                - caixas_tensor[:, 0]
            )

            alturas = (
                caixas_tensor[:, 3]
                - caixas_tensor[:, 1]
            )

            areas = larguras * alturas

        else:
            caixas_tensor = torch.zeros(
                (0, 4),
                dtype=torch.float32,
            )

            labels_tensor = torch.zeros(
                (0,),
                dtype=torch.int64,
            )

            areas = torch.zeros(
                (0,),
                dtype=torch.float32,
            )

        image_id = torch.tensor(
            [indice],
            dtype=torch.int64,
        )

        iscrowd = torch.zeros(
            (caixas_tensor.shape[0],),
            dtype=torch.int64,
        )

        target = {
            "boxes": caixas_tensor,
            "labels": labels_tensor,
            "image_id": image_id,
            "area": areas,
            "iscrowd": iscrowd,
        }

        return imagem, target


def collate_fn(batch):
    imagens = []
    targets = []

    for imagem, target in batch:
        imagens.append(imagem)
        targets.append(target)

    return imagens, targets


def criar_dataloaders():
    dataset_treino = PokemonDetectionDataset(
        "train"
    )

    dataset_validacao = PokemonDetectionDataset(
        "val"
    )

    dataset_teste = PokemonDetectionDataset(
        "test"
    )

    loader_treino = DataLoader(
        dataset_treino,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    loader_validacao = DataLoader(
        dataset_validacao,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    loader_teste = DataLoader(
        dataset_teste,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    return (
        loader_treino,
        loader_validacao,
        loader_teste,
    )


def testar_dataset():
    dispositivo = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("DATASET DE DETECCAO")
    print("=" * 60)
    print()

    print(f"Dispositivo: {dispositivo}")

    (
        loader_treino,
        loader_validacao,
        loader_teste,
    ) = criar_dataloaders()

    print()
    print("=" * 60)
    print("QUANTIDADES")
    print("=" * 60)
    print()

    print(
        f"Treino: {len(loader_treino.dataset)} imagens"
    )

    print(
        f"Validacao: {len(loader_validacao.dataset)} imagens"
    )

    print(
        f"Teste: {len(loader_teste.dataset)} imagens"
    )

    print()
    print("=" * 60)
    print("TESTANDO BATCH")
    print("=" * 60)

    imagens, targets = next(
        iter(loader_treino)
    )

    print()

    print(
        f"Quantidade de imagens no batch: {len(imagens)}"
    )

    for indice in range(len(imagens)):
        imagem = imagens[indice]
        target = targets[indice]

        quantidade_pokemons = (
            target["boxes"].shape[0]
        )

        print()
        print(f"Imagem {indice + 1}")

        print(
            f"Formato: {imagem.shape}"
        )

        print(
            f"Boxes: {target['boxes'].shape}"
        )

        print(
            f"Labels: {target['labels'].tolist()}"
        )

        print(
            f"Quantidade de Pokemon: {quantidade_pokemons}"
        )

        if quantidade_pokemons > 0:
            print(
                f"Primeira box: "
                f"{target['boxes'][0].tolist()}"
            )

            print(
                f"Area da primeira box: "
                f"{target['area'][0].item():.2f}"
            )

    print()
    print("=" * 60)
    print("ENVIANDO BATCH PARA GPU")
    print("=" * 60)

    imagens_gpu = []

    for imagem in imagens:
        imagem_gpu = imagem.to(dispositivo)
        imagens_gpu.append(imagem_gpu)

    targets_gpu = []

    for target in targets:
        target_gpu = {}

        for chave, valor in target.items():
            target_gpu[chave] = valor.to(dispositivo)

        targets_gpu.append(target_gpu)

    print()

    print(
        f"Primeira imagem em: "
        f"{imagens_gpu[0].device}"
    )

    print(
        f"Boxes da primeira imagem em: "
        f"{targets_gpu[0]['boxes'].device}"
    )

    print(
        f"Labels da primeira imagem em: "
        f"{targets_gpu[0]['labels'].device}"
    )

    print()
    print("=" * 60)
    print("DATASET DE DETECCAO FUNCIONANDO")
    print("=" * 60)


if __name__ == "__main__":
    testar_dataset()