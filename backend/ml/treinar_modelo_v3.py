import math
import sys
import time
from collections import Counter
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from backend.ml.modelo import criar_arquitetura
from backend.ml.preparar_dados import carregar_amostras


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODELOS_PATH = ROOT_DIR / "modelos"
ARQUITETURA_PADRAO = "resnet18"
NUM_EPOCAS = 30
PACIENCIA_EARLY_STOPPING = 7
EPOCAS_AQUECIMENTO = 1
BATCH_SIZE = 64
TAXA_APRENDIZADO = 0.0003
WEIGHT_DECAY = 0.05
SUAVIZACAO_ROTULOS = 0.1
TAMANHO_CACHE = 256
MEDIA = [0.485, 0.456, 0.406]
DESVIO = [0.229, 0.224, 0.225]
SEED = 42


transform_treino = transforms.Compose([
    transforms.RandomResizedCrop(
        224,
        scale=(0.6, 1.0),
        ratio=(0.8, 1.25),
    ),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.03,
    ),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEDIA, std=DESVIO),
    transforms.RandomErasing(p=0.25),
])

transform_validacao = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEDIA, std=DESVIO),
])


class DatasetEmMemoria(Dataset):
    def __init__(self, amostras, transform):
        self.transform = transform
        self.rotulos = [classe for _, classe in amostras]
        self.imagens = []

        for indice, (caminho, _) in enumerate(amostras, start=1):
            with Image.open(caminho) as imagem:
                self.imagens.append(
                    imagem.convert("RGB").resize(
                        (TAMANHO_CACHE, TAMANHO_CACHE),
                        Image.Resampling.BILINEAR,
                    )
                )

            if indice % 2000 == 0 or indice == len(amostras):
                print(f"  {indice}/{len(amostras)}")

    def __len__(self):
        return len(self.imagens)

    def __getitem__(self, indice):
        return (
            self.transform(self.imagens[indice]),
            self.rotulos[indice],
        )


def criar_sampler(rotulos):

    contagem = Counter(rotulos)

    pesos = [1.0 / contagem[rotulo] for rotulo in rotulos]

    return WeightedRandomSampler(
        pesos,
        num_samples=len(rotulos),
        replacement=True,
        generator=torch.Generator().manual_seed(SEED),
    )


def criar_scheduler(otimizador, passos_por_epoca):

    passos_aquecimento = EPOCAS_AQUECIMENTO * passos_por_epoca
    passos_totais = NUM_EPOCAS * passos_por_epoca

    def fator(passo):
        if passo < passos_aquecimento:
            return (passo + 1) / passos_aquecimento

        progresso = (
            (passo - passos_aquecimento)
            / max(1, passos_totais - passos_aquecimento)
        )

        return 0.5 * (1 + math.cos(math.pi * progresso))

    return torch.optim.lr_scheduler.LambdaLR(otimizador, fator)


def treinar_uma_epoca(
    modelo,
    loader,
    criterio,
    otimizador,
    scheduler,
    dispositivo,
):
    modelo.train()

    loss_total = 0.0
    corretos = 0
    total = 0

    for imagens, rotulos in loader:
        imagens = imagens.to(dispositivo, non_blocking=True)
        rotulos = rotulos.to(dispositivo, non_blocking=True)

        otimizador.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=dispositivo.type,
            dtype=torch.bfloat16,
            enabled=dispositivo.type == "cuda",
        ):
            saidas = modelo(imagens)
            loss = criterio(saidas, rotulos)

        loss.backward()
        otimizador.step()
        scheduler.step()

        loss_total += loss.item() * imagens.size(0)
        corretos += (saidas.argmax(dim=1) == rotulos).sum().item()
        total += rotulos.size(0)

    return loss_total / total, corretos / total * 100


def validar(modelo, loader, dispositivo):
    modelo.eval()

    corretos = 0
    total = 0

    with torch.no_grad(), torch.autocast(
        device_type=dispositivo.type,
        dtype=torch.bfloat16,
        enabled=dispositivo.type == "cuda",
    ):
        for imagens, rotulos in loader:
            saidas = modelo(imagens.to(dispositivo))

            corretos += (
                saidas.argmax(dim=1).cpu() == rotulos
            ).sum().item()
            total += rotulos.size(0)

    return corretos / total * 100


def main():
    arquitetura = (
        sys.argv[1] if len(sys.argv) > 1 else ARQUITETURA_PADRAO
    )

    continuar = len(sys.argv) > 2 and sys.argv[2] == "continuar"

    modelo_path = MODELOS_PATH / f"pokemon_{arquitetura}_v3.pth"
    estado_path = MODELOS_PATH / f"pokemon_{arquitetura}_v3_estado.pth"

    torch.manual_seed(SEED)

    dispositivo = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print(f"TREINAMENTO DO CLASSIFICADOR V3 ({arquitetura})")
    print("=" * 60)

    print()
    print(f"Dispositivo: {dispositivo}")

    classes, _, treino, validacao, _ = carregar_amostras()

    print()
    print(f"Classes: {len(classes)}")
    print(f"Treino: {len(treino)} | Validacao: {len(validacao)}")

    print()
    print("Carregando imagens de treino na memoria...")
    dataset_treino = DatasetEmMemoria(treino, transform_treino)

    print("Carregando imagens de validacao na memoria...")
    dataset_validacao = DatasetEmMemoria(validacao, transform_validacao)

    loader_treino = DataLoader(
        dataset_treino,
        batch_size=BATCH_SIZE,
        sampler=criar_sampler(dataset_treino.rotulos),
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    loader_validacao = DataLoader(
        dataset_validacao,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    modelo = criar_arquitetura(
        arquitetura,
        len(classes),
        pre_treinado=True,
    ).to(dispositivo)

    criterio = nn.CrossEntropyLoss(label_smoothing=SUAVIZACAO_ROTULOS)

    otimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=TAXA_APRENDIZADO,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = criar_scheduler(otimizador, len(loader_treino))

    melhor_accuracy = 0.0
    epocas_sem_melhoria = 0
    primeira_epoca = 1

    if continuar:
        estado = torch.load(
            estado_path,
            map_location=dispositivo,
            weights_only=False,
        )

        modelo.load_state_dict(estado["modelo"])
        otimizador.load_state_dict(estado["otimizador"])
        scheduler.load_state_dict(estado["scheduler"])
        torch.set_rng_state(estado["rng"].cpu())

        melhor_accuracy = estado["melhor_accuracy"]
        epocas_sem_melhoria = estado["epocas_sem_melhoria"]
        primeira_epoca = estado["epoca"] + 1

        print()
        print(
            f"Continuando da epoca {primeira_epoca} "
            f"(melhor validacao ate agora: {melhor_accuracy:.2f}%)"
        )

    print()
    print("=" * 60)
    print("INICIO DO TREINAMENTO")
    print("=" * 60)

    for epoca in range(primeira_epoca, NUM_EPOCAS + 1):
        inicio = time.time()

        loss_treino, accuracy_treino = treinar_uma_epoca(
            modelo,
            loader_treino,
            criterio,
            otimizador,
            scheduler,
            dispositivo,
        )

        accuracy_validacao = validar(
            modelo,
            loader_validacao,
            dispositivo,
        )

        print(
            f"Epoca {epoca:2d}/{NUM_EPOCAS} | "
            f"treino loss {loss_treino:.3f} acc {accuracy_treino:5.2f}% | "
            f"validacao {accuracy_validacao:5.2f}% | "
            f"{time.time() - inicio:.0f}s"
        )

        if accuracy_validacao > melhor_accuracy:
            melhor_accuracy = accuracy_validacao
            epocas_sem_melhoria = 0

            torch.save(
                {
                    "model_state_dict": modelo.state_dict(),
                    "classes": classes,
                    "num_classes": len(classes),
                    "architecture": arquitetura,
                    "validation_accuracy": melhor_accuracy,
                    "epoch": epoca,
                },
                modelo_path,
            )

            print("  melhor modelo salvo")

        else:
            epocas_sem_melhoria += 1

        torch.save(
            {
                "modelo": modelo.state_dict(),
                "otimizador": otimizador.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": torch.get_rng_state(),
                "epoca": epoca,
                "melhor_accuracy": melhor_accuracy,
                "epocas_sem_melhoria": epocas_sem_melhoria,
            },
            estado_path,
        )

        if epocas_sem_melhoria >= PACIENCIA_EARLY_STOPPING:
            print()
            print("Early stopping ativado.")
            break

    print()
    print("=" * 60)
    print("TREINAMENTO CONCLUIDO")
    print("=" * 60)

    print()
    print(f"Melhor accuracy de validacao (limpa): {melhor_accuracy:.2f}%")
    print(f"Modelo salvo em: {modelo_path}")
    print()
    print("Avalie com:")
    print(f"python -m backend.ml.avaliar_classificador {modelo_path.name}")


if __name__ == "__main__":
    main()
