import math
import sys
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from backend.ml.preparar_dados_deteccao import (
    PokemonDetectionDataset,
    collate_fn,
)


ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATASET_PATH = ROOT_DIR / "datasets" / "deteccao_v2"
MODELOS_PATH = ROOT_DIR / "modelos"
MODELO_PATH = MODELOS_PATH / "pokemon_object_detector_v2.pth"
ESTADO_PATH = MODELOS_PATH / "pokemon_object_detector_v2_estado.pth"
NUM_EPOCAS = 8
EPOCAS_AQUECIMENTO = 1
BATCH_SIZE = 4
TAXA_APRENDIZADO = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
PROBABILIDADE_ESPELHAR = 0.5
SEED = 42


def criar_modelo():
    modelo = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )

    modelo.roi_heads.box_predictor = FastRCNNPredictor(
        modelo.roi_heads.box_predictor.cls_score.in_features,
        2,
    )

    return modelo


def espelhar(imagens, targets):

    for indice, (imagem, target) in enumerate(zip(imagens, targets)):
        if torch.rand(1).item() >= PROBABILIDADE_ESPELHAR:
            continue

        largura = imagem.shape[-1]

        imagens[indice] = imagem.flip(-1)

        caixas = target["boxes"].clone()

        if len(caixas) > 0:
            caixas[:, [0, 2]] = largura - target["boxes"][:, [2, 0]]

        target["boxes"] = caixas

    return imagens, targets


def mover(imagens, targets, dispositivo):
    return (
        [imagem.to(dispositivo) for imagem in imagens],
        [
            {chave: valor.to(dispositivo) for chave, valor in target.items()}
            for target in targets
        ],
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


def treinar_uma_epoca(modelo, loader, otimizador, scheduler, escalador, dispositivo):
    modelo.train()

    loss_total = 0.0

    for indice, (imagens, targets) in enumerate(loader, start=1):
        imagens, targets = espelhar(list(imagens), list(targets))
        imagens, targets = mover(imagens, targets, dispositivo)

        otimizador.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=dispositivo.type,
            enabled=dispositivo.type == "cuda",
        ):
            loss = sum(modelo(imagens, targets).values())

        escalador.scale(loss).backward()
        escalador.step(otimizador)
        escalador.update()
        scheduler.step()

        loss_total += loss.item()

        if indice % 250 == 0:
            print(f"  batch {indice}/{len(loader)} - loss {loss.item():.4f}")

    return loss_total / len(loader)


def calcular_loss_validacao(modelo, loader, dispositivo):

    modelo.train()

    loss_total = 0.0

    with torch.no_grad(), torch.autocast(
        device_type=dispositivo.type,
        enabled=dispositivo.type == "cuda",
    ):
        for imagens, targets in loader:
            imagens, targets = mover(list(imagens), list(targets), dispositivo)

            loss_total += sum(modelo(imagens, targets).values()).item()

    return loss_total / len(loader)


def main():
    continuar = len(sys.argv) > 1 and sys.argv[1] == "continuar"

    torch.manual_seed(SEED)

    dispositivo = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("TREINAMENTO DO DETECTOR DE OBJETOS V2")
    print("=" * 60)

    print()
    print(f"Dispositivo: {dispositivo}")

    loader_treino = DataLoader(
        PokemonDetectionDataset("train", DATASET_PATH),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(SEED),
    )

    loader_validacao = DataLoader(
        PokemonDetectionDataset("val", DATASET_PATH),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    print(
        f"Treino: {len(loader_treino.dataset)} cenas | "
        f"Validacao: {len(loader_validacao.dataset)} cenas"
    )

    modelo = criar_modelo().to(dispositivo)

    otimizador = torch.optim.SGD(
        [parametro for parametro in modelo.parameters() if parametro.requires_grad],
        lr=TAXA_APRENDIZADO,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = criar_scheduler(otimizador, len(loader_treino))

    escalador = torch.amp.GradScaler(enabled=dispositivo.type == "cuda")

    melhor_loss = float("inf")
    primeira_epoca = 1

    if continuar:
        estado = torch.load(ESTADO_PATH, map_location=dispositivo, weights_only=False)

        modelo.load_state_dict(estado["modelo"])
        otimizador.load_state_dict(estado["otimizador"])
        scheduler.load_state_dict(estado["scheduler"])
        escalador.load_state_dict(estado["escalador"])
        torch.set_rng_state(estado["rng"].cpu())

        melhor_loss = estado["melhor_loss"]
        primeira_epoca = estado["epoca"] + 1

        print()
        print(
            f"Continuando da epoca {primeira_epoca} "
            f"(melhor loss de validacao ate agora: {melhor_loss:.4f})"
        )

    for epoca in range(primeira_epoca, NUM_EPOCAS + 1):
        inicio = time.time()

        print()
        print(f"Epoca {epoca}/{NUM_EPOCAS}")

        loss_treino = treinar_uma_epoca(
            modelo,
            loader_treino,
            otimizador,
            scheduler,
            escalador,
            dispositivo,
        )

        loss_validacao = calcular_loss_validacao(
            modelo,
            loader_validacao,
            dispositivo,
        )

        print(
            f"Epoca {epoca}/{NUM_EPOCAS} | treino {loss_treino:.4f} | "
            f"validacao {loss_validacao:.4f} | {time.time() - inicio:.0f}s"
        )

        if loss_validacao < melhor_loss:
            melhor_loss = loss_validacao

            # Pesos em meia precisão: o arquivo cai de ~158 MB para
            # ~80 MB e cabe no limite de 100 MB do GitHub, sem mudar
            # os resultados (conferido com avaliar_pipeline).
            torch.save(
                {
                    "model_state_dict": {
                        chave: (
                            valor.half()
                            if valor.is_floating_point()
                            else valor
                        )
                        for chave, valor in modelo.state_dict().items()
                    },
                    "precisao": "float16",
                    "architecture": "fasterrcnn_resnet50_fpn",
                    "num_classes": 2,
                    "classes": ["background", "pokemon"],
                    "validation_loss": melhor_loss,
                    "epoch": epoca,
                },
                MODELO_PATH,
            )

            print("  melhor modelo salvo")

        torch.save(
            {
                "modelo": modelo.state_dict(),
                "otimizador": otimizador.state_dict(),
                "scheduler": scheduler.state_dict(),
                "escalador": escalador.state_dict(),
                "rng": torch.get_rng_state(),
                "epoca": epoca,
                "melhor_loss": melhor_loss,
            },
            ESTADO_PATH,
        )

    print()
    print("=" * 60)
    print("TREINAMENTO CONCLUIDO")
    print("=" * 60)
    print(f"Melhor loss de validacao: {melhor_loss:.4f}")
    print(f"Modelo salvo em: {MODELO_PATH}")


if __name__ == "__main__":
    main()
