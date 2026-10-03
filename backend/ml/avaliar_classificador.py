import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from backend.ml.modelo import criar_arquitetura
from backend.ml.preparar_dados import (
    PokemonDataset,
    carregar_amostras,
    transform_avaliacao,
)


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODELOS_PATH = ROOT_DIR / "modelos"

MODELO_PADRAO = "pokemon_resnet18_v2.pth"

# Referência do v2 no teste original, que um modelo novo precisa
# superar.
ACCURACY_V2_TESTE_ORIGINAL = 78.92

QUANTIDADE_PIORES = 10
QUANTIDADE_CONFUSOES = 10


def carregar_modelo(caminho, dispositivo):
    checkpoint = torch.load(
        caminho,
        map_location=dispositivo,
        weights_only=False,
    )

    modelo = criar_arquitetura(
        checkpoint["architecture"],
        checkpoint["num_classes"],
    )

    modelo.load_state_dict(checkpoint["model_state_dict"])

    return modelo.to(dispositivo).eval(), checkpoint


def prever(modelo, amostras, dispositivo):
    loader = DataLoader(
        PokemonDataset(amostras, transform=transform_avaliacao),
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    previsoes_top5 = []

    with torch.no_grad():
        for imagens, _ in loader:
            saidas = modelo(imagens.to(dispositivo))
            previsoes_top5.append(saidas.topk(5, dim=1).indices.cpu())

    return torch.cat(previsoes_top5)


def resumir(nome, amostras, top5, classes):
    rotulos = torch.tensor([classe for _, classe in amostras])

    top1 = top5[:, 0]

    accuracy = (top1 == rotulos).float().mean().item() * 100
    accuracy_top5 = (
        (top5 == rotulos[:, None]).any(dim=1).float().mean().item() * 100
    )

    print()
    print(f"{nome}: {len(amostras)} imagens")
    print(f"  Top-1: {accuracy:.2f}%")
    print(f"  Top-5: {accuracy_top5:.2f}%")

    return accuracy, rotulos, top1


def main():
    nome_arquivo = sys.argv[1] if len(sys.argv) > 1 else MODELO_PADRAO

    caminho = MODELOS_PATH / nome_arquivo

    dispositivo = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print(f"AVALIACAO DO CLASSIFICADOR: {nome_arquivo}")
    print("=" * 60)

    modelo, checkpoint = carregar_modelo(caminho, dispositivo)

    print()
    print(f"Arquitetura: {checkpoint['architecture']}")
    print(f"Epoca salva: {checkpoint.get('epoch', '-')}")
    print(
        "Melhor validacao no treino: "
        f"{checkpoint['validation_accuracy']:.2f}%"
    )

    classes, _, _, _, teste_original = carregar_amostras(
        aplicar_exclusoes=False
    )
    _, _, _, _, teste_limpo = carregar_amostras()

    if checkpoint["classes"] != classes:
        raise ValueError(
            "A ordem das classes do modelo nao corresponde ao dataset."
        )

    accuracy_original, _, _ = resumir(
        "Teste original",
        teste_original,
        prever(modelo, teste_original, dispositivo),
        classes,
    )

    _, rotulos, top1 = resumir(
        "Teste limpo",
        teste_limpo,
        prever(modelo, teste_limpo, dispositivo),
        classes,
    )

    total_por_classe = Counter(rotulos.tolist())
    acertos_por_classe = Counter(
        rotulo
        for rotulo, previsto in zip(rotulos.tolist(), top1.tolist())
        if rotulo == previsto
    )

    print()
    print("Piores classes no teste limpo:")

    for classe in sorted(
        total_por_classe,
        key=lambda c: acertos_por_classe[c] / total_por_classe[c],
    )[:QUANTIDADE_PIORES]:
        print(
            f"  {acertos_por_classe[classe] / total_por_classe[classe] * 100:6.2f}%"
            f"  {classes[classe]} "
            f"({acertos_por_classe[classe]}/{total_por_classe[classe]})"
        )

    confusoes = Counter(
        f"{classes[rotulo]} -> {classes[previsto]}"
        for rotulo, previsto in zip(rotulos.tolist(), top1.tolist())
        if rotulo != previsto
    )

    print()
    print("Confusoes mais frequentes no teste limpo:")

    for confusao, quantidade in confusoes.most_common(QUANTIDADE_CONFUSOES):
        print(f"  {quantidade:3d}  {confusao}")

    diferenca = accuracy_original - ACCURACY_V2_TESTE_ORIGINAL

    print()
    print(
        f"Teste original: {accuracy_original:.2f}% contra "
        f"{ACCURACY_V2_TESTE_ORIGINAL:.2f}% do v2 "
        f"({diferenca:+.2f} pontos)"
    )


if __name__ == "__main__":
    main()
