from statistics import mean, median

import torch
from torchvision.datasets import CIFAR10

from modelo import ModeloPokemon
from preparar_dados import (
    carregar_amostras,
    transform_avaliacao,
    PokemonDataset,
)


QUANTIDADE_CIFAR = 1755

THRESHOLDS_CONFIANCA = [
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
]

THRESHOLDS_MARGEM = [
    5,
    10,
    20,
    30,
    40,
    50,
]


def obter_metricas_saida(saidas):
    probabilidades = torch.softmax(
        saidas,
        dim=1,
    )

    top_probabilidades, _ = torch.topk(
        probabilidades,
        k=2,
        dim=1,
    )

    top1 = top_probabilidades[:, 0] * 100
    top2 = top_probabilidades[:, 1] * 100

    margem = top1 - top2

    return (
        top1.cpu().tolist(),
        top2.cpu().tolist(),
        margem.cpu().tolist(),
    )


def analisar_pokemons(
    modelo_pokemon,
    loader,
):
    top1_total = []
    top2_total = []
    margens_total = []

    modelo = modelo_pokemon.modelo
    dispositivo = modelo_pokemon.dispositivo

    modelo.eval()

    with torch.no_grad():
        for imagens, _ in loader:
            imagens = imagens.to(
                dispositivo
            )

            saidas = modelo(
                imagens
            )

            top1, top2, margens = (
                obter_metricas_saida(saidas)
            )

            top1_total.extend(top1)
            top2_total.extend(top2)
            margens_total.extend(margens)

    return (
        top1_total,
        top2_total,
        margens_total,
    )


def analisar_cifar(
    modelo_pokemon,
    dataset_cifar,
):
    top1_total = []
    top2_total = []
    margens_total = []

    modelo = modelo_pokemon.modelo
    dispositivo = modelo_pokemon.dispositivo

    modelo.eval()

    print()
    print(
        f"Analisando {QUANTIDADE_CIFAR} "
        "imagens do CIFAR-10..."
    )

    with torch.no_grad():
        for indice in range(
            QUANTIDADE_CIFAR
        ):
            imagem, _ = dataset_cifar[
                indice
            ]

            imagem = transform_avaliacao(
                imagem
            )

            imagem = imagem.unsqueeze(0)

            imagem = imagem.to(
                dispositivo
            )

            saidas = modelo(
                imagem
            )

            top1, top2, margens = (
                obter_metricas_saida(saidas)
            )

            top1_total.extend(top1)
            top2_total.extend(top2)
            margens_total.extend(margens)

            if (indice + 1) % 250 == 0:
                print(
                    f"{indice + 1}/"
                    f"{QUANTIDADE_CIFAR}"
                )

    return (
        top1_total,
        top2_total,
        margens_total,
    )


def mostrar_resumo(
    titulo,
    top1,
    top2,
    margens,
):
    print()
    print("=" * 60)
    print(titulo)
    print("=" * 60)

    print()
    print(
        f"Quantidade: {len(top1)}"
    )

    print()
    print("Confiança Top 1:")

    print(
        f"  Média: "
        f"{mean(top1):.2f}%"
    )

    print(
        f"  Mediana: "
        f"{median(top1):.2f}%"
    )

    print(
        f"  Mínima: "
        f"{min(top1):.2f}%"
    )

    print(
        f"  Máxima: "
        f"{max(top1):.2f}%"
    )

    print()
    print("Confiança Top 2:")

    print(
        f"  Média: "
        f"{mean(top2):.2f}%"
    )

    print(
        f"  Mediana: "
        f"{median(top2):.2f}%"
    )

    print()
    print("Margem Top 1 - Top 2:")

    print(
        f"  Média: "
        f"{mean(margens):.2f}"
    )

    print(
        f"  Mediana: "
        f"{median(margens):.2f}"
    )


def testar_thresholds_confianca(
    positivos,
    negativos,
):
    print()
    print("=" * 60)
    print("THRESHOLD DE CONFIANÇA")
    print("=" * 60)

    print()
    print(
        "Regra: confiança >= threshold "
        "é considerada Pokémon."
    )

    print()

    for threshold in THRESHOLDS_CONFIANCA:
        pokemon_aceitos = sum(
            valor >= threshold
            for valor in positivos
        )

        nao_pokemon_rejeitados = sum(
            valor < threshold
            for valor in negativos
        )

        taxa_pokemon = (
            pokemon_aceitos
            / len(positivos)
        ) * 100

        taxa_rejeicao = (
            nao_pokemon_rejeitados
            / len(negativos)
        ) * 100

        accuracy_binaria = (
            pokemon_aceitos
            + nao_pokemon_rejeitados
        ) / (
            len(positivos)
            + len(negativos)
        ) * 100

        print(
            f"Threshold {threshold}%"
        )

        print(
            f"  Pokémon aceitos: "
            f"{taxa_pokemon:.2f}%"
        )

        print(
            f"  Não-Pokémon rejeitados: "
            f"{taxa_rejeicao:.2f}%"
        )

        print(
            f"  Accuracy binária: "
            f"{accuracy_binaria:.2f}%"
        )

        print()


def testar_thresholds_margem(
    positivos,
    negativos,
):
    print()
    print("=" * 60)
    print("THRESHOLD DE MARGEM")
    print("=" * 60)

    print()
    print(
        "Regra: Top1 - Top2 >= threshold "
        "é considerada Pokémon."
    )

    print()

    for threshold in THRESHOLDS_MARGEM:
        pokemon_aceitos = sum(
            valor >= threshold
            for valor in positivos
        )

        nao_pokemon_rejeitados = sum(
            valor < threshold
            for valor in negativos
        )

        taxa_pokemon = (
            pokemon_aceitos
            / len(positivos)
        ) * 100

        taxa_rejeicao = (
            nao_pokemon_rejeitados
            / len(negativos)
        ) * 100

        accuracy_binaria = (
            pokemon_aceitos
            + nao_pokemon_rejeitados
        ) / (
            len(positivos)
            + len(negativos)
        ) * 100

        print(
            f"Margem {threshold}%"
        )

        print(
            f"  Pokémon aceitos: "
            f"{taxa_pokemon:.2f}%"
        )

        print(
            f"  Não-Pokémon rejeitados: "
            f"{taxa_rejeicao:.2f}%"
        )

        print(
            f"  Accuracy binária: "
            f"{accuracy_binaria:.2f}%"
        )

        print()

def testar_thresholds_combinados(
    top1_positivos,
    margem_positivos,
    top1_negativos,
    margem_negativos,
):
    print()
    print("=" * 60)
    print("THRESHOLDS COMBINADOS")
    print("=" * 60)

    thresholds_confianca = [
        20,
        30,
        40,
        50,
        60,
    ]

    thresholds_margem = [
        5,
        10,
        20,
        30,
        40,
    ]

    resultados = []

    for threshold_confianca in thresholds_confianca:
        for threshold_margem in thresholds_margem:

            pokemon_aceitos = sum(
                confianca >= threshold_confianca
                and margem >= threshold_margem
                for confianca, margem in zip(
                    top1_positivos,
                    margem_positivos,
                )
            )

            nao_pokemon_rejeitados = sum(
                not (
                    confianca >= threshold_confianca
                    and margem >= threshold_margem
                )
                for confianca, margem in zip(
                    top1_negativos,
                    margem_negativos,
                )
            )

            taxa_pokemon = (
                pokemon_aceitos
                / len(top1_positivos)
            ) * 100

            taxa_rejeicao = (
                nao_pokemon_rejeitados
                / len(top1_negativos)
            ) * 100

            accuracy = (
                pokemon_aceitos
                + nao_pokemon_rejeitados
            ) / (
                len(top1_positivos)
                + len(top1_negativos)
            ) * 100

            resultados.append({
                "confianca": threshold_confianca,
                "margem": threshold_margem,
                "pokemon_aceitos": taxa_pokemon,
                "nao_pokemon_rejeitados": taxa_rejeicao,
                "accuracy": accuracy,
            })

    resultados.sort(
        key=lambda resultado: resultado["accuracy"],
        reverse=True,
    )

    print()
    print("10 melhores combinações:")
    print()

    for resultado in resultados[:10]:
        print(
            f"Confiança >= {resultado['confianca']}% "
            f"e Margem >= {resultado['margem']}%"
        )

        print(
            f"  Pokémon aceitos: "
            f"{resultado['pokemon_aceitos']:.2f}%"
        )

        print(
            f"  Não-Pokémon rejeitados: "
            f"{resultado['nao_pokemon_rejeitados']:.2f}%"
        )

        print(
            f"  Accuracy binária: "
            f"{resultado['accuracy']:.2f}%"
        )

        print()        


def main():
    print("=" * 60)
    print("ANÁLISE DE REJEIÇÃO")
    print("=" * 60)

    print()
    print("Carregando ResNet18...")

    modelo_pokemon = ModeloPokemon()

    print(
        f"Modelo carregado em: "
        f"{modelo_pokemon.dispositivo}"
    )

    print()
    print(
        "Preparando imagens positivas..."
    )

    (
        classes,
        class_to_idx,
        amostras_treino,
        amostras_validacao,
        amostras_teste,
    ) = carregar_amostras()

    dataset_teste = PokemonDataset(
        amostras_teste,
        transform=transform_avaliacao,
    )

    loader_teste = torch.utils.data.DataLoader(
        dataset_teste,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    print(
        f"Pokémon Gen 1 para análise: "
        f"{len(dataset_teste)}"
    )

    print()
    print("Preparando CIFAR-10...")

    dataset_cifar = CIFAR10(
        root="datasets/cifar10",
        train=False,
        download=True,
    )

    (
        top1_pokemon,
        top2_pokemon,
        margem_pokemon,
    ) = analisar_pokemons(
        modelo_pokemon,
        loader_teste,
    )

    (
        top1_cifar,
        top2_cifar,
        margem_cifar,
    ) = analisar_cifar(
        modelo_pokemon,
        dataset_cifar,
    )

    mostrar_resumo(
        "POKÉMON GEN 1",
        top1_pokemon,
        top2_pokemon,
        margem_pokemon,
    )

    mostrar_resumo(
        "CIFAR-10 / NÃO-POKÉMON",
        top1_cifar,
        top2_cifar,
        margem_cifar,
    )

    testar_thresholds_confianca(
        top1_pokemon,
        top1_cifar,
    )

    testar_thresholds_margem(
        margem_pokemon,
        margem_cifar,
    )

    testar_thresholds_combinados(
        top1_pokemon,
        margem_pokemon,
        top1_cifar,
        margem_cifar,
    )

    print()
    print("=" * 60)
    print("ANÁLISE CONCLUÍDA")
    print("=" * 60)


if __name__ == "__main__":
    main()