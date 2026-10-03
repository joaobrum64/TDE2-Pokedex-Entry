import csv
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

from backend.ml.pipeline import DETECTOR_PATH, PipelinePokemon


ROOT_DIR = Path(__file__).resolve().parent.parent.parent

DATASETS_PATH = ROOT_DIR / "datasets"

GABARITO_PATH = (
    DATASETS_PATH
    / "avaliacao_pipeline"
    / "gabarito.json"
)

OUTPUT_PATH = ROOT_DIR / "saidas"

QUANTIDADE_CONFUSOES = 15
QUANTIDADE_PIORES_CLASSES = 10


def porcentagem(parte, total):
    if total == 0:
        return None

    return round(parte / total * 100, 2)


def formatar(valor):
    if valor is None:
        return "     -"

    return f"{valor:5.2f}%"


def processar_imagens(pipeline, itens):
    linhas = []

    for indice, item in enumerate(itens, start=1):
        with Image.open(
            DATASETS_PATH / item["imagem"]
        ) as imagem:
            imagem.load()

        resultados = pipeline.processar(imagem)

        linhas.append(
            {
                "imagem": item["imagem"],
                "categoria": item["categoria"],
                "esperado": item["esperado"],
                "obtido": [
                    resultado["pokemon"]
                    for resultado in resultados
                ],
                "fallback": any(
                    resultado["fallback"]
                    for resultado in resultados
                ),
            }
        )

        if indice % 200 == 0 or indice == len(itens):
            print(f"{indice}/{len(itens)}")

    return linhas


def resumir_categoria(linhas):
    exatas = 0
    esperados = 0
    devolvidos = 0
    corretos = 0
    vazias = 0
    vazias_com_falso_positivo = 0
    com_fallback = 0

    for linha in linhas:
        esperado = Counter(linha["esperado"])
        obtido = Counter(linha["obtido"])

        if esperado == obtido:
            exatas += 1

        esperados += sum(esperado.values())
        devolvidos += sum(obtido.values())
        corretos += sum((esperado & obtido).values())

        if not esperado:
            vazias += 1

            if obtido:
                vazias_com_falso_positivo += 1

        if linha["fallback"]:
            com_fallback += 1

    return {
        "imagens": len(linhas),
        # Imagem exata: devolveu exatamente os Pokémon esperados,
        # sem faltar e sem sobrar nenhum.
        "imagens_exatas": porcentagem(exatas, len(linhas)),
        # Recall: dos Pokémon que existiam, quantos foram achados.
        "recall": porcentagem(corretos, esperados),
        # Precisão: dos Pokémon devolvidos, quantos estavam certos.
        "precisao": porcentagem(corretos, devolvidos),
        "falso_positivo_sem_pokemon": porcentagem(
            vazias_com_falso_positivo,
            vazias,
        ),
        "imagens_com_fallback": com_fallback,
    }


def resumir_unicos(linhas):
    confusoes = Counter()
    nao_encontrados = 0
    total_por_classe = Counter()
    exatas_por_classe = Counter()

    for linha in linhas:
        esperado = linha["esperado"][0]

        total_por_classe[esperado] += 1

        if linha["obtido"] == [esperado]:
            exatas_por_classe[esperado] += 1
            continue

        if not linha["obtido"]:
            nao_encontrados += 1

        for obtido in linha["obtido"]:
            if obtido != esperado:
                confusoes[f"{esperado} -> {obtido}"] += 1

    piores_classes = sorted(
        (
            (
                porcentagem(
                    exatas_por_classe[classe],
                    total,
                ),
                classe,
                total,
            )
            for classe, total in total_por_classe.items()
        )
    )[:QUANTIDADE_PIORES_CLASSES]

    return {
        "nao_encontrados": nao_encontrados,
        "confusoes": confusoes.most_common(
            QUANTIDADE_CONFUSOES
        ),
        "piores_classes": [
            {
                "classe": classe,
                "imagens_exatas": exatas,
                "imagens": total,
            }
            for exatas, classe, total in piores_classes
        ],
    }


def mostrar_resumo(resumo):
    print()
    print("=" * 78)
    print("RESULTADO POR CATEGORIA")
    print("=" * 78)

    print()
    print(
        f"{'categoria':<12}"
        f"{'imagens':>8}"
        f"{'exatas':>10}"
        f"{'recall':>10}"
        f"{'precisao':>10}"
        f"{'falso pos.':>12}"
        f"{'fallback':>10}"
    )

    for categoria, dados in resumo["categorias"].items():
        print(
            f"{categoria:<12}"
            f"{dados['imagens']:>8}"
            f"{formatar(dados['imagens_exatas']):>10}"
            f"{formatar(dados['recall']):>10}"
            f"{formatar(dados['precisao']):>10}"
            f"{formatar(dados['falso_positivo_sem_pokemon']):>12}"
            f"{dados['imagens_com_fallback']:>10}"
        )

    unicos = resumo["unicos"]

    print()
    print(
        "Pokemon unico sem nenhum resultado: "
        f"{unicos['nao_encontrados']}"
    )

    print()
    print("Confusoes mais frequentes (esperado -> obtido):")

    for confusao, quantidade in unicos["confusoes"]:
        print(f"  {quantidade:>3}  {confusao}")

    print()
    print("Classes com menos imagens exatas:")

    for classe in unicos["piores_classes"]:
        print(
            f"  {formatar(classe['imagens_exatas'])}  "
            f"{classe['classe']} "
            f"({classe['imagens']} imagens)"
        )


def salvar_resultados(rotulo, resumo, linhas):
    OUTPUT_PATH.mkdir(exist_ok=True)

    caminho_resumo = (
        OUTPUT_PATH
        / f"avaliacao_pipeline_{rotulo}.json"
    )

    caminho_imagens = (
        OUTPUT_PATH
        / f"avaliacao_pipeline_{rotulo}.csv"
    )

    with open(
        caminho_resumo,
        mode="w",
        encoding="utf-8",
    ) as arquivo:
        json.dump(
            resumo,
            arquivo,
            ensure_ascii=False,
            indent=2,
        )

    with open(
        caminho_imagens,
        mode="w",
        encoding="utf-8-sig",
        newline="",
    ) as arquivo:
        writer = csv.writer(arquivo)

        writer.writerow(
            [
                "imagem",
                "categoria",
                "esperado",
                "obtido",
                "exata",
                "fallback",
            ]
        )

        for linha in linhas:
            writer.writerow(
                [
                    linha["imagem"],
                    linha["categoria"],
                    " | ".join(sorted(linha["esperado"])),
                    " | ".join(sorted(linha["obtido"])),
                    Counter(linha["esperado"])
                    == Counter(linha["obtido"]),
                    linha["fallback"],
                ]
            )

    print()
    print(f"Resumo salvo em: {caminho_resumo}")
    print(f"Resultado por imagem salvo em: {caminho_imagens}")


def main():
    # O rótulo identifica a versão avaliada nos arquivos de saída,
    # para comparar execuções (ex.: "v2", "v3").
    rotulo = sys.argv[1] if len(sys.argv) > 1 else "atual"

    # Opcional: outro classificador em modelos/ (ex.:
    # pokemon_resnet18_v3.pth). Sem ele, usa o mesmo da API.
    caminho_classificador = (
        ROOT_DIR / "modelos" / sys.argv[2]
        if len(sys.argv) > 2 and sys.argv[2] != "-"
        else None
    )

    # Opcional: outro detector em modelos/. Use "-" no lugar do
    # classificador para manter o da API, ex.:
    # avaliar_pipeline det_v2 - pokemon_object_detector_v2.pth
    caminho_detector = (
        ROOT_DIR / "modelos" / sys.argv[3]
        if len(sys.argv) > 3
        else DETECTOR_PATH
    )

    if not GABARITO_PATH.exists():
        print(f"Gabarito nao encontrado: {GABARITO_PATH}")
        print("Gere com:")
        print("python -m backend.ml.gerar_avaliacao_pipeline")
        return

    with open(
        GABARITO_PATH,
        mode="r",
        encoding="utf-8",
    ) as arquivo:
        itens = json.load(arquivo)

    print("=" * 78)
    print("AVALIACAO DO PIPELINE COMPLETO")
    print("=" * 78)

    print()
    print("Carregando pipeline...")

    pipeline = PipelinePokemon(caminho_classificador, caminho_detector)

    print(f"Pipeline carregado em: {pipeline.dispositivo}")
    print()

    linhas = processar_imagens(pipeline, itens)

    categorias = sorted(
        {linha["categoria"] for linha in linhas}
    )

    resumo = {
        "rotulo": rotulo,
        "categorias": {
            categoria: resumir_categoria(
                [
                    linha
                    for linha in linhas
                    if linha["categoria"] == categoria
                ]
            )
            for categoria in categorias
        },
        "unicos": resumir_unicos(
            [
                linha
                for linha in linhas
                if linha["categoria"] == "unico"
            ]
        ),
    }

    mostrar_resumo(resumo)

    salvar_resultados(rotulo, resumo, linhas)


if __name__ == "__main__":
    main()
