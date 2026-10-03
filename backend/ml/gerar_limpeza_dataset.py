import csv
import html
import json
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

from backend.ml.analisar_duplicatas import RELATORIO_PATH


ROOT_DIR = Path(__file__).resolve().parent.parent.parent

LIMPEZA_PATH = ROOT_DIR / "datasets" / "limpeza_pokemons.csv"
REVISAO_PATH = ROOT_DIR / "saidas" / "revisao_limpeza.html"

# Entre classes diferentes, só distâncias pequenas são cópias de
# verdade; de 3 em diante aparecem molduras e cartões parecidos.
DISTANCIA_MAXIMA_ENTRE_CLASSES = 2

# Dentro da mesma classe, a cópia mantida é a do conjunto mais
# protegido, para preservar o conjunto de teste.
PRIORIDADE_CONJUNTO = {
    "teste": 0,
    "validacao": 1,
    "treino": 2,
    "fora_da_divisao": 3,
}


def agrupar_mesma_classe(pares):
    pai = {}

    def raiz(imagem):
        pai.setdefault(imagem, imagem)

        while pai[imagem] != imagem:
            pai[imagem] = pai[pai[imagem]]
            imagem = pai[imagem]

        return imagem

    conjuntos = {}

    for par in pares:
        conjuntos[par["a"]] = par["conjunto_a"]
        conjuntos[par["b"]] = par["conjunto_b"]
        pai[raiz(par["a"])] = raiz(par["b"])

    grupos = defaultdict(list)

    for imagem in conjuntos:
        grupos[raiz(imagem)].append(imagem)

    return [
        sorted(
            grupo,
            key=lambda imagem: (
                PRIORIDADE_CONJUNTO[conjuntos[imagem]],
                imagem,
            ),
        )
        for grupo in grupos.values()
    ], conjuntos


def caminho_html(imagem):
    return "../datasets/pokemons/" + quote(imagem)


def figura(imagem, conjunto, destaque):
    classe = imagem.split("/")[0]

    return (
        f'<figure class="{destaque}">'
        f'<img src="{caminho_html(imagem)}" loading="lazy">'
        f"<figcaption><b>{html.escape(classe)}</b> · {conjunto}"
        f"<br><small>{html.escape(imagem)}</small></figcaption>"
        "</figure>"
    )


def gerar_html(pares_entre_classes, grupos, conjuntos):
    partes = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Revisão da limpeza</title>",
        "<style>",
        "body{font-family:sans-serif;margin:16px;background:#fafafa}",
        ".linha{display:flex;gap:8px;align-items:flex-start;"
        "border-bottom:1px solid #ddd;padding:8px 0}",
        "figure{margin:0;width:180px}",
        "img{max-width:180px;max-height:160px;display:block}",
        "small{color:#777;word-break:break-all}",
        ".remover{opacity:.55;outline:3px solid #d33}",
        ".manter{outline:3px solid #2a2}",
        ".num{width:40px;color:#999}",
        "</style>",
        "<h1>Revisão da limpeza do dataset</h1>",
        "<p>Vermelho = proposta de remover, verde = mantida. "
        "Para mudar uma decisão, edite a coluna <code>acao</code> de "
        "<code>datasets/limpeza_pokemons.csv</code> para "
        "<code>manter</code>.</p>",
        f"<h2>1. Mesma imagem em classes diferentes "
        f"({len(pares_entre_classes)} pares) — remover as duas</h2>",
        "<p>Não dá para saber automaticamente qual pasta está certa. "
        "Se uma delas estiver certa, marque essa como "
        "<code>manter</code> no CSV.</p>",
    ]

    for numero, par in enumerate(pares_entre_classes, start=1):
        partes.append(
            "<div class='linha'>"
            f"<div class='num'>{numero}</div>"
            + figura(par["a"], par["conjunto_a"], "remover")
            + figura(par["b"], par["conjunto_b"], "remover")
            + "</div>"
        )

    partes.append(
        f"<h2>2. Cópias dentro da mesma classe ({len(grupos)} grupos) "
        "— manter uma</h2>"
    )

    for numero, grupo in enumerate(grupos, start=1):
        partes.append(
            "<div class='linha'>"
            f"<div class='num'>{numero}</div>"
            + "".join(
                figura(
                    imagem,
                    conjuntos[imagem],
                    "manter" if indice == 0 else "remover",
                )
                for indice, imagem in enumerate(grupo)
            )
            + "</div>"
        )

    return "\n".join(partes)


def main():
    print("=" * 60)
    print("PROPOSTA DE LIMPEZA DO DATASET")
    print("=" * 60)

    if not RELATORIO_PATH.exists():
        print(f"Relatorio nao encontrado: {RELATORIO_PATH}")
        print("Gere com:")
        print("python -m backend.ml.analisar_duplicatas")
        return

    if LIMPEZA_PATH.exists():
        print(f"{LIMPEZA_PATH.name} ja existe e nao sera sobrescrito.")
        print("Apague o arquivo para gerar uma proposta nova.")
        return

    with open(
        RELATORIO_PATH,
        mode="r",
        encoding="utf-8",
    ) as arquivo:
        relatorio = json.load(arquivo)

    pares = relatorio["pares"]

    pares_entre_classes = [
        par
        for par in pares
        if par["tipo"] == "classes_diferentes"
        and par["distancia"] <= DISTANCIA_MAXIMA_ENTRE_CLASSES
    ]

    grupos, conjuntos = agrupar_mesma_classe(
        [
            par
            for par in pares
            if par["tipo"] != "classes_diferentes"
        ]
    )

    remocoes = {}

    for par in pares_entre_classes:
        for lado, outro in (("a", "b"), ("b", "a")):
            remocoes.setdefault(
                par[lado],
                {
                    "conjunto": par[f"conjunto_{lado}"],
                    "motivo": "classe_duvidosa",
                    "copia_de": par[outro],
                },
            )

    for grupo in grupos:
        mantida = grupo[0]

        for imagem in grupo[1:]:
            remocoes.setdefault(
                imagem,
                {
                    "conjunto": conjuntos[imagem],
                    "motivo": "copia_mesma_classe",
                    "copia_de": mantida,
                },
            )

    with open(
        LIMPEZA_PATH,
        mode="w",
        encoding="utf-8-sig",
        newline="",
    ) as arquivo:
        writer = csv.writer(arquivo)

        writer.writerow(
            ["imagem", "conjunto", "motivo", "copia_de", "acao"]
        )

        for imagem in sorted(remocoes):
            dados = remocoes[imagem]

            writer.writerow(
                [
                    imagem,
                    dados["conjunto"],
                    dados["motivo"],
                    dados["copia_de"],
                    "remover",
                ]
            )

    REVISAO_PATH.parent.mkdir(exist_ok=True)

    REVISAO_PATH.write_text(
        gerar_html(pares_entre_classes, grupos, conjuntos),
        encoding="utf-8",
    )

    print()
    print(f"Pares entre classes diferentes: {len(pares_entre_classes)}")
    print(f"Grupos de copias na mesma classe: {len(grupos)}")

    print()
    print(f"Imagens propostas para remocao: {len(remocoes)}")

    for conjunto in ("treino", "validacao", "teste"):
        quantidade = sum(
            1
            for dados in remocoes.values()
            if dados["conjunto"] == conjunto
        )

        print(f"  {conjunto}: {quantidade}")

    print()
    print(f"Lista para revisar/editar: {LIMPEZA_PATH}")
    print(f"Pagina de revisao: {REVISAO_PATH}")


if __name__ == "__main__":
    main()
