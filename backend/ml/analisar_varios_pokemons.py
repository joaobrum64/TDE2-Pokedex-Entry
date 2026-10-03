import csv
import html
from pathlib import Path
from urllib.parse import quote
from PIL import Image
from backend.ml.analise_dataset import normalizar_nome
from backend.ml.pipeline import PipelinePokemon
from backend.ml.preparar_dados import (
    DATASET_PATH,
    LIMPEZA_PATH,
    carregar_amostras,
)


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
REVISAO_PATH = ROOT_DIR / "saidas" / "revisao_varios_pokemons.html"
SCORE_MINIMO_REGIAO = 70.0
AREA_MINIMA_REGIAO = 0.04
CONFIANCA_MINIMA_OUTRA_ESPECIE = 80.0


def analisar_imagem(pipeline, caminho, classe_esperada):
    with Image.open(caminho) as imagem:
        imagem = imagem.convert("RGB")

    largura, altura = imagem.size
    area_imagem = largura * altura

    regioes = []

    for deteccao in pipeline.detectar_regioes(imagem):
        x1, y1, x2, y2 = deteccao["box"]

        if deteccao["score"] < SCORE_MINIMO_REGIAO:
            continue

        if (x2 - x1) * (y2 - y1) < AREA_MINIMA_REGIAO * area_imagem:
            continue

        melhor = pipeline.classificador.prever(
            imagem.crop(deteccao["box"])
        )[0]

        regioes.append(
            {
                "box": (
                    x1 / largura * 100,
                    y1 / altura * 100,
                    (x2 - x1) / largura * 100,
                    (y2 - y1) / altura * 100,
                ),
                "pokemon": melhor["pokemon"],
                "confianca": melhor["confianca"],
                "outra_especie": (
                    normalizar_nome(melhor["pokemon"])
                    != normalizar_nome(classe_esperada)
                    and melhor["confianca"]
                    >= CONFIANCA_MINIMA_OUTRA_ESPECIE
                ),
            }
        )

    outras = [
        regiao
        for regiao in regioes
        if regiao["outra_especie"]
    ]

    if len(regioes) < 2 or not outras:
        return None

    return {
        "regioes": regioes,
        "outras_especies": sorted(
            {regiao["pokemon"] for regiao in outras}
        ),
        "suspeita": max(regiao["confianca"] for regiao in outras),
    }


def gerar_html(suspeitas):
    partes = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Revisão de vários Pokémon</title>",
        "<style>",
        "body{font-family:sans-serif;margin:16px;background:#fafafa}",
        ".grade{display:flex;flex-wrap:wrap;gap:12px}",
        "figure{margin:0;width:260px;background:#fff;padding:6px;"
        "border:1px solid #ddd}",
        ".quadro{position:relative;display:inline-block}",
        ".quadro img{max-width:248px;max-height:248px;display:block}",
        ".caixa{position:absolute;border:2px solid #2a2;"
        "box-sizing:border-box}",
        ".caixa.outra{border-color:#d33}",
        ".caixa span{position:absolute;top:0;left:0;font-size:10px;"
        "background:inherit;color:#fff;padding:0 2px;"
        "background:#2a2;white-space:nowrap}",
        ".caixa.outra span{background:#d33}",
        "small{color:#777;word-break:break-all}",
        "</style>",
        "<h1>Imagens que podem ter mais de um Pokémon</h1>",
        "<p>Vermelho = região classificada como outra espécie; verde = "
        "a espécie da pasta (ou baixa confiança). Ordenadas da mais "
        "suspeita para a menos. Para tirar uma imagem do treino, troque "
        "a <code>acao</code> dela de <code>revisar</code> para "
        "<code>remover</code> em <code>datasets/limpeza_pokemons.csv</code>; "
        "<code>revisar</code> e <code>manter</code> mantêm a imagem.</p>",
        "<div class='grade'>",
    ]

    for numero, item in enumerate(suspeitas, start=1):
        caixas = "".join(
            f"<div class='caixa{' outra' if regiao['outra_especie'] else ''}' "
            f"style='left:{regiao['box'][0]:.1f}%;top:{regiao['box'][1]:.1f}%;"
            f"width:{regiao['box'][2]:.1f}%;height:{regiao['box'][3]:.1f}%'>"
            f"<span>{html.escape(regiao['pokemon'])} "
            f"{regiao['confianca']:.0f}%</span></div>"
            for regiao in item["regioes"]
        )

        partes.append(
            "<figure>"
            "<div class='quadro'>"
            f"<img src='../datasets/pokemons/{quote(item['imagem'])}' "
            "loading='lazy'>"
            f"{caixas}</div>"
            f"<figcaption>#{numero} <b>{html.escape(item['classe'])}</b> · "
            f"{item['conjunto']}<br><small>{html.escape(item['imagem'])}"
            "</small></figcaption></figure>"
        )

    partes.append("</div>")

    return "\n".join(partes)


def main():
    print("=" * 60)
    print("BUSCA DE IMAGENS COM MAIS DE UM POKEMON")
    print("=" * 60)

    if not LIMPEZA_PATH.exists():
        print(f"Arquivo nao encontrado: {LIMPEZA_PATH}")
        return

    with open(
        LIMPEZA_PATH,
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as arquivo:
        ja_listadas = {
            linha["imagem"]
            for linha in csv.DictReader(arquivo)
        }

    classes, _, treino, validacao, teste = carregar_amostras()

    amostras = [
        (caminho, classes[classe], nome_conjunto)
        for nome_conjunto, conjunto in (
            ("treino", treino),
            ("validacao", validacao),
            ("teste", teste),
        )
        for caminho, classe in conjunto
        if caminho.relative_to(DATASET_PATH).as_posix()
        not in ja_listadas
    ]

    print()
    print("Carregando pipeline...")

    pipeline = PipelinePokemon()

    print(f"Pipeline carregado em: {pipeline.dispositivo}")
    print(f"Imagens a analisar: {len(amostras)}")
    print()

    suspeitas = []

    for indice, (caminho, classe, conjunto) in enumerate(
        amostras,
        start=1,
    ):
        resultado = analisar_imagem(pipeline, caminho, classe)

        if resultado is not None:
            suspeitas.append(
                {
                    "imagem": caminho.relative_to(DATASET_PATH).as_posix(),
                    "classe": classe,
                    "conjunto": conjunto,
                    **resultado,
                }
            )

        if indice % 1000 == 0 or indice == len(amostras):
            print(
                f"{indice}/{len(amostras)} "
                f"(suspeitas: {len(suspeitas)})"
            )

    suspeitas.sort(
        key=lambda item: item["suspeita"],
        reverse=True,
    )

    with open(
        LIMPEZA_PATH,
        mode="a",
        encoding="utf-8",
        newline="",
    ) as arquivo:
        writer = csv.writer(arquivo, lineterminator="\n")

        for item in suspeitas:
            writer.writerow(
                [
                    item["imagem"],
                    item["conjunto"],
                    "varios_pokemons",
                    " | ".join(item["outras_especies"]),
                    "revisar",
                ]
            )

    REVISAO_PATH.parent.mkdir(exist_ok=True)

    REVISAO_PATH.write_text(
        gerar_html(suspeitas),
        encoding="utf-8",
    )

    print()
    print(f"Imagens suspeitas: {len(suspeitas)}")

    for conjunto in ("treino", "validacao", "teste"):
        quantidade = sum(
            1
            for item in suspeitas
            if item["conjunto"] == conjunto
        )

        print(f"  {conjunto}: {quantidade}")

    print()
    print(f"Adicionadas a {LIMPEZA_PATH.name} com acao 'revisar'")
    print(f"Pagina de revisao: {REVISAO_PATH}")


if __name__ == "__main__":
    main()
