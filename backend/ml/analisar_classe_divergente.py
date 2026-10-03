import csv
import html
import random
from pathlib import Path
from urllib.parse import quote
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from backend.ml.analise_dataset import normalizar_nome
from backend.ml.preparar_dados import (
    DATASET_PATH,
    LIMPEZA_PATH,
    PokemonDataset,
    carregar_amostras,
    transform_avaliacao,
)


ROOT_DIR = Path(__file__).resolve().parent.parent.parent

CSV_PATH = ROOT_DIR / "datasets" / "pokemon_gen1.csv"
REVISAO_PATH = ROOT_DIR / "saidas" / "revisao_classe_divergente.html"
CONFIANCA_MINIMA = 60.0
QUANTIDADE_DOBRAS = 5
PASSOS_TREINO_LINEAR = 300
SEED = 42


def extrair_caracteristicas(caminhos, dispositivo):
    modelo = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    modelo.classifier[2] = nn.Identity()

    modelo = modelo.to(dispositivo).eval()

    loader = DataLoader(
        PokemonDataset(
            [(caminho, 0) for caminho in caminhos],
            transform=transform_avaliacao,
        ),
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    lotes = []

    with torch.no_grad():
        for indice, (imagens, _) in enumerate(loader, start=1):
            lotes.append(modelo(imagens.to(dispositivo)).cpu())

            if indice % 20 == 0 or indice == len(loader):
                print(f"Caracteristicas: lote {indice}/{len(loader)}")

    return torch.cat(lotes)


def treinar_linear(x, y, quantidade_classes, dispositivo):
    camada = nn.Linear(x.shape[1], quantidade_classes).to(dispositivo)

    otimizador = torch.optim.Adam(
        camada.parameters(),
        lr=0.01,
        weight_decay=0.0001,
    )

    for _ in range(PASSOS_TREINO_LINEAR):
        otimizador.zero_grad()
        nn.functional.cross_entropy(camada(x), y).backward()
        otimizador.step()

    return camada


def prever_fora_da_dobra(x, y, quantidade_classes, dispositivo):

    sorteio = random.Random(SEED)

    dobra = torch.empty(len(y), dtype=torch.long)

    for classe in range(quantidade_classes):
        indices = (y == classe).nonzero().flatten().tolist()
        sorteio.shuffle(indices)

        for posicao, indice in enumerate(indices):
            dobra[indice] = posicao % QUANTIDADE_DOBRAS

    media = x.mean(dim=0)
    desvio = x.std(dim=0) + 1e-6

    x = ((x - media) / desvio).to(dispositivo)
    y = y.to(dispositivo)
    dobra = dobra.to(dispositivo)

    probabilidades = torch.empty(
        len(y),
        quantidade_classes,
        device=dispositivo,
    )

    for numero in range(QUANTIDADE_DOBRAS):
        teste = dobra == numero

        camada = treinar_linear(
            x[~teste],
            y[~teste],
            quantidade_classes,
            dispositivo,
        )

        with torch.no_grad():
            probabilidades[teste] = torch.softmax(camada(x[teste]), dim=1)

        print(f"Dobra {numero + 1}/{QUANTIDADE_DOBRAS} concluida")

    return probabilidades.cpu()


def carregar_numeros():
    with open(
        CSV_PATH,
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as arquivo:
        return {
            normalizar_nome(pokemon["name"]): pokemon["pokedex_number"]
            for pokemon in csv.DictReader(arquivo)
        }


def sprite(numeros, classe):
    numero = numeros[normalizar_nome(classe)]

    return (
        f"<img class='sprite' src='../datasets/sprites/{numero}.png' "
        f"title='{html.escape(classe)}'>"
    )


def gerar_html(suspeitas, numeros):
    partes = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Revisão de classe divergente</title>",
        "<style>",
        "body{font-family:sans-serif;margin:16px;background:#fafafa}",
        ".grade{display:flex;flex-wrap:wrap;gap:12px}",
        "figure{margin:0;width:260px;background:#fff;padding:6px;"
        "border:1px solid #ddd}",
        ".foto{max-width:248px;max-height:220px;display:block}",
        ".sprites{display:flex;justify-content:space-between;"
        "align-items:center;font-size:12px}",
        ".sprite{width:64px;height:64px;image-rendering:pixelated}",
        ".pasta{color:#2a2}.modelo{color:#d33}",
        "small{color:#777;word-break:break-all}",
        "</style>",
        "<h1>Imagens em que o modelo vê outra espécie</h1>",
        "<p>Embaixo de cada foto: à esquerda (verde) a espécie da pasta, "
        "à direita (vermelho) a espécie que o modelo escolheu. Ordenadas "
        "da maior confiança para a menor. Para tirar uma imagem do "
        "treino, troque a <code>acao</code> dela de <code>revisar</code> "
        "para <code>remover</code> em "
        "<code>datasets/limpeza_pokemons.csv</code>.</p>",
        "<div class='grade'>",
    ]

    for numero, item in enumerate(suspeitas, start=1):
        partes.append(
            "<figure>"
            f"<img class='foto' src='../datasets/pokemons/"
            f"{quote(item['imagem'])}' loading='lazy'>"
            "<div class='sprites'>"
            f"<span class='pasta'>{sprite(numeros, item['classe'])}"
            f"<br>{html.escape(item['classe'])}</span>"
            f"<span>→</span>"
            f"<span class='modelo'>{sprite(numeros, item['previsto'])}"
            f"<br>{html.escape(item['previsto'])} "
            f"{item['confianca']:.0f}%</span>"
            "</div>"
            f"<figcaption>#{numero} · {item['conjunto']}<br>"
            f"<small>{html.escape(item['imagem'])}</small></figcaption>"
            "</figure>"
        )

    partes.append("</div>")

    return "\n".join(partes)


def main():
    print("=" * 60)
    print("BUSCA DE IMAGENS COM CLASSE DIVERGENTE")
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
        (caminho, classe, nome_conjunto)
        for nome_conjunto, conjunto in (
            ("treino", treino),
            ("validacao", validacao),
            ("teste", teste),
        )
        for caminho, classe in conjunto
        if caminho.relative_to(DATASET_PATH).as_posix()
        not in ja_listadas
    ]

    dispositivo = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(f"Imagens a analisar: {len(amostras)}")
    print()

    x = extrair_caracteristicas(
        [caminho for caminho, _, _ in amostras],
        dispositivo,
    )

    y = torch.tensor([classe for _, classe, _ in amostras])

    print()

    probabilidades = prever_fora_da_dobra(
        x,
        y,
        len(classes),
        dispositivo,
    )

    confiancas, previstos = probabilidades.max(dim=1)

    acertos = (previstos == y).float().mean().item() * 100

    print()
    print(f"Acuracia fora da dobra do modelo de checagem: {acertos:.2f}%")

    suspeitas = []

    for (caminho, classe, conjunto), previsto, confianca in zip(
        amostras,
        previstos.tolist(),
        (confiancas * 100).tolist(),
    ):
        if previsto != classe and confianca >= CONFIANCA_MINIMA:
            suspeitas.append(
                {
                    "imagem": caminho.relative_to(DATASET_PATH).as_posix(),
                    "classe": classes[classe],
                    "conjunto": conjunto,
                    "previsto": classes[previsto],
                    "confianca": confianca,
                }
            )

    suspeitas.sort(
        key=lambda item: item["confianca"],
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
                    "classe_divergente",
                    f"{item['previsto']} {item['confianca']:.0f}%",
                    "revisar",
                ]
            )

    REVISAO_PATH.parent.mkdir(exist_ok=True)

    REVISAO_PATH.write_text(
        gerar_html(suspeitas, carregar_numeros()),
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
