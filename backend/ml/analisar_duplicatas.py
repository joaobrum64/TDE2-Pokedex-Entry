import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops

from backend.ml.preparar_dados import (
    DATASET_PATH,
    EXTENSOES_ACEITAS,
    carregar_amostras,
)


ROOT_DIR = Path(__file__).resolve().parent.parent.parent

OUTPUT_PATH = ROOT_DIR / "saidas"
RELATORIO_PATH = OUTPUT_PATH / "duplicatas_dataset.json"
DISTANCIA_MAXIMA = 4
TAMANHO_BLOCO = 1000

def calcular_dhash(imagem):

    cinza = imagem.convert("L")
    fundo = Image.new("L", cinza.size, cinza.getpixel((0, 0)))
    diferenca = ImageChops.difference(cinza, fundo).point(
        lambda valor: 255 if valor > 16 else 0
    )
    caixa = diferenca.getbbox()

    if caixa is not None:
        cinza = cinza.crop(caixa)

    pequena = cinza.resize(
        (9, 8),
        Image.Resampling.LANCZOS,
    )

    pixels = np.asarray(pequena, dtype=np.int16)

    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()

    return int(
        "".join("1" if bit else "0" for bit in bits),
        2,
    )


def analisar_arquivos(conjunto_por_caminho):
    invalidos = []
    extensoes_ignoradas = []
    registros = []

    arquivos = sorted(
        arquivo
        for pasta in DATASET_PATH.iterdir()
        if pasta.is_dir()
        for arquivo in pasta.iterdir()
        if arquivo.is_file()
    )

    for indice, arquivo in enumerate(arquivos, start=1):
        relativo = arquivo.relative_to(DATASET_PATH).as_posix()

        if arquivo.suffix.lower() not in EXTENSOES_ACEITAS:
            extensoes_ignoradas.append(relativo)
            continue

        conteudo = arquivo.read_bytes()

        try:
            with Image.open(arquivo) as imagem:
                imagem.load()
                largura, altura = imagem.size
                dhash = calcular_dhash(imagem)

        except Exception as erro:
            invalidos.append(
                {
                    "imagem": relativo,
                    "erro": f"{type(erro).__name__}: {erro}",
                }
            )
            continue

        registros.append(
            {
                "imagem": relativo,
                "classe": arquivo.parent.name,
                "conjunto": conjunto_por_caminho.get(
                    relativo,
                    "fora_da_divisao",
                ),
                "md5": hashlib.md5(conteudo).hexdigest(),
                "dhash": dhash,
                "largura": largura,
                "altura": altura,
            }
        )

        if indice % 2000 == 0 or indice == len(arquivos):
            print(f"{indice}/{len(arquivos)}")

    return registros, invalidos, extensoes_ignoradas


def encontrar_pares(registros):
    hashes = np.array(
        [registro["dhash"] for registro in registros],
        dtype=np.uint64,
    )

    pares = []

    for inicio in range(0, len(hashes), TAMANHO_BLOCO):
        bloco = hashes[inicio:inicio + TAMANHO_BLOCO]

        distancias = np.bitwise_count(
            bloco[:, None] ^ hashes[None, :]
        )

        linhas, colunas = np.nonzero(
            distancias <= DISTANCIA_MAXIMA
        )

        for linha, coluna in zip(linhas, colunas):
            i = inicio + int(linha)
            j = int(coluna)

            if i < j:
                pares.append(
                    (i, j, int(distancias[linha, coluna]))
                )

    return pares


def agrupar(quantidade, pares):

    pai = list(range(quantidade))

    def raiz(indice):
        while pai[indice] != indice:
            pai[indice] = pai[pai[indice]]
            indice = pai[indice]

        return indice

    for i, j, _ in pares:
        pai[raiz(i)] = raiz(j)

    grupos = defaultdict(list)

    for indice in range(quantidade):
        grupos[raiz(indice)].append(indice)

    return [
        grupo
        for grupo in grupos.values()
        if len(grupo) > 1
    ]


def main():
    print("=" * 60)
    print("ANALISE DE DUPLICATAS DO DATASET")
    print("=" * 60)

    _, _, treino, validacao, teste = carregar_amostras(aplicar_exclusoes=False)

    conjunto_por_caminho = {}

    for nome_conjunto, amostras in (
        ("treino", treino),
        ("validacao", validacao),
        ("teste", teste),
    ):
        for caminho, _ in amostras:
            conjunto_por_caminho[
                caminho.relative_to(DATASET_PATH).as_posix()
            ] = nome_conjunto

    print()
    print("Lendo imagens...")

    registros, invalidos, extensoes_ignoradas = analisar_arquivos(
        conjunto_por_caminho
    )

    print()
    print("Comparando hashes...")

    pares = encontrar_pares(registros)

    grupos = agrupar(len(registros), pares)

    exatos = sum(
        1
        for i, j, _ in pares
        if registros[i]["md5"] == registros[j]["md5"]
    )

    tipos_de_par = Counter()
    pares_relatorio = []

    for i, j, distancia in pares:
        a = registros[i]
        b = registros[j]

        mesma_classe = a["classe"] == b["classe"]

        conjuntos = "/".join(sorted((a["conjunto"], b["conjunto"])))

        if not mesma_classe:
            tipo = "classes_diferentes"
        elif a["conjunto"] != b["conjunto"]:
            tipo = f"vazamento_{conjuntos}"
        else:
            tipo = f"dentro_{a['conjunto']}"

        tipos_de_par[tipo] += 1

        pares_relatorio.append(
            {
                "tipo": tipo,
                "distancia": distancia,
                "identico": a["md5"] == b["md5"],
                "a": a["imagem"],
                "conjunto_a": a["conjunto"],
                "b": b["imagem"],
                "conjunto_b": b["conjunto"],
            }
        )

    teste_vazado = set()

    for par in pares_relatorio:
        if par["tipo"] != "vazamento_teste/treino":
            continue

        if par["conjunto_a"] == "teste":
            teste_vazado.add(par["a"])
        else:
            teste_vazado.add(par["b"])

    pequenas = [
        registro["imagem"]
        for registro in registros
        if min(registro["largura"], registro["altura"]) < 64
    ]

    relatorio = {
        "distancia_maxima": DISTANCIA_MAXIMA,
        "imagens_lidas": len(registros),
        "invalidos": invalidos,
        "extensoes_ignoradas": extensoes_ignoradas,
        "imagens_menores_que_64px": pequenas,
        "pares_quase_iguais": len(pares),
        "pares_identicos": exatos,
        "pares_por_tipo": dict(tipos_de_par),
        "grupos": len(grupos),
        "imagens_em_grupos": sum(len(g) for g in grupos),
        "imagens_de_teste_com_copia_no_treino": sorted(teste_vazado),
        "pares": sorted(
            pares_relatorio,
            key=lambda par: (par["tipo"], par["distancia"]),
        ),
    }

    OUTPUT_PATH.mkdir(exist_ok=True)

    with open(
        RELATORIO_PATH,
        mode="w",
        encoding="utf-8",
    ) as arquivo:
        json.dump(
            relatorio,
            arquivo,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 60)
    print("RESUMO")
    print("=" * 60)

    print()
    print(f"Imagens lidas: {len(registros)}")
    print(f"Arquivos que nao abrem: {len(invalidos)}")
    print(
        "Arquivos com extensao ignorada pelo treino: "
        f"{len(extensoes_ignoradas)}"
    )
    print(f"Imagens menores que 64px: {len(pequenas)}")

    print()
    print(
        f"Pares quase iguais (distancia <= {DISTANCIA_MAXIMA}): "
        f"{len(pares)} ({exatos} identicos byte a byte)"
    )

    for tipo, quantidade in sorted(tipos_de_par.items()):
        print(f"  {tipo}: {quantidade}")

    print()
    print(
        f"Grupos de imagens repetidas: {len(grupos)} "
        f"({relatorio['imagens_em_grupos']} imagens)"
    )

    print(
        "Imagens de teste com quase copia da mesma classe "
        f"no treino: {len(teste_vazado)} de {len(teste)}"
    )

    print()
    print(f"Relatorio salvo em: {RELATORIO_PATH}")


if __name__ == "__main__":
    main()
