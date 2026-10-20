import csv
import re
import unicodedata
from pathlib import Path

from backend.tipos import (
    multiplicador_defensivo,
    resumo_defensivo,
    super_efetivo_contra,
)


ROOT_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = (
    ROOT_DIR
    / "datasets"
    / "pokemon_gen1.csv"
)

SPRITES_PATH = (
    ROOT_DIR
    / "datasets"
    / "sprites"
)

# Os golpes vêm da versão base da Primeira Geração (Red/Blue).
COLUNA_GOLPES = "level_up_moves_red_blue"

# Outras versões da Primeira Geração presentes no CSV. Só entram na
# ficha quando os golpes diferem dos de Red/Blue.
COLUNAS_OUTRAS_VERSOES = {
    "Yellow": "level_up_moves_yellow",
    "Red/Green (Japão)": "level_up_moves_red_green_japan",
    "Blue (Japão)": "level_up_moves_blue_japan",
}

# Máximo de fichas no contexto de uma pergunta (detectados + citados
# + evoluções), para não estourar a janela de contexto do LLM.
MAXIMO_FICHAS_CONTEXTO = 6

# Máximo de linhas "X atacando Y" no contexto.
MAXIMO_CONFRONTOS = 12

# Tipos em português, usados no texto de contexto do LLM.
TIPOS_PT = {
    "bug": "Inseto",
    "dragon": "Dragão",
    "electric": "Elétrico",
    "fighting": "Lutador",
    "fire": "Fogo",
    "flying": "Voador",
    "ghost": "Fantasma",
    "grass": "Planta",
    "ground": "Terra",
    "ice": "Gelo",
    "normal": "Normal",
    "poison": "Venenoso",
    "psychic": "Psíquico",
    "rock": "Pedra",
    "water": "Água",
}

# Nomes de evolução no CSV que não seguem a grafia oficial.
NOMES_EVOLUCAO = {
    "nidoran-m": "Nidoran♂",
    "nidoran-f": "Nidoran♀",
}


def normalizar_nome(nome):
    # As classes do modelo vêm dos nomes das pastas de treino
    # (ex.: "Farfetchd", "MrMime"), enquanto o CSV usa a grafia
    # oficial ("Farfetch'd", "Mr. Mime"). Os símbolos ♀ e ♂ são
    # mantidos para não confundir os dois Nidoran.
    return (
        nome
        .lower()
        .replace(" ", "")
        .replace(".", "")
        .replace("'", "")
    )


def nome_legivel_golpe(nome):
    # "thunder-shock" -> "Thunder Shock"
    return " ".join(
        parte.capitalize()
        for parte in nome.split("-")
    )


def separar_golpes(texto):
    # "1:thunder-shock|9:thunder-wave" -> lista de {nivel, nome}
    golpes = []

    for item in texto.split("|"):
        if not item:
            continue

        nivel, nome = item.split(":", 1)

        golpes.append(
            {
                "nivel": int(nivel),
                "nome": nome_legivel_golpe(nome),
            }
        )

    return golpes


def simplificar_texto(texto):
    # Minúsculas, sem acentos e sem pontuação, para achar nomes de
    # Pokémon escritos de qualquer jeito numa pergunta.
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(
        caractere
        for caractere in texto
        if not unicodedata.combining(caractere)
    )

    return re.sub(r"[^a-z0-9♀♂]+", " ", texto).strip()


def melhor_ataque(atacante, defensor):
    # Maior multiplicador entre os tipos do atacante contra o defensor.
    return max(
        (
            multiplicador_defensivo(tipo, defensor["tipos"]),
            tipo,
        )
        for tipo in atacante["tipos"]
    )


def numero_ou_nada(texto):
    # Proteção para linhas novas no CSV sem altura ou peso.
    return float(texto) if texto else None


def carregar_linhas():
    with open(
        CSV_PATH,
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as arquivo:
        return list(csv.DictReader(arquivo))


class Pokedex:
    # Fichas dos 151 Pokémon montadas a partir de pokemon_gen1.csv.
    # É também a entrada para quem precisa dos dados de um Pokémon
    # identificado (ex.: a integração com o LLM): obter_ficha("Pikachu")
    # ou obter_ficha(25).

    def __init__(self):
        linhas = carregar_linhas()

        self.numero_por_nome = {
            normalizar_nome(linha["name"]): int(linha["pokedex_number"])
            for linha in linhas
        }

        self.nome_por_numero = {
            int(linha["pokedex_number"]): linha["name"]
            for linha in linhas
        }

        self.fichas = {
            int(linha["pokedex_number"]): self.montar_ficha(linha)
            for linha in linhas
        }

        # Formas simplificadas de cada nome para a busca em perguntas.
        # Os dois Nidoran também são achados por "nidoran" sozinho.
        self.nomes_para_busca = []

        for numero, nome in self.nome_por_numero.items():
            simplificado = simplificar_texto(nome)

            self.nomes_para_busca.append((simplificado, numero))

            sem_simbolo = simplificado.replace("♀", "").replace("♂", "").strip()

            if sem_simbolo != simplificado:
                self.nomes_para_busca.append((sem_simbolo, numero))

            sem_espaco = simplificado.replace(" ", "")

            if sem_espaco != simplificado:
                self.nomes_para_busca.append((sem_espaco, numero))

    def referencia(self, nome):
        # Nome escrito de qualquer jeito (classe do modelo ou campo de
        # evolução do CSV) -> {numero, nome oficial}.
        nome = NOMES_EVOLUCAO.get(nome, nome)

        numero = self.numero_por_nome.get(
            normalizar_nome(nome.replace("-", ""))
        )

        if numero is None:
            return None

        return {
            "numero": numero,
            "nome": self.nome_por_numero[numero],
        }

    def lista_de_referencias(self, texto):
        return [
            self.referencia(nome)
            for nome in texto.split("|")
            if nome
        ]

    def montar_ficha(self, linha):
        numero = int(linha["pokedex_number"])

        return {
            "numero": numero,
            "nome": linha["name"],
            "classificacao": linha["classification"],
            "tipos": [
                tipo
                for tipo in (linha["type1"], linha["type2"])
                if tipo
            ],
            "altura_m": numero_ou_nada(linha["height_m"]),
            "peso_kg": numero_ou_nada(linha["weight_kg"]),
            # Na Primeira Geração existe um único stat "Especial";
            # no CSV ele aparece repetido em sp_attack e sp_defense.
            "stats": {
                "hp": int(linha["hp"]),
                "ataque": int(linha["attack"]),
                "defesa": int(linha["defense"]),
                "especial": int(linha["sp_attack"]),
                "velocidade": int(linha["speed"]),
            },
            "lendario": linha["is_legendary"] == "1",
            "descricao": linha["description"],
            "evolui_de": self.lista_de_referencias(linha["evolves_from"]),
            "evolui_para": self.lista_de_referencias(linha["evolves_to"]),
            "golpes": separar_golpes(linha[COLUNA_GOLPES]),
            "golpes_outras_versoes": {
                versao: separar_golpes(linha[coluna])
                for versao, coluna in COLUNAS_OUTRAS_VERSOES.items()
                if linha[coluna] != linha[COLUNA_GOLPES]
            },
            "sprite": f"/sprites/{numero}.png",
        }

    def obter_ficha(self, nome_ou_numero):
        if isinstance(nome_ou_numero, int):
            return self.fichas.get(nome_ou_numero)

        referencia = self.referencia(nome_ou_numero)

        if referencia is None:
            return None

        return self.fichas[referencia["numero"]]

    def encontrar_citados(self, texto):
        # Números dos Pokémon mencionados num texto, pelo nome
        # ("blastoise", "Mr Mime") ou pelo número ("#9", "nº 9",
        # "número 9"), na ordem em que aparecem.
        simplificado = f" {simplificar_texto(texto)} "

        encontrados = []

        for nome, numero in self.nomes_para_busca:
            posicao = simplificado.find(f" {nome} ")

            if posicao >= 0:
                encontrados.append((posicao, numero))

        # Aqui a pontuação importa ("#", "º"), então só os acentos são
        # removidos ("número" -> "numero").
        sem_acentos = "".join(
            caractere
            for caractere in unicodedata.normalize("NFD", texto.lower())
            if not unicodedata.combining(caractere)
        )

        for correspondencia in re.finditer(
            r"(?:#\s*|\bn\.?[º°]\s*|\bnumero\s+)(\d{1,3})\b",
            sem_acentos,
        ):
            numero = int(correspondencia.group(1))

            if numero in self.fichas:
                encontrados.append((correspondencia.start(), numero))

        numeros = []

        for _, numero in sorted(encontrados):
            if numero not in numeros:
                numeros.append(numero)

        return numeros

    def montar_contexto_conversa(self, numeros_detectados, textos):
        # Contexto do RAG para uma pergunta.
        # textos: [pergunta atual, perguntas anteriores...]
        #
        # A ordem importa para a velocidade: primeiro vem o que é fixo na
        # conversa (fichas dos detectados e das evoluções), depois o que
        # muda a cada pergunta (citados e resumo calculado). O Ollama
        # reaproveita o começo do prompt que se repete entre perguntas,
        # então nas perguntas seguintes ele só precisa ler o final.
        #
        # O resumo calculado existe porque, nos testes, os modelos
        # pequenos erravam comparações de números e vantagens de tipo
        # mesmo com os dados nas fichas.
        citados_agora = self.encontrar_citados(textos[0]) if textos else []

        citados = []

        for texto in textos:
            for numero in self.encontrar_citados(texto):
                if numero not in citados and numero not in numeros_detectados:
                    citados.append(numero)

        detectados = list(dict.fromkeys(numeros_detectados))

        evolucoes = []

        for numero in detectados:
            for chave in ("evolui_de", "evolui_para"):
                for referencia in self.fichas[numero][chave]:
                    if (
                        referencia["numero"] not in detectados
                        and referencia["numero"] not in evolucoes
                    ):
                        evolucoes.append(referencia["numero"])

        # Fichas completas: detectados e citados, até o limite. As
        # evoluções entram resumidas, porque raramente são o assunto.
        completas = (detectados + citados)[:MAXIMO_FICHAS_CONTEXTO]
        evolucoes = [n for n in evolucoes if n not in completas]

        def nome(numero):
            return self.nome_por_numero[numero]

        partes = [
            "POKÉMON DETECTADOS NA IMAGEM DO USUÁRIO: "
            + (", ".join(nome(n) for n in numeros_detectados) or "nenhum"),
        ]

        for numero in detectados:
            if numero in completas:
                partes.append(
                    "--- FICHA (detectado na imagem) ---\n"
                    + montar_contexto(self.fichas[numero])
                )

        for numero in evolucoes:
            # Cabeçalho explícito: sem ele, o modelo às vezes descrevia a
            # evolução (ex.: Raichu) quando perguntavam sobre o detectado.
            parentes = ", ".join(
                nome(detectado)
                for detectado in detectados
                if numero in [
                    referencia["numero"]
                    for chave in ("evolui_de", "evolui_para")
                    for referencia in self.fichas[detectado][chave]
                ]
            )

            partes.append(
                f"--- FICHA RESUMIDA DE {nome(numero).upper()} (forma "
                f"evolutiva de {parentes}; NÃO foi detectado na imagem) ---\n"
                + montar_contexto(self.fichas[numero], resumida=True)
            )

        # Daqui para baixo muda a cada pergunta.
        partes.append(
            "POKÉMON DOS REGISTROS DE KANTO CITADOS NA PERGUNTA ATUAL: "
            + (
                ", ".join(nome(n) for n in citados_agora)
                or "nenhum (se a pergunta fala de um POKéMON pelo nome, "
                "ele não existe nos registros de Kanto)"
            )
        )

        for numero in citados:
            if numero in completas:
                partes.append(
                    "--- FICHA (citado na conversa) ---\n"
                    + montar_contexto(self.fichas[numero])
                )

        resumo = self.resumo_calculado(completas)

        if resumo:
            partes.append(
                "--- RESUMO CALCULADO (use exatamente estes resultados "
                "para comparações e batalhas) ---\n" + resumo
            )

        return "\n\n".join(partes)

    def lembrete_para_pergunta(self, numeros_detectados, pergunta):
        # Quando a pergunta é sobre outro Pokémon, um lembrete colado na
        # própria pergunta faz o modelo comparar com o detectado (só a
        # regra no prompt, longe da pergunta, era esquecida nos testes).
        outros = [
            numero
            for numero in self.encontrar_citados(pergunta)
            if numero not in numeros_detectados
        ]

        if not outros or not numeros_detectados:
            return ""

        detectados = ", ".join(
            self.nome_por_numero[numero] for numero in numeros_detectados
        )

        return (
            f"(Lembrete do sistema: responda e depois compare com "
            f"{detectados}, detectado na imagem, usando o RESUMO CALCULADO.)"
        )

    def resumo_calculado(self, numeros):
        # Comparações prontas entre dois ou mais Pokémon: ordem de cada
        # stat, razões de altura e peso e quem leva vantagem de tipo.
        if len(numeros) < 2:
            return ""

        fichas = [self.fichas[numero] for numero in numeros]

        linhas = []

        medidas = [
            ("Altura", lambda f: f["altura_m"], " m"),
            ("Peso", lambda f: f["peso_kg"], " kg"),
            ("HP", lambda f: f["stats"]["hp"], ""),
            ("Ataque", lambda f: f["stats"]["ataque"], ""),
            ("Defesa", lambda f: f["stats"]["defesa"], ""),
            ("Especial", lambda f: f["stats"]["especial"], ""),
            ("Velocidade", lambda f: f["stats"]["velocidade"], ""),
        ]

        for rotulo, valor, unidade in medidas:
            ordenadas = sorted(fichas, key=valor, reverse=True)

            texto = f"{ordenadas[0]['nome']} ({valor(ordenadas[0]):g}{unidade})"

            for anterior, atual in zip(ordenadas, ordenadas[1:]):
                sinal = "=" if valor(atual) == valor(anterior) else ">"
                texto += f" {sinal} {atual['nome']} ({valor(atual):g}{unidade})"

            maiores = [
                f["nome"]
                for f in ordenadas
                if valor(f) == valor(ordenadas[0])
            ]

            linhas.append(
                f"{rotulo}, do maior para o menor: {texto}. "
                + (
                    f"Maior {rotulo}: {maiores[0]}."
                    if len(maiores) == 1
                    else f"Empate no maior {rotulo}: {', '.join(maiores)}."
                )
            )

        # Quantas vezes um é mais alto/pesado que o outro, em relação
        # ao primeiro (o detectado na imagem).
        base = fichas[0]

        for outra in fichas[1:]:
            for rotulo, chave, unidade in (
                ("alto", "altura_m", "m"),
                ("pesado", "peso_kg", "kg"),
            ):
                maior, menor = sorted(
                    (base, outra),
                    key=lambda f: f[chave],
                    reverse=True,
                )

                if menor[chave]:
                    razao = maior[chave] / menor[chave]

                    linhas.append(
                        f"{maior['nome']} ({maior[chave]:g} {unidade}) é "
                        f"{razao:.1f} vezes mais {rotulo} que "
                        f"{menor['nome']} ({menor[chave]:g} {unidade})."
                    )

        # Batalhas em frases curtas, nas duas direções ("X vence quem?" e
        # "quem vence X?"). Frases longas com vários multiplicadores
        # confundiam o modelo quando havia 3 ou mais Pokémon.
        def motivo(atacante, defensor):
            valor, tipo = melhor_ataque(atacante, defensor)
            return (
                f"{atacante['nome']} usa golpes do tipo {TIPOS_PT[tipo]}, "
                f"que causam {valor:g}x em {defensor['nome']}"
            )

        def leva_vantagem(atacante, defensor):
            return melhor_ataque(atacante, defensor)[0] > melhor_ataque(defensor, atacante)[0]

        for ficha in fichas:
            vence = [
                outra for outra in fichas
                if outra is not ficha and leva_vantagem(ficha, outra)
            ]
            perde = [
                outra for outra in fichas
                if outra is not ficha and leva_vantagem(outra, ficha)
            ]

            linhas.append(
                f"Batalha - {ficha['nome']} leva vantagem de tipo sobre: "
                + (
                    "; ".join(
                        f"{outra['nome']} ({motivo(ficha, outra)})"
                        for outra in vence
                    )
                    or "nenhum dos outros"
                )
                + "."
            )

            linhas.append(
                f"Batalha - quem leva vantagem de tipo sobre {ficha['nome']}: "
                + (
                    "; ".join(
                        f"{outra['nome']} ({motivo(outra, ficha)})"
                        for outra in perde
                    )
                    or "nenhum dos outros"
                )
                + "."
            )

        return "\n".join(linhas)

def montar_contexto(ficha, resumida=False):
    # Texto em português com os dados da ficha, para ser colocado no
    # prompt do LLM (RAG). A descrição da Pokédex e os nomes dos golpes
    # estão em inglês, como no jogo original. resumida=True deixa só os
    # dados básicos (usado para as evoluções, que raramente são o
    # assunto e só aumentariam o tempo de leitura do prompt).
    def nomes(referencias):
        return ", ".join(
            f"{referencia['nome']} (nº {referencia['numero']})"
            for referencia in referencias
        )

    stats = ficha["stats"]

    linhas = [
        f"Pokémon: {ficha['nome']} (nº {ficha['numero']} da Pokédex, "
        "Primeira Geração)",
        f"Classificação: {ficha['classificacao']}",
        "Tipo: " + " / ".join(
            TIPOS_PT.get(tipo, tipo) for tipo in ficha["tipos"]
        ),
        f"Altura: {ficha['altura_m']} m",
        f"Peso: {ficha['peso_kg']} kg",
        f"Lendário: {'sim' if ficha['lendario'] else 'não'}",
        (
            "Stats base: "
            f"HP {stats['hp']}, Ataque {stats['ataque']}, "
            f"Defesa {stats['defesa']}, Especial {stats['especial']}, "
            f"Velocidade {stats['velocidade']}"
        ),
        "Evolui de: " + (nomes(ficha["evolui_de"]) or "não evolui de nenhum Pokémon"),
        "Evolui para: " + (nomes(ficha["evolui_para"]) or "não evolui para nenhum Pokémon"),
    ]

    if resumida:
        return "\n".join(linhas)

    linhas.append(
        "Golpes aprendidos por nível (Pokémon Red/Blue): " + ", ".join(
            f"{golpe['nome']} (nível {golpe['nivel']})"
            for golpe in ficha["golpes"]
        )
    )

    # Vantagens e fraquezas calculadas pela tabela de tipos da Gen 1
    # (backend/tipos.py), para o LLM não depender da memória.
    def tipos_pt(tipos, multiplicadores=None):
        return ", ".join(
            TIPOS_PT[tipo]
            + (f" ({multiplicadores[tipo]:g}x)" if multiplicadores else "")
            for tipo in tipos
        ) or "nenhum"

    defesa = resumo_defensivo(ficha["tipos"])

    linhas.append(
        "Tabela de tipos da Primeira Geração. Recebe dano aumentado de: "
        + tipos_pt(defesa["fraquezas"], defesa["fraquezas"])
        + ". Recebe dano reduzido de: "
        + tipos_pt(defesa["resistencias"], defesa["resistencias"])
        + ". Imune a: "
        + tipos_pt(defesa["imunidades"])
        + "."
    )

    for tipo in ficha["tipos"]:
        linhas.append(
            f"Golpes do tipo {TIPOS_PT[tipo]} causam dano dobrado em: "
            + tipos_pt(super_efetivo_contra(tipo))
            + "."
        )

    for versao, golpes in ficha["golpes_outras_versoes"].items():
        linhas.append(
            f"Golpes por nível diferentes em Pokémon {versao}: "
            + ", ".join(
                f"{golpe['nome']} (nível {golpe['nivel']})"
                for golpe in golpes
            )
        )

    linhas.append(
        f"Descrição da Pokédex (em inglês): {ficha['descricao']}"
    )

    return "\n".join(linhas)
